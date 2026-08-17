"""Classification metrics, computed locally rather than pulling in scikit-learn for
forty lines of arithmetic.

Confusion matrices are `cm[true][pred]` - rows truth, columns predictions.
Transposing that silently swaps precision and recall, so tests assert it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ClassMetrics:
    label: str
    support: int          # how many true examples of this class
    predicted: int        # how many times the model said this class
    correct: int
    precision: float
    recall: float
    f1: float


@dataclass
class Report:
    labels: list[str]
    accuracy: float
    macro_f1: float
    per_class: list[ClassMetrics]
    confusion: list[list[int]]         # confusion[true][pred]
    n: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "labels": self.labels,
            "per_class": [
                {"label": c.label, "support": c.support, "predicted": c.predicted,
                 "correct": c.correct, "precision": round(c.precision, 4),
                 "recall": round(c.recall, 4), "f1": round(c.f1, 4)}
                for c in self.per_class
            ],
            "confusion_true_by_pred": self.confusion,
            "notes": self.notes,
        }


def confusion_matrix(true: list[str], pred: list[str], labels: list[str]
                     ) -> list[list[int]]:
    """`cm[true][pred]`. Values outside `labels` are ignored, not silently bucketed."""
    if len(true) != len(pred):
        raise ValueError(f"length mismatch: {len(true)} true vs {len(pred)} pred")
    index = {lab: i for i, lab in enumerate(labels)}
    cm = [[0] * len(labels) for _ in labels]
    for t, p in zip(true, pred):
        if t in index and p in index:
            cm[index[t]][index[p]] += 1
    return cm


def evaluate(true: list[str], pred: list[str], labels: list[str]) -> Report:
    """Accuracy, macro F1 and per-class precision/recall/F1.

    Macro F1 averages over EVERY label, including ones never predicted - those score 0
    and drag the average down. Averaging only over predicted classes is how a model
    that ignores a class entirely still looks competent.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    cm = confusion_matrix(true, pred, labels)
    scored = [(t, p) for t, p in zip(true, pred) if t in labels and p in labels]
    n = len(scored)
    correct_total = sum(1 for t, p in scored if t == p)

    per_class: list[ClassMetrics] = []
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        support = sum(cm[i])                       # row: all true examples
        predicted = sum(cm[r][i] for r in range(len(labels)))   # column
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom else 0.0
        per_class.append(ClassMetrics(lab, support, predicted, tp,
                                      precision, recall, f1))

    notes = []
    unseen = [c.label for c in per_class if c.support == 0]
    if unseen:
        notes.append(f"labels with no examples in the data: {unseen} "
                     "(they score F1 0 and still count in the macro average)")
    never = [c.label for c in per_class if c.support > 0 and c.predicted == 0]
    if never:
        notes.append(f"labels the model NEVER predicted: {never}")
    dropped = len(true) - n
    if dropped:
        notes.append(f"{dropped} row(s) ignored: value outside the label set")

    return Report(
        labels=list(labels),
        accuracy=correct_total / n if n else 0.0,
        macro_f1=sum(c.f1 for c in per_class) / len(per_class),
        per_class=per_class,
        confusion=cm,
        n=n,
        notes=notes,
    )


def format_confusion(cm: list[list[int]], labels: list[str]) -> str:
    """Rows are truth, columns predictions - stated in the header, not assumed."""
    w = max(len(l) for l in labels) + 2
    head = " " * w + "".join(f"{l[:9]:>10}" for l in labels)
    lines = ["rows = truth, columns = predicted", head]
    for lab, row in zip(labels, cm):
        lines.append(f"{lab:<{w}}" + "".join(f"{v:>10}" for v in row))
    return "\n".join(lines)


def format_report(rep: Report, title: str) -> str:
    out = [f"{title}", "=" * len(title),
           f"n={rep.n}   accuracy={rep.accuracy:.1%}   macro F1={rep.macro_f1:.3f}", ""]
    out.append(f"{'label':<14}{'support':>8}{'pred':>7}{'correct':>9}"
               f"{'prec':>7}{'recall':>8}{'F1':>7}")
    for c in rep.per_class:
        out.append(f"{c.label:<14}{c.support:>8}{c.predicted:>7}{c.correct:>9}"
                   f"{c.precision:>7.2f}{c.recall:>8.2f}{c.f1:>7.3f}")
    out.append("")
    out.append(format_confusion(rep.confusion, rep.labels))
    for note in rep.notes:
        out.append(f"NOTE: {note}")
    return "\n".join(out)
