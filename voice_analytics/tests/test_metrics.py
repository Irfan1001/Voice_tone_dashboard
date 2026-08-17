"""Metric arithmetic, including the orientation mistakes that silently invert results."""

from __future__ import annotations

import pytest

from evaluation.metrics import confusion_matrix, evaluate

LABELS = ["neutral", "satisfied", "frustrated", "upset", "distressed"]


def test_confusion_is_true_by_pred_not_the_transpose():
    """cm[true][pred]. Transposing it swaps precision and recall everywhere, which
    is invisible in aggregate numbers - so pin the orientation explicitly."""
    cm = confusion_matrix(["upset"], ["neutral"], LABELS)
    assert cm[LABELS.index("upset")][LABELS.index("neutral")] == 1
    assert cm[LABELS.index("neutral")][LABELS.index("upset")] == 0


def test_perfect_predictions_score_one():
    rep = evaluate(LABELS, LABELS, LABELS)
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == 1.0


def test_a_class_the_model_never_predicts_scores_zero_and_lowers_macro_f1():
    """The failure macro F1 exists to catch: 4 of 5 classes perfect, one ignored."""
    true = ["neutral", "satisfied", "frustrated", "upset", "distressed"]
    pred = ["neutral", "satisfied", "frustrated", "upset", "upset"]
    rep = evaluate(true, pred, LABELS)
    dist = next(c for c in rep.per_class if c.label == "distressed")
    assert dist.f1 == 0.0
    assert dist.predicted == 0
    assert rep.macro_f1 < 1.0
    assert any("NEVER predicted" in n for n in rep.notes)


def test_precision_and_recall_are_not_interchangeable():
    """Two 'upset' truths, three 'upset' predictions, two right: recall 1.0,
    precision 2/3. If these come out equal the orientation is wrong."""
    true = ["upset", "upset", "neutral"]
    pred = ["upset", "upset", "upset"]
    rep = evaluate(true, pred, LABELS)
    upset = next(c for c in rep.per_class if c.label == "upset")
    assert upset.recall == pytest.approx(1.0)
    assert upset.precision == pytest.approx(2 / 3)
    assert upset.f1 == pytest.approx(2 * (2 / 3) / (1 + 2 / 3))


def test_macro_f1_counts_absent_labels_so_it_cannot_be_inflated():
    """Scoring only the classes present would report 1.0 here, which would be a lie
    about a 5-class problem."""
    rep = evaluate(["upset", "upset"], ["upset", "upset"], LABELS)
    assert rep.accuracy == 1.0
    assert rep.macro_f1 == pytest.approx(1 / 5)
    assert any("no examples" in n for n in rep.notes)


def test_values_outside_the_label_set_are_reported_not_silently_dropped():
    rep = evaluate(["upset", "bogus"], ["upset", "upset"], LABELS)
    assert rep.n == 1
    assert any("ignored" in n for n in rep.notes)


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="length mismatch"):
        confusion_matrix(["upset"], ["upset", "neutral"], LABELS)


def test_empty_label_set_is_an_error():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate(["upset"], ["upset"], [])


def test_support_and_predicted_come_from_row_and_column():
    true = ["upset", "neutral", "neutral"]
    pred = ["neutral", "neutral", "upset"]
    rep = evaluate(true, pred, LABELS)
    neutral = next(c for c in rep.per_class if c.label == "neutral")
    assert neutral.support == 2      # two true neutrals
    assert neutral.predicted == 2    # predicted neutral twice
    assert neutral.correct == 1
