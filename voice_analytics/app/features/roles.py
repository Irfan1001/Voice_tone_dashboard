"""Customer isolation: split the two parties, then decide which one is the customer.

`emotional_tone` is defined as the *customer's* tone, so emotion must be scored on
the customer's turns rather than averaged over both speakers.

1. Separate the speakers with `num_speakers=2` passed explicitly. Left to choose,
   pyannote over-splits one voice into 4-5 clusters on narrowband dual-mono
   telephony, where short turns give weak speaker embeddings.
2. Decide which speaker is the customer from WHAT IS SAID: a tiny ASR model
   transcribes the opening seconds and both speakers are scored against generic
   contact-centre script patterns.

Deliberately unused as role signals: emotion (circular - it picks the speaker who
confirms the hypothesis, then reports their emotion), "first speaker = agent" (a
customer backchannel can precede the greeting), and names/companies (they change
per agent and client; the script shape does not). Audio-only cues were measured
and rejected - dual-mono summing destroys the per-leg codec fingerprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# Things only an agent says. A role decision REQUIRES at least one of these.
STRONG_AGENT_PATTERNS: tuple[str, ...] = (
    r"thank(s)? (you )?for calling",
    r"how (can|may) i (help|assist)",
    r"i('m| am) \w+ from",
    r"you('re| are) (speaking|through) (with|to)",
    r"(can|may|could) i (have|get|take) your",
    r"is there anything else",
    r"(place|put) you on hold",
    r"what type of (service|issue|problem)",
)

# Agent-ish, but a customer can say them too (they are asked their name constantly).
# These may add margin; they can never carry a decision on their own, because a
# customer who out-scores the agent inverts the roles silently.
WEAK_AGENT_PATTERNS: tuple[str, ...] = (
    r"my name is",
    r"this is \w+ speaking",
    r"let me (check|look|see|pull)",
    r"i('ll| will| would)? ?(be )?happy to",
    r"your account",
    r"bear with me",
)

AGENT_PATTERNS: tuple[str, ...] = STRONG_AGENT_PATTERNS + WEAK_AGENT_PATTERNS

# The customer side. Scoring both and taking the difference stops a speaker who
# plainly sounds like a customer winning on one incidental agent-ish phrase.
CUSTOMER_PATTERNS: tuple[str, ...] = (
    r"i('m| am)? ?calling (about|because|regarding)",
    r"i have a (problem|question|issue)",
    r"i (want|need) to",
    r"it('s| is)? not working",
    r"i('ve| have) been (waiting|charged|trying)",
    r"can you (help|tell) me",
    r"i was (told|charged|expecting)",
    r"my (car|account|order|appointment|phone|bill)",
)


@dataclass(frozen=True)
class ScriptScore:
    """How much one speaker's opening sounds like an agent, and like a customer."""

    agent: int
    customer: int
    strong: int
    hits: tuple[str, ...] = ()

    @property
    def net(self) -> int:
        return self.agent - self.customer


@dataclass
class CustomerIsolation:
    """The outcome of role resolution. `customer_spans` drives emotion scoring."""

    customer_spans: list[tuple[float, float]] = field(default_factory=list)
    customer_label: str = ""
    agent_label: str = ""
    basis: str = "unavailable"
    margin: int = 0
    transcripts: dict[str, str] = field(default_factory=dict)
    scores: dict[str, int] = field(default_factory=dict)
    strong: dict[str, int] = field(default_factory=dict)
    n_speakers: int = 0
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.customer_spans) and self.basis != "unavailable"

    def to_evidence(self) -> dict:
        return {
            "customer_isolated": self.resolved,
            "role_basis": self.basis,
            "role_margin": self.margin,
            "customer_speaker": self.customer_label,
            "agent_speaker": self.agent_label,
            "n_speakers": self.n_speakers,
            "opening_transcripts": self.transcripts,
            "agent_script_scores": self.scores,
            "strong_pattern_hits": self.strong,
            "customer_speech_s": round(
                sum(e - s for s, e in self.customer_spans), 2),
            "reason": self.reason,
        }


def score_text(text: str) -> ScriptScore:
    """Score one speaker's opening transcript against both script vocabularies."""
    strong_hits = [p for p in STRONG_AGENT_PATTERNS if re.search(p, text, re.I)]
    weak_hits = [p for p in WEAK_AGENT_PATTERNS if re.search(p, text, re.I)]
    cust_hits = [p for p in CUSTOMER_PATTERNS if re.search(p, text, re.I)]
    return ScriptScore(
        agent=len(strong_hits) + len(weak_hits),
        customer=len(cust_hits),
        strong=len(strong_hits),
        hits=tuple(strong_hits + weak_hits),
    )


def pick_customer(scores: dict[str, ScriptScore], min_margin: int
                  ) -> tuple[str, str, int]:
    """Choose (customer, agent, margin). Empty labels mean "refuse to decide".

    Both conditions must hold: the winner leads on net score by `min_margin`, and
    the winner matched at least one strong (agent-only) pattern. Refusing makes the
    caller score all speech and flag `customer_isolated: false` - visibly
    incomplete rather than confidently wrong.
    """
    if len(scores) < 2:
        return "", "", 0
    ranked = sorted(scores, key=lambda s: -scores[s].net)
    agent, other = ranked[0], ranked[1]
    margin = scores[agent].net - scores[other].net
    if margin < min_margin or scores[agent].strong < 1:
        return "", "", margin
    return other, agent, margin


class CustomerIsolator:
    """Diarizes into two speakers and resolves roles from the opening transcript."""

    def __init__(self, diarization_id: str, asr_id: str, token: str | None,
                 device: str = "cpu", seed: int = 0, opening_s: float = 20.0,
                 min_margin: int = 1):
        self.diarization_id = diarization_id
        self.asr_id = asr_id
        self.token = token
        self.device = device
        self.seed = seed
        self.opening_s = opening_s
        self.min_margin = min_margin
        self._dia = None
        self._asr = None
        self._proc = None
        self.reason = ""

    def _load(self) -> bool:
        if self._dia is not None:
            return True
        if not self.token:
            self.reason = "no HF token found (set HF_TOKEN); pyannote models are gated"
            return False
        try:
            import torch
            from pyannote.audio import Pipeline
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            torch.manual_seed(self.seed)
            dia = Pipeline.from_pretrained(self.diarization_id, token=self.token)
            if dia is None:
                self.reason = (f"pyannote returned no pipeline for "
                               f"{self.diarization_id}; terms probably not accepted")
                return False
            self._dia = dia.to(torch.device(self.device))
            self._proc = AutoProcessor.from_pretrained(self.asr_id)
            asr = AutoModelForSpeechSeq2Seq.from_pretrained(self.asr_id)
            asr.eval()
            self._asr = asr.to(torch.device(self.device))
            return True
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            return False

    def _transcribe(self, x: np.ndarray, sr: int) -> str:
        if len(x) < int(0.3 * sr):
            return ""
        import torch

        feats = self._proc(x, sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            ids = self._asr.generate(
                feats.input_features.to(torch.device(self.device)),
                max_new_tokens=110)
        return self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def run(self, x: np.ndarray, sr: int) -> CustomerIsolation:
        """Never raises and never guesses: an unresolved role returns
        `basis="unavailable"` with the reason attached.
        """
        if not self._load():
            return CustomerIsolation(reason=self.reason)
        import torch

        try:
            wav = {"waveform": torch.from_numpy(
                np.ascontiguousarray(x, dtype=np.float32)).unsqueeze(0),
                "sample_rate": sr}
            with torch.inference_mode():
                result = self._dia(wav, num_speakers=2)
            ann = getattr(result, "speaker_diarization", result)

            turns: dict[str, list[tuple[float, float]]] = {}
            for turn, _, label in ann.itertracks(yield_label=True):
                turns.setdefault(str(label), []).append(
                    (float(turn.start), float(turn.end)))
            if len(turns) < 2:
                return CustomerIsolation(
                    n_speakers=len(turns),
                    reason=f"diarization found {len(turns)} speaker(s); "
                           "role cannot be resolved on a single-speaker recording")

            transcripts, scores = {}, {}
            for spk, spans in turns.items():
                opening = [(s, min(e, self.opening_s))
                           for s, e in sorted(spans) if s < self.opening_s]
                if not opening:
                    transcripts[spk] = ""
                    scores[spk] = ScriptScore(0, 0, 0)
                    continue
                seg = np.concatenate(
                    [x[int(s * sr):int(e * sr)] for s, e in opening]) \
                    if opening else np.zeros(0, dtype=np.float32)
                text = self._transcribe(seg, sr)
                transcripts[spk] = text
                scores[spk] = score_text(text)

            customer, agent, margin = pick_customer(scores, self.min_margin)
            iso = CustomerIsolation(
                customer_label=customer, agent_label=agent, margin=margin,
                transcripts=transcripts,
                scores={k: v.net for k, v in scores.items()},
                strong={k: v.strong for k, v in scores.items()},
                n_speakers=len(turns),
            )
            if not customer:
                iso.reason = (
                    f"no speaker's opening matched the agent script by the required "
                    f"margin (best margin {margin}); emotion will cover all speech")
                return iso
            iso.customer_spans = sorted(turns[customer])
            iso.basis = "opening-transcript agent-script match"
            return iso
        except Exception as exc:
            return CustomerIsolation(reason=f"{type(exc).__name__}: {exc}")
