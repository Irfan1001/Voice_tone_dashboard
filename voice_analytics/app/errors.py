"""Typed errors. Every failure says what broke and why.

Design rule: nothing substitutes a fabricated value for a failed measurement. A
component that cannot run raises and the clip fails, because a fabricated
"neutral" is indistinguishable from a measured one in the output.

`code` is machine-readable so a batch report can group failures without parsing
prose.
"""

from __future__ import annotations


class PipelineError(Exception):
    code = "pipeline_error"

    def __init__(self, message: str):
        super().__init__(message)


class AudioLoadError(PipelineError):
    """Decoding failed. Carries a specific code set by the caller."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class VadUnavailableError(PipelineError):
    """Voice activity detection could not run.

    Not recoverable by design: VAD defines "speech" for the whole pipeline, so a
    cruder substitute would silently change the noise floor, both SNR estimates,
    the silence measurement and the emotion windows at once.
    """

    code = "vad_unavailable"


class IncompletePredictionError(PipelineError):
    """One or more predictors could not produce their fields.

    Raised instead of emitting placeholder values: the clip produces no output.
    """

    code = "incomplete_prediction"

    def __init__(self, unavailable: dict[str, str]):
        self.unavailable = unavailable
        detail = "; ".join(f"{slot}: {why}" for slot, why in unavailable.items())
        super().__init__(
            f"{len(unavailable)} predictor(s) unavailable, so the result would be "
            f"incomplete -> {detail}. Fix the cause; there is no degraded mode."
        )
