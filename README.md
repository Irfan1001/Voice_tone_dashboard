# Voice Tone Dashboard

Deterministic call-audio analysis: one audio file in, a fixed nine-field JSON
verdict out. Fully local, no LLMs, byte-identical on repeated runs.

```json
{
  "emotional_tone": "upset",
  "emotional_intensity": "high",
  "background_noise_present": false,
  "background_noise_type": "",
  "background_noise_severity": "none",
  "audio_quality": "clear",
  "speaker_overlap_present": false,
  "long_silence_present": false,
  "confidence": 0.69
}
```

Ships with a CLI, an HTTP API with batch ZIP upload and API-key auth, and a
self-contained browser dashboard.

## Run it

```bash
cd voice_analytics
echo 'HF_TOKEN=hf_xxx' > .env     # required: two pyannote models are gated
./run.sh                          # builds the venv, starts the API, opens the dashboard
```

**[Full documentation → `voice_analytics/README.md`](voice_analytics/README.md)** —
architecture, model choices and the evidence for them, measured results with
confusion matrices, performance and sizing, and known limitations.

## How it works

```
Audio → Preprocessor → Feature Extraction → Predictors → Rule Engine → JSON
                       VAD, DSP,            measure       decides
                       overlap, roles       only          (all thresholds here)
```

