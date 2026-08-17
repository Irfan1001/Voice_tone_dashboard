# Voice Analytics Pipeline

Analyses call audio and returns a fixed nine-field JSON verdict. Deterministic,
fully local, no LLMs — the same audio in always produces byte-identical output.

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

Ships with a CLI, an HTTP API with batch ZIP upload and API-key auth, and a browser
dashboard.

---

## Quick start

```bash
./run.sh
```

That creates the virtualenv, installs dependencies, starts the API on
<http://127.0.0.1:8000> and opens the dashboard. Drop any call recording onto it to
try it. `Ctrl-C` stops it.

First run takes a while — the dependencies include torch, and about 2 GB of model
weights download on the first request.

> **No audio ships with this repo.** The trial recordings are real customer calls, so
> they are gitignored rather than published. Put your own `wav`/`ogg`/`mp3`/`flac`
> files in `data/` (any name) and every command below works unchanged. The measured
> results further down refer to the three original trial calls.

### Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | `run.sh` finds it and builds `.venv` itself |
| **`HF_TOKEN`** | **Required.** Two pyannote models are gated — see below |
| `ffmpeg` | Optional. Only needed for `m4a`/`aac`; `brew install ffmpeg` |

### The HuggingFace token is not optional

Speaker overlap and customer isolation use two **gated** pyannote models. A token
alone is not enough — the terms must be accepted on each model page, using the same
account the token belongs to:

1. Accept the terms on all three pages:
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM>
2. Create a **read** token at <https://huggingface.co/settings/tokens>
3. Save it: `echo 'HF_TOKEN=hf_xxx' > .env`

Without it the pipeline **fails the clip** rather than guessing those fields. That is
deliberate — see [No fabricated values](#no-fabricated-values).

### Manual setup, if you would rather not use the script

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

echo 'HF_TOKEN=hf_xxx' > .env

python run_pipeline.py data/your_call.ogg     # CLI
uvicorn api.main:app --port 8000             # API + dashboard on :8000
pytest tests/ -q                             # 66 tests, no models needed
```

---

## Using it

### CLI

```bash
python run_pipeline.py data/your_call.ogg                  # the nine fields on stdout
python run_pipeline.py data/your_call.ogg --diagnostics    # evidence + latency to stderr
python run_pipeline.py data/your_call.ogg --out r.json --compact
python run_pipeline.py data/your_call.ogg -v               # per-stage logs
```

Only the result goes to stdout, so it pipes into `jq` or a file. Diagnostics,
warnings and errors go to stderr.

### HTTP API

```bash
export VOICE_API_KEYS="$(openssl rand -hex 24)"     # omit for open mode (local only)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /` | dashboard (`/docs` for the OpenAPI UI) |
| `GET /health` | liveness **and** readiness: `loading`, `ok`, or `error` with the cause |
| `POST /v1/analyze` | one audio file → **202** + job id |
| `POST /v1/batch` | a ZIP of audio → **202** + job id |
| `GET /v1/jobs` | recent jobs |
| `GET /v1/jobs/{id}` | state, progress, per-file results |
| `GET /v1/jobs/{id}/csv` | flat CSV of the nine fields |
| `DELETE /v1/jobs/{id}` | delete a finished job's audio (it is customer PII) |

```bash
curl -H "X-API-Key: $VOICE_API_KEYS" -F file=@data/your_call.ogg localhost:8000/v1/analyze
curl -H "X-API-Key: $VOICE_API_KEYS" localhost:8000/v1/jobs/<job_id>
```

**Submissions are asynchronous** because the pipeline runs slower than realtime; a
blocking endpoint would sit past most proxy timeouts. Clients get a job id and poll.

**Auth** is an `X-API-Key` header compared with `secrets.compare_digest`. With
`VOICE_API_KEYS` unset the service starts **OPEN**, warns in the logs and reports
`auth: OPEN` in `/health` — loud rather than closed, so local development works and
nobody exposes it unaware.

**ZIP uploads are treated as hostile input**: path-traversal and symlink entries
rejected, decompressed bytes counted as written (so a lying header cannot pass a
header-only check), 500 files / 200 MB per file / 2 GB expanded / 500 MB per request.
Each guard has a named test in `tests/test_uploads.py`.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | *unset* | **Required.** Gated pyannote models |
| `VOICE_API_KEYS` | *unset* | Comma-separated API keys. Unset means the API is **open** |
| `VOICE_UPLOAD_DIR` | system temp | Where uploads land |
| `VOICE_KEEP_UPLOADS` | `false` | Keep audio after processing. Off by default — it is customer PII |

### Docker

```bash
docker build --secret id=hf,env=HF_TOKEN -t voice-analytics .
docker run -p 8000:8000 -e VOICE_API_KEYS="$VOICE_API_KEYS" \
           -v voice-data:/data --memory=5g --cpus=1 voice-analytics
```

Weights are baked in at build time, so a fresh container serves immediately and a
HuggingFace outage cannot block a deployment. Size the memory limit for the **peak**
during model loading (4.0 GB measured), not the 2.9 GB steady state.

---

## Architecture

```
                    ┌──────────────────── one pass, shared by everything ─────────┐
Audio ─► Preprocessor ─► Feature Extraction                                       │
         decode              ├ Silero VAD          → one definition of "speech"    │
         downmix mono        ├ Acoustic DSP        → SNR, levels, defect metrics   │
         16 kHz              ├ Segmentation-3.0    → seconds of simultaneous speech│
         measure clipping    └ Diarization + ASR   → which speaker is the customer │
         normalise      └────────────────────────────────────────────────────────┬─┘
                                                                                │
                        ┌───────────── Predictors: MEASURE only ────────────────┘
                        ├ Emotion   wav2vec2 → arousal / dominance / valence
                        ├ Noise     AudioSet events (type) + DSP (presence, severity)
                        ├ Quality   DSP: clipping, dropouts, echo, level, bandwidth
                        ├ Silence   longest stretch that is non-speech AND quiet
                        └ Overlap   seconds of simultaneous speech
                                          │
                        ┌───────────── Rule Engine: DECIDES ─────────┐
                        │ the only module that emits schema values   │
                        └────────────────────┬───────────────────────┘
                                             ▼
                                    nine-field JSON + diagnostics
```

### Two invariants that explain most of the design

**1. Predictors measure. The rule engine decides.** A predictor never emits
`"emotional_tone": "upset"` — it emits arousal, dominance and valence. Every
threshold in the system therefore lives in one reviewable file (`app/config.py`) and
one testable module (`app/rules/engine.py`), instead of being scattered across five
models. It also means the mapping is unit-testable without loading a single model.

**2. <a name="no-fabricated-values"></a>No fabricated values.** Nothing substitutes a
placeholder for a failed measurement. A predictor that cannot run yields
`available=False` with a reason, and the clip produces **no output at all**:

```
$ python run_pipeline.py data/your_call.ogg          # exit code 2
{ "error": "incomplete_prediction",
  "message": "1 predictor(s) unavailable ... overlap: no HF token found" }
```

A fabricated `"neutral"` is indistinguishable from a measured one once it reaches the
JSON, so silent degradation would ship a confident-looking wrong answer. There is no
degraded mode and no secondary estimator standing in. Batch runs still isolate
failures per file, so one bad clip cannot end a 500-file job.

### One rule is deliberately absent

Nothing links `audio_quality` to any noise field, in either direction. A clip can
carry loud static while its speech stays perfectly intelligible. Any correlation
between the two must come from what the predictors independently measured.

### Model choices

| Field | Method | Why this one |
|---|---|---|
| `emotional_tone`, `emotional_intensity` | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` → arousal/dominance/valence | Dimensional maps onto the taxonomy directly (intensity essentially *is* arousal) and MSP-Podcast is natural conversational speech, not acted studio recordings. Best of five tested — see [the bake-off](#emotion-model-bake-off) |
| `background_noise_type` | `MIT/ast-finetuned-audioset-10-10-0.4593` | Naming noise is a classification problem. Hand-written spectral rules capped near 70% on 60 controlled clips: low-frequency dominance cannot separate hum from music from road rumble, and crest factor rates television 66 vs static 77 |
| `background_noise_present`, `background_noise_severity` | DSP: dual SNR estimators | Validated — 98% detection accuracy, 100% precision, zero severity inversions across 8 noise kinds × 5 SNRs |
| `audio_quality` | DSP: clipping, dropouts, cepstral echo, level, speech bandwidth | Reads only distortion-specific evidence, never an SNR |
| `speaker_overlap_present` | `pyannote/segmentation-3.0` (powerset) | The **segmentation** model, not the diarization pipeline — see [below](#why-segmentation-and-not-diarization) |
| `long_silence_present` | Silero VAD + a relative level gate | "Dead air" means absence of speech, so noise-filled stretches do not count |
| *customer isolation* | `speaker-diarization-3.1` + `whisper-tiny.en` on the opening 20 s | Emotion must be the **customer's**. Role comes from what is said, scored against generic contact-centre script patterns |

Two DSP details worth naming, because both were wrong in the obvious implementation:

- **Two SNR estimators are required.** Gap-based SNR runs *backwards* on speech-like
  background: a television counts as speech and leaves no gaps, measuring 11 dB at a
  true 30 dB SNR. Above a speech ratio of 0.85 the pipeline switches to
  min-statistics. Recall 81% → 97%.
- **Clipping is a flat top, not a loud sample,** and its ceiling is relative to the
  signal's own peak. Lossy codecs legitimately overshoot ±1.0 (the supplied Opus
  files peak at 1.004 and 1.107 while being labelled `clear`), and clipping followed
  by gain reduction leaves flat tops well below full scale. It must also be measured
  *before* normalisation, which erases the evidence.

### <a name="why-segmentation-and-not-diarization"></a>Why segmentation, not diarization, for overlap

In pyannote 4.x the diarization pipeline returns a **strictly exclusive** timeline —
it can never report two speakers at once. On audio containing 4.00 s of verified
two-voice speech, `speaker_diarization` and `exclusive_speaker_diarization` came back
byte-identical at 0.00 s simultaneous. Any non-zero "overlap" read from it is a
segment-boundary artifact.

`segmentation-3.0` is a **powerset** model: each 17 ms frame is classified as
"nobody" / "speaker 1" / … / "speakers 1+2", so simultaneous speech is a native
prediction made *before* the clustering step that collapses it. It is also cheaper —
no speaker embeddings, no clustering.

---

## Results

### The three supplied calls

The trial's own label file is **reference material, not the scoring target.** Three clips
cannot establish accuracy — each field can only score 0/3 to 3/3 — and at least two
of the three labels are demonstrably wrong:

| Evidence | Why the label cannot be trusted |
|---|---|
| `confidence` is **0.82 on all three** | That is the example value printed in the brief. A placeholder, so there is nothing to calibrate against |
| `call_003` → `long_silence_present: false` | The audio holds **~8 s of dead air**, confirmed by listening. We measure 7.35 s that is both non-speech and >25 dB below speech level |
| `call_002` → `neutral`, `call_003` → `satisfied` | call_002 measures **higher on all three emotion axes**. No monotonic rule can satisfy both labels at once |

So these are what the pipeline **measures**, not a score:

| Field | call_001 | call_002 | call_003 |
|---|---|---|---|
| `emotional_tone` | upset | distressed | satisfied |
| `emotional_intensity` | high | medium | medium |
| `background_noise_present` | false | true | true |
| `background_noise_type` | "" | television | sharp static |
| `background_noise_severity` | none | medium | medium |
| `audio_quality` | clear | clear | clear |
| `speaker_overlap_present` | false (0.31 s) | true (0.73 s) | true (1.95 s) |
| `long_silence_present` | false | false | **true** (7.35 s) |
| `confidence` | 0.69 | 0.64 | 0.72 |
| *arousal / dominance / valence* | .652 / .644 / .356 | .613 / .594 / .543 | .548 / .574 / .492 |
| *customer isolated* | ✅ margin 2 | ✅ margin 2 | ✅ margin 5 |

### Per-field status

| Field | vs the three labels | Confidence in the method |
|---|---|---|
| `emotional_intensity` | **3/3** | Grid-fitting 680 CREMA-D clips did not improve the thresholds (0.517 → 0.512), so they are already at the optimum |
| `background_noise_present` | **3/3** | DSP validated on 60 controlled clips |
| `background_noise_severity` | **3/3** | Zero severity inversions across 8 noise kinds × 5 SNRs |
| `audio_quality` | **3/3** | 10/10 injected defects detected; 33/33 noise-only clips stayed `clear` |
| `speaker_overlap_present` | **3/3** | Method is sound; **threshold fitted to n=3** |
| `background_noise_type` | 2/3 exact, **3/3 semantic** | Free text — "television" vs a label of "TV" is not a miss |
| `long_silence_present` | 2/3 vs labels, **3/3 vs the audio** | The label is wrong; verified by listening |
| `emotional_tone` | **2/3, and 2/3 is the ceiling here** | The weakest field — see [limitations](#limitations) |
| `confidence` | unscoreable | All three labels are the same placeholder value |

### <a name="emotion-model-bake-off"></a>Emotion model bake-off

Five models on the same 300 CREMA-D clips and the same actor-grouped split. Each
dimensional model got its **own** nearest-centroid mapping fitted on TRAIN actors —
judging a new model through thresholds fitted to another's score distribution would
measure resemblance, not quality. Scored against CREMA-D's voice-only **listening
study** (`VoiceVote`), not the actor's intent, because humans recover intent only
41.6% of the time from audio alone.

| Model | macro F1 | accuracy | stability κ | ms/clip | licence |
|---|---|---|---|---|---|
| emotion2vec_plus_base | 0.753 | 75.2% | +0.39 | 131 | **disqualified — see below** |
| **audeering** *(ships)* | **0.425** | 47.5% | +0.18 | 189 | cc-by-nc-sa-4.0 |
| voxprofile (WavLM MSP) | 0.376 | 39.6% | +0.32 | 329 | openrail weights / NOASSERTION code |
| superb (IEMOCAP) | 0.244 | 29.7% | **−0.73** | 110 | apache-2.0 |
| whisper-large-v3 | 0.229 | 25.7% | +0.07 | 5567 | apache-2.0 |

**Stability κ is chance-corrected self-consistency on two halves of one utterance.**
Raw agreement is unusable as a metric: `superb` predicted `upset` for 89 of 101
clips, so its 62.3% raw self-agreement sits *below* the 78.2% its own prior gives by
chance. Any future comparison must use the corrected figure.

Each dimensional model here was scored through a *fitted* nearest-centroid mapping, so
these numbers are not directly comparable to the shipping configuration measured in
the [confusion matrix](#confusion-matrix) below.

**Why emotion2vec is disqualified despite looking like a landslide.** It scored 0.753
and solved `frustrated` (0.792) and `distressed` (0.710) — both at F1 0.000 under the
fitted centroid mappings. But its training data is undisclosed, spans 40k hours, and
CREMA-D is among the most-used public SER corpora. Tested on the three
proprietary calls, which cannot be in any training set, it **collapses to `neutral`**
(23 of 25 windows on call_003) with *negative* chance-corrected consistency on two of
three calls. The CREMA-D result was memorisation.

**The remaining gain is not in the checkpoint.** On real calls `audeering` is also
unstable (33% window-to-window on call_003), valence self-correlates at **−0.01**
across two halves of one sentence, and no non-degenerate model exceeded κ 0.39. The
bottleneck is the representation, not the checkpoint.

Reproduce: `python -m evaluation.compare_emotion_models --limit 120`

### <a name="confusion-matrix"></a>Confusion matrix — `emotional_tone`

The **shipping** mapping (`arousal → dominance → valence`, thresholds exactly as in
`app/config.py`), scored against CREMA-D's human voice-only vote. 300 clips, 60 per
human-voted tone, so no class dominates the average:

```
                            predicted
truth          neutral satisfied frustrated  upset distressed   recall
neutral             19        11         11      5         14     0.32
satisfied           10        11          6     28          5     0.18
frustrated           5         4         24      8         19     0.40
upset                1         4          9     44          2     0.73
distressed           5         2         12     14         27     0.45

accuracy 41.7%   macro F1 0.398
```

| Class | precision | recall | F1 |
|---|---|---|---|
| `upset` | 0.44 | **0.73** | **0.553** |
| `distressed` | 0.40 | 0.45 | 0.425 |
| `frustrated` | 0.39 | 0.40 | 0.393 |
| `neutral` | 0.47 | 0.32 | 0.380 |
| `satisfied` | 0.34 | **0.18** | **0.239** |

**Against the human ceiling on the same 300 clips: humans agree with the actor's
intent 65.3% of the time (macro F1 0.627).** So the model reaches roughly two-thirds
of human performance on acted speech — not close to solved, and not noise either.

Read it for the failure modes rather than the diagonal:

- **`upset` is the one strong class.** High arousal plus high dominance is a genuinely
  separable signature, and it is also the class a contact centre most needs to catch.
- **`satisfied` is the weakest, and it fails toward `upset`** — 28 of 60 satisfied
  clips are called `upset`. Arousal decides the band first, so an animated positive
  customer lands in the high-arousal band, and only valence could rescue it. Valence
  is the axis that barely separates anything (ratio 0.71), so it does not.
- **`frustrated` and `distressed` are recovered but confusable** (F1 0.39 / 0.43),
  mostly with each other and with `neutral`. Note this contrasts with the bake-off
  table above, where both scored F1 0.000 for every dimensional model: that used a
  *fitted nearest-centroid* mapping per model. The hand-built arousal-first structure
  recovers these two classes where centroid fitting collapsed them.
- **`neutral` has the best precision but poor recall** — the band is narrow by design,
  because a wide neutral band swallows everything.

**Stability on the same clips: 38.3%** — the same utterance, split in half and scored
twice, gives the same verdict only 38% of the time. The most common flips are
`frustrated ↔ upset` (41), `distressed ↔ frustrated` (31) and `frustrated ↔ neutral`
(26). This is the real limit on the field, and it is a property of the representation
rather than the thresholds.

CREMA-D is **acted** speech, so this validates the mapping's *shape*, not production
accuracy. Reproduce: `python -m evaluation.human_agreement --limit 60`

### Performance

Measured single-threaded on CPU, warm process (excluding the one-off model load):

| Metric | With customer isolation | Without diarization |
|---|---|---|
| Compute per audio minute | **73.5 s** | **7.3 s** |
| vs realtime | **0.82× (slower)** | **8.3× faster** |
| Throughput per worker | ~49 audio-min/hour | ~490 audio-min/hour |
| Cost per audio minute¹ | $0.00082 | $0.00008 |

¹ at $0.04/core-hour, against a $0.003/audio-minute ceiling — 3.6× headroom even in
the slow configuration.

| Resource | Measured |
|---|---|
| RSS steady state | **2.9 GB** |
| RSS peak (during model load) | **4.0 GB** — size limits for this, not the steady state |
| Cold start | **16 s** |
| CPU | **1 core**, pinned |

**Where the time goes** (warm, 207 s of audio in 254 s wall):

| Stage | Share |
|---|---|
| Feature extraction — diarization, role ASR, VAD, segmentation, DSP | **95%** |
| Noise (AudioSet classifier) | 2.8% |
| Emotion | 1.5% |
| Preprocess | 0.4% |

**Diarization is the entire scaling lever.** It exists so `emotional_tone` is the
customer's rather than an average over both parties. Turning it off is a 10×
throughput gain at the cost of that guarantee — a product decision, not a tuning one.

**Scale horizontally.** One worker per container, no shared state.
`containers = (audio-minutes per hour) / 48`. Memory dominates, not CPU — ~5 GB per
container against 1 core — so pick RAM-weighted instances. A second worker in the
same process would load another 2.9 GB of identical weights and then queue behind the
single torch thread that determinism requires.

### Determinism

Verified: repeated runs produce byte-identical JSON. Enforced by fixed seeds,
`eval()` + `inference_mode`, no random cropping or sampling anywhere, and
single-threaded torch (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) because parallel
float reductions reorder and change low-order bits.

---

## Evaluation

```bash
pytest tests/ -q                                        # 66 tests, no models needed
python -m evaluation.evaluate data/ --out out/run1      # predictions + latency
python -m evaluation.evaluate data/ --labels labels.csv --out out/run1  # optional
python -m evaluation.human_agreement --limit 250        # vs human perception + stability
python -m evaluation.compare_emotion_models --limit 120 # the five-model bake-off
```

`evaluate.py` writes `predictions.csv`, `metrics.json`, `report.txt` and
`run_meta.json`. The last records every model id, every threshold that could change
an output, the seed and library versions: two runs with matching `run_meta.json` must
produce identical predictions, and if they do not, that is a determinism bug rather
than noise to average away.

The unit tests cover the parts that can be tested without models — role resolution
(the decision surface, including every case where it must *refuse* to decide), the
metrics arithmetic, and ZIP upload safety.

### Threshold provenance

Every threshold in `app/config.py` carries a tag. Read it before trusting the number:

| Tag | Meaning |
|---|---|
| `[principled]` | from signal-processing / telephony standards |
| `[validated]` | tested against ground truth (synthetic set or CREMA-D) |
| `[measured]` | informed by the 3 supplied calls — n=3, weak |
| `[arbitrary]` | a starting guess |

---

## <a name="limitations"></a>Limitations

**1. `emotional_tone` is the weak field, and 2/3 is the ceiling on these calls.**
The mapping keys on **arousal, then dominance**, with valence only breaking mid-band
ties. That follows from the field definitions:

| Tone | Definition | Dimensionally |
|---|---|---|
| `upset` | "clearly angry, agitated, strongly dissatisfied" | high arousal, **high** dominance |
| `distressed` | "highly emotional, overwhelmed, panicked, escalated" | high arousal, **low** dominance |
| `frustrated` | "annoyed, impatient, dissatisfied *without* strong anger" | negative valence, mid arousal |
| `satisfied` | "pleased, relieved, appreciative, clearly positive" | positive valence |
| `neutral` | "no clear positive or negative emotion" | mid valence |

Checked against 9 cases written straight from those definitions, this structure
matches **9/9**; a nearest-centroid mapping fitted for CREMA-D macro F1 matched only
**2/9** — it placed `distressed` at arousal 0.397, the opposite of "escalated". A
better benchmark F1 was buying worse specification compliance, so the fitted variant
was dropped.

Valence is demoted deliberately: its separation ratio is **0.71** against 1.62 for
arousal and 1.74 for dominance, and it self-correlates at **−0.01** across two halves
of one utterance. It carries less signal than noise.

*Why 2/3 is a ceiling, not a tuning failure:* call_002 measures higher than call_003
on all three axes yet carries the less-positive label. No monotonic rule over these
scores orders them correctly. Fixing this needs labelled call data or a better
representation — not different thresholds.

**2. The emotion model forbids commercial use.**
`audeering/...-msp-dim` is **CC-BY-NC-SA-4.0**, "research purpose only". It is the
best technical fit, and the licence is disclosed in `config.py` and in every
diagnostics dump so nobody deploys it unaware — but it is a procurement gate before
revenue traffic. audEERING sell a commercial licence; or swap the model.

**3. `speaker_overlap_present` works on a threshold fitted to n=3.** 0.5 s sits
mid-gap between the measured 0.31 s (negative) and 0.73 s (positive). Principled
measurement, weak threshold. Note also that a purpose-built synthetic set carrying
1–4 s of audited two-voice overlap measured **0.00 s** — the model does not respond
to two clean recordings summed to mono at equal level, so only real call audio can
validate this field.

**4. `confidence` is a deterministic formula, not calibrated.** Predictor confidence
weighted by fields owned, then penalised for fired rules, poor audio quality and very
short clips. It is a sensible ordering, but nothing has calibrated it against
outcomes, because all three supplied labels hold the same value.

**5. The supplied audio is dual-mono** (channel correlation +1.0000), so per-channel
speaker separation is unavailable and customer isolation has to be inferred from the
transcript. Stereo delivery would make it exact and free.

**6. Jobs are held in memory** and lost on restart. Deliberate for a demo service;
nothing in the API assumes the store is local, so Postgres or Redis is a contained
change. The service also speaks plain HTTP — terminate TLS in front of it — and has
no rate limiting, metrics or tracing.

**7. Uploaded audio is customer PII.** It is deleted after processing unless
`VOICE_KEEP_UPLOADS=true`, `DELETE /v1/jobs/{id}` removes a finished job's audio, and
nothing beyond filenames is logged. Set a retention policy on the volume before real
traffic.

---

## Getting labelled data

The single highest-value input is 50–100 labelled real calls. Two rules matter more
than the rest:

1. **Label the customer, not the agent,** for both emotion fields. If the agent is
   audibly annoyed and the customer is calm, the call is calm.
2. **Never copy a value you are unsure of.** An empty cell is usable data; a guess is
   not. (The existing `confidence: 0.82` on all three calls is the brief's example
   value, and it tells us nothing.)

Sample **randomly** across a date range rather than sending memorable calls —
emotionally dramatic calls are rare, and a set over-weighted toward them produces a
model that cries wolf on ordinary traffic. Include the awkward cases: hold music,
transfers, IVR preambles, three or more speakers, very short calls, calls where the
customer barely speaks. Double-label at least 20% independently; if two labellers
agree less than ~70% on `emotional_tone`, a definition needs sharpening before the
rest is labelled, because no model can be more consistent than its labels.

Deliver one CSV row per call, `name` matching the audio filename, booleans as
`true`/`false`, empty for "could not judge". **And if the telephony platform can
export two-channel audio with one party per channel, that removes a whole class of
error at zero cost.**

---

## Layout

```
app/
  config.py                every model id and threshold, each tagged with provenance
  schema.py                the output contract + the pipeline's dataclasses
  errors.py                typed failures; each names what broke and why
  env.py                   minimal .env loader
  signal_utils.py          run-length helpers
  preprocessing/           decode → mono → 16 kHz → normalise
  features/
    vad.py                 Silero VAD — one definition of "speech" for everyone
    acoustic.py            levels, dual SNR, spectral character, defect metrics
    overlap_detect.py      segmentation-3.0 powerset → simultaneous-speech seconds
    roles.py               diarization + opening-transcript role resolution
    extractor.py           assembles one AudioFeatures per clip
  models/                  predictor plugins + registry
  rules/engine.py          the only module that produces schema values
  pipeline/                AudioPipeline.run()
api/
  main.py                  FastAPI app, auth, endpoints
  jobs.py                  job store + the single sequential worker
  uploads.py               hostile-input handling for files and ZIPs
  dashboard.html           self-contained UI, no external assets
evaluation/
  evaluate.py              reproducible runner over a directory of audio
  metrics.py               confusion matrix, precision/recall/F1
  human_agreement.py       scoring against human perception + stability
  compare_emotion_models.py  the five-model bake-off
tests/
  test_rules.py            the rule engine: tone/intensity/noise/threshold decisions
  test_roles.py            role resolution, including when it must refuse to decide
  test_metrics.py          confusion-matrix and F1 arithmetic
  test_uploads.py          ZIP hostile-input guards
  test_csv_export.py       CSV rendering
                           66 tests total, no model downloads required
run.sh                     one-command local start
run_pipeline.py            the CLI
Dockerfile                 CPU image with weights baked in
```

## Licences

| Component | Licence |
|---|---|
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | **CC-BY-NC-SA-4.0 — non-commercial** |
| `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause |
| `pyannote/segmentation-3.0`, `speaker-diarization-3.1` | MIT (gated: terms must be accepted) |
| `openai/whisper-tiny.en` | Apache-2.0 |
| `snakers4/silero-vad` | MIT |
| CREMA-D (evaluation data only) | ODbL |

`evaluation/compare_emotion_models.py` fetches a NOASSERTION-licensed wrapper file
from GitHub at runtime for one comparison model. That is evaluation-only and is never
vendored into `app/`.
