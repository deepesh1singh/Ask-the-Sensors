# Ask the Sensors

Grounded, Explainable Activity Question Answering from Wearable Signals.
CS60055 Ubiquitous Computing — Hackathon Challenge 1, IIT Kharagpur.

This repository implements a full pipeline:

```
Sensor recording (accel + gyro features, 25 Hz canonical rate)
        │
        ▼
Preprocessing  (clean / align / resample the per-minute example stream)
        │
        ▼
Recognition layer   (compact 7-class activity classifier, engineered
                      features + gradient-boosted trees; also a small
                      MLP variant for the accuracy/overhead curve)
        │
        ▼
Aggregation layer    (per-window predictions -> activity intervals,
                      durations, counts, transitions, evidence spans)
        │
        ▼
Interface layer      (natural-language question -> structured operation
                      over the timeline -> answer, formatted and grounded
                      by a local LLM served through Ollama, qwen3:8b)
        │
        ▼
Structured answer (Answer / Activity-Event / Evidence / Explanation)
```

The recognition + aggregation layers are **not** LLM-based: they are a
deterministic, auditable ML pipeline over the ExtraSensory features. The LLM
(via Ollama) is used **only** in the final interface layer, to parse the
natural-language question into a structured query over the already-computed
evidence table, and to phrase the `Explanation` field. The LLM never sees
raw sensor numbers — only the aggregated, verifiable evidence — so every
claim it makes is traceable back to real, deterministic signal processing.

## 1. Which datasets you need (and why)

The ExtraSensory dataset (http://extrasensory.ucsd.edu/) offers several
downloadable pieces. **You only need two of them**:

| # | Dataset | Needed? | Reason |
|---|---|---|---|
| 1 | **Primary data — features + labels** (per-user `<UUID>_features_labels_csv.zip`, or the single bulk `ExtraSensory.per_uuid_features_labels.zip`, ~215 MB total) | **Yes** | Contains, per user, one CSV row per recorded minute with ~225 precomputed features from the phone accelerometer (`raw_acc:*`), phone gyroscope (`proc_gyro:*`), and other sensors, plus ~51 cleaned ground-truth label columns and a `timestamp` column. All 7 main-activity/posture classes the brief lists have a real label column in the public release (`label:LYING_DOWN`, `label:SITTING`, `label:OR_standing`, `label:STANDING_AND_MOVING`, `label:FIX_walking`, `label:FIX_running`, `label:BICYCLING`) — see section 1b below for a note on per-user label-column variation. This is sufficient for all four QA tiers. |
| 2 | **Cross-validation partition** (`cv5Folds.zip`, also seen distributed as `Cross_validation_partition.zip`) | **Yes** | The official, paper-comparable, user-independent 5-fold split (`fold_i_{train,test}_{iphone,android}_uuids.txt`). Required so that no user appears in both train and test within a fold, and so results are comparable to Vaizman et al. 2017. |
| 3 | Raw accelerometer (`ExtraSensory.raw_measurements.raw_acc.zip`, ~6.1 GB) | **No** | Not needed. The precomputed `raw_acc:*` features in dataset #1 already summarize this signal (magnitude statistics, spectral peaks, autocorrelation, etc.) at the same time resolution the QA tasks operate on (one-minute examples). Re-deriving these from 40 Hz raw waveforms would cost ~30x the disk/bandwidth for no accuracy benefit at this task granularity, and is explicitly *not* what the brief's own suggested pipeline does (features -> classifier -> timeline). |
| 4 | Raw gyroscope (`ExtraSensory.raw_measurements.proc_gyro.zip`, ~8.7 GB) | **No** | Same reasoning as #3 — `proc_gyro:*` features are already present in dataset #1. |

If you already downloaded the two large raw-measurement archives, you do not
need to place them anywhere for this code to run; it never reads them.

## 2. Quick start

```bash
# 0. from the repository root
cd ask-the-sensors

# 1. create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. place ALL the zip file(s) you downloaded/collected here (bulk archive,
#    or all ~60 per-user archives, plus the CV partition archive):
mkdir -p data/downloads
cp /path/to/*.zip data/downloads/

# 3. unpack them into the expected locations (idempotent; re-run any time
#    you add more per-user zips)
python -m scripts.setup_data

# 4. configure the LLM connection (defaults shown; override if needed)
export OLLAMA_HOST=" "
export OLLAMA_MODEL=" "

# 5. run the full pipeline: preprocessing -> features -> train -> evaluate -> figures
python -m scripts.run_all

# 6. ask a question interactively against a chosen user's recording
python -m scripts.ask --uuid <UUID> --question "How long was the user walking?"

# 6b. ...or against a raw accelerometer+gyroscope stream (the brief's own
#     stated input format -- see section 3c below)
python -m scripts.ask --raw-stream path/to/recording.csv \
    --question "How long was the user walking?"

# 7. run the evaluation-time batch QA server used by graders (questions.json
#    entries may use "uuid" or "raw_stream_path" -- see section 3c)
python -m scripts.answer_batch --questions path/to/questions.json --out outputs/answers/answers.json
```

## 3. Repository layout

```
ask-the-sensors/
├── README.md                    
├── requirements.txt
├── configs/
│   └── config.yaml                <- all paths, hyperparameters, LLM config
├── data/
│   ├── downloads/                 <- put the two zip files here (gitignored)
│   └── raw/
│       ├── primary_data_files/    <- unzipped per-user csv.gz (gitignored)
│       └── cv5Folds/              <- unzipped fold uuid txt files (gitignored)
├── src/ask_the_sensors/
│   ├── config.py                  <- config loader + Ollama settings
│   ├── data_io.py                 <- reading per-user csv.gz, label parsing
│   ├── preprocessing.py           <- cleaning, resampling to a canonical
│   │                                  25 Hz-equivalent per-minute grid,
│   │                                  gap handling, windowing
│   ├── cv_split.py                <- loads cv5Folds, enforces user-level
│   │                                  train/test separation
│   ├── recognition_model.py       <- activity classifiers: full (GBM),
│   │                                  hierarchical (two-stage, typically
│   │                                  most accurate), and compressed
│   │                                  (small quantized MLP) for the
│   │                                  efficiency curve; SMOTE oversampling
│   ├── aggregation.py             <- per-window predictions -> intervals,
│   │                                  durations, counts, transitions,
│   │                                  evidence spans (deterministic, no LLM)
│   ├── evidence.py                <- builds the Evidence block (timestamps,
│   │                                  modality, channels) for any answer
│   ├── qa_engine.py                <- the 4-tier question router: decides
│   │                                  which deterministic operation(s) over
│   │                                  the timeline answer the question, and
│   │                                  calls the LLM only to parse the NL
│   │                                  question type/args and to phrase the
│   │                                  final natural-language fields
│   ├── llm_client.py               <- thin wrapper around the Ollama HTTP
│   │                                  API (chat endpoint), retries, JSON
│   │                                  mode parsing
│   ├── output_format.py            <- renders the mandated 5-field answer
│   │                                  block (Answer / Activity-Event /
│   │                                  Evidence / Explanation) exactly as
│   │                                  specified in the brief
│   ├── metrics.py                   <- all accuracy/robustness/grounding
│   │                                  metrics described in the brief
│   ├── efficiency.py                <- model size (params + on-disk),
│   │                                  peak memory, and per-query latency
│   │                                  measurement utilities
│   ├── figures.py                   <- the five mandated figures
│   └── raw_stream_features.py       <- resamples a raw accel+gyro stream
│                                       (the brief's stated system input) to
│                                       25 Hz and extracts the same feature
│                                       schema the trained model expects; see
│                                       README section 3c
├── scripts/
│   ├── setup_data.py                <- unzips + validates the two archives
│   ├── build_features.py            <- preprocessing + feature engineering
│   ├── train.py                     <- trains full + compressed models,
│   │                                  5-fold CV using cv5Folds
│   ├── evaluate.py                  <- runs the mandated metrics + 5 figures
│   ├── run_all.py                   <- orchestrates the full pipeline
│   ├── ask.py                       <- interactive single-question CLI;
│   │                                  accepts --uuid or --raw-stream
│   └── answer_batch.py              <- evaluation-time batch runner:
│                                       (recording, questions) -> answers.json;
│                                       each question may use "uuid" or
│                                       "raw_stream_path"
├── tests/
│   ├── test_pipeline.py             <- unit tests using tiny in-memory
│   │                                    fixtures (does not need the real
│   │                                    dataset; validates output format,
│   │                                    aggregation logic, metric functions,
│   │                                    intent parsing, tier-specific answer
│   │                                    builders, and raw-stream resampling
│   │                                    / feature extraction)
│   └── make_synthetic_dataset.py    <- optional dev utility: generates a
│                                        12-user synthetic ExtraSensory-format
│                                        dataset for smoke-testing the full
│                                        pipeline before the real download
│                                        finishes (see README section 3b)
├── outputs/
│   ├── figures/                     <- the five required PNGs land here
│   ├── models/                      <- trained model artifacts (.joblib)
│   ├── logs/                        <- cv_predictions.csv.gz (out-of-fold
│   │                                    predictions), evaluation_results.json,
│   │                                    training_summary.json, and other
│   │                                    per-run diagnostics
│   └── answers/                     <- batch QA outputs (answers.json)
└── REPORT_NOTES.md                  <- pointers mapping code to each
                                        report section required by the brief
```
