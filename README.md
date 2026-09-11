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

## 1b. A note on per-user label-column variation (including "STANDING_AND_MOVING")

The brief lists 7 main-activity/posture classes, and **all 7 are real,
ExtraSensory-published labels** (`STANDING_AND_MOVING` is documented on
ExtraSensory's own site as present for 58 of 60 users, 29,754 examples).
An earlier version of this project's docs incorrectly claimed this label
was entirely absent from the public release — that conclusion was based on
inspecting only one real user's file, which happens to be one of a small
number of users whose file lacks that particular column (each user's
`.features_labels.csv` can have a slightly different set of label columns,
since a label with zero examples for that user isn't included in their
file's header).

To handle this correctly, `data_io.py::get_primary_activity_label` looks up
each of the 7 classes defensively per user: if a given user's file is
missing one of the label columns, that class is simply treated as "never
applies to this user" rather than raising an error. This means:
- `scripts/train.py` trains on the full 7-class label space across all 60
  users without crashing on schema differences between users;
- questions about any of the 7 classes (including "standing and moving")
  are answered normally through the regular Task 1-4 pipeline, with no
  special-casing.

## 2. Where to put the downloaded files

You can get the primary data **either** as one bulk archive **or** as many
per-user archives — both work with this codebase:

- **Bulk**: `ExtraSensory.per_uuid_features_labels.zip` from
  http://extrasensory.ucsd.edu/#download (~215 MB, all ~60 users at once).
- **Per-user**: one zip per user, e.g.
  `00EABED2-271D-49D8-B599-1D4A09240601_features_labels_csv.zip` — each
  contains a single plain `.csv` (not gzip-compressed) for that one user.
  If this is what you have, download/collect **all ~60 of them**.

For the cross-validation partition, download the official 5-fold split —
you may see it named `cv5Folds.zip` or `Cross_validation_partition.zip`;
both are accepted.

**You do not need to unzip anything by hand.** Place whichever zip file(s)
you have directly into `data/downloads/` (mixing bulk + per-user is fine),
then run `python -m scripts.setup_data` (see Quick start below), which:

- extracts every recognized primary-data zip's contents into
  `data/raw/primary_data_files/`, flattening away any internal folder
  structure and keeping only `<UUID>.features_labels.csv` or `.csv.gz`
  files;
- extracts the CV partition zip's contents into `data/raw/cv5Folds/`,
  flattening away its internal folder (e.g. `Cross validation partition/`)
  so you end up with the fold `.txt` files directly inside `cv5Folds/`;
- validates that ~60 users and all 5 folds' files are present and reports
  clearly if anything is missing.

After a successful run, the layout looks like:

```text
ask-the-sensors/
└── data/
    ├── downloads/                                    <- your original zip(s), untouched
    │   ├── ExtraSensory.per_uuid_features_labels.zip  <- if using the bulk archive
    │   ├── <UUID-1>_features_labels_csv.zip           <- if using per-user archives
    │   ├── <UUID-2>_features_labels_csv.zip
    │   ├── ...
    │   └── Cross_validation_partition.zip             <- (or cv5Folds.zip)
    └── raw/
        ├── primary_data_files/                        <- populated by setup_data.py
        │   ├── <UUID-1>.features_labels.csv           <- plain csv (per-user zips)
        │   ├── <UUID-2>.features_labels.csv.gz         <- or gzip csv (bulk zip)
        │   ├── ...                                     (60 files total)
        └── cv5Folds/                                   <- populated by setup_data.py
            ├── fold_0_test_android_uuids.txt
            ├── fold_0_test_iphone_uuids.txt
            ├── fold_0_train_android_uuids.txt
            ├── fold_0_train_iphone_uuids.txt
            ├── fold_1_test_android_uuids.txt
            ├── ...                                     (5 folds x 4 files = 20 files total)
```

`src/ask_the_sensors/data_io.py` reads either `.csv` or `.csv.gz` per-user
files transparently, so it doesn't matter which download route you used.

## 3. Quick start

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
export OLLAMA_HOST="http://10.14.20.23:11434"
export OLLAMA_MODEL="qwen3:8b"

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

Every step is independently re-runnable and idempotent (re-running does not
require deleting previous outputs; they are overwritten deterministically
given the same random seed, set in `configs/config.yaml`).

## 3b. Optional: smoke-test the pipeline before downloading the real dataset

`tests/make_synthetic_dataset.py` generates a small (12-user) synthetic
dataset in the exact ExtraSensory file format and directory layout. This
lets you verify your environment and the full pipeline work correctly
*before* waiting on the 215 MB real download:

```bash
python tests/make_synthetic_dataset.py   # writes synthetic data into data/raw/...
python -m scripts.train
python -m scripts.evaluate
```

This does not call `scripts.setup_data` (there is nothing to unzip), and it
overwrites whatever is currently in `data/raw/primary_data_files/` and
`data/raw/cv5Folds/` — **delete the synthetic files before placing your
real downloaded data**, or simply re-run `python -m scripts.setup_data`
afterwards, which will re-extract the real archives over the synthetic
ones. The unit tests (`pytest`) do not need this script; they use their own
small in-memory fixtures.

## 3c. Raw sensor-stream input (the brief's stated system input format)

The brief specifies the system's input as *"a timestamped, multivariate
time series from a triaxial accelerometer... and a triaxial
gyroscope... Resample every sensor stream to 25 Hz before processing."*
`scripts/ask.py` and `scripts/answer_batch.py` both accept a recording in
exactly that form, as an alternative to an ExtraSensory-corpus UUID:

```bash
python -m scripts.ask --raw-stream recording.csv --recording-id patient-042 \
    --question "Did the user lie down for a prolonged period?"
```

`recording.csv` must have a header row with these columns:

```
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```

- `timestamp`: seconds since the start of the recording (does not need to
  be evenly spaced -- see below).
- `acc_x/y/z`: accelerometer, in g (1.0 at rest on any axis combination,
  matching ExtraSensory's own `raw_acc` convention).
- `gyro_x/y/z`: gyroscope, in rad/s (matching ExtraSensory's `proc_gyro`
  convention).

Internally (`src/ask_the_sensors/raw_stream_features.py`), the stream is:

1. **Genuinely resampled to 25 Hz** by linear interpolation onto a
   uniform grid (not merely labeled as 25 Hz -- real per-sample
   resampling of real data). Gaps in the source longer than 5 seconds are
   left as `NaN` rather than interpolated across, so a real recording
   dropout isn't silently papered over with fabricated motion.
2. **Split into 60-second windows** (matching this project's existing
   `aggregation.window_seconds` convention, so the output slots directly
   into the same timeline/interval machinery used for ExtraSensory-corpus
   recordings).
3. **Converted into the same 26-feature-per-sensor schema**
   (`magnitude_stats`, `magnitude_spectrum`, `magnitude_autocorrelation`,
   `3d`) that ExtraSensory's own precomputed feature files provide, so a
   model trained on the ExtraSensory corpus can be applied to a raw
   stream unmodified -- no retraining needed to use this mode.

For `answer_batch.py`, use `"raw_stream_path"` (and optionally
`"recording_id"`) in place of `"uuid"` in a `questions.json` entry; see
that script's own docstring for the exact schema and an example mixing
both kinds of entries in one batch.

**Note on accuracy in this mode:** the feature-extraction formulas above
are this project's own from-scratch implementation of ExtraSensory's
*published* feature schema (Vaizman, Ellis & Lanckriet 2017), not a
byte-for-byte reproduction of their original (closed-source) extraction
code, so values will not be bit-identical to ExtraSensory's own for the
same underlying motion. They are validated to be quantitatively sound
(see `tests/test_pipeline.py::TestRawStreamFeatures` and the design
notes in `raw_stream_features.py`'s docstring) and discriminate the same
way ExtraSensory's own features do -- e.g. a stationary window's
accelerometer magnitude mean sits at ~1.0g, and periodic motion produces
a much stronger autocorrelation peak than static motion. Accuracy on a
model trained purely on ExtraSensory's own precomputed features will
still depend on how closely a given raw-stream recording's true sensor
characteristics (sampling behavior, mounting, exact unit conventions)
match ExtraSensory's, same as it would for any other cross-dataset
generalization question.

## 4. Troubleshooting

- **`FileNotFoundError` mentioning `data/downloads/...`**: you haven't
  placed the two required zip files yet, or placed them somewhere other
  than `data/downloads/`. See section 2.
- **`scripts.setup_data` reports "expected 60 users, found N"**: the
  primary-data zip you downloaded doesn't match the expected release, or
  didn't fully extract. Re-download from the URL printed by the script.
- **Ollama connection errors / "falling back to rule-based..." messages**:
  the pipeline is designed to keep working even if Ollama is unreachable
  (see `src/ask_the_sensors/llm_client.py` and `qa_engine.py`) — you'll get
  templated explanations instead of LLM-phrased ones, and Tier 1/2
  questions are entirely unaffected since they never call the LLM. Check
  that `OLLAMA_HOST`/`OLLAMA_MODEL` are exported correctly and that the
  Ollama server at that address is reachable from this machine (e.g.
  `curl $OLLAMA_HOST/api/tags`).
- **`scripts.evaluate` fails with a missing `cv_predictions.csv.gz`**: run
  `python -m scripts.train` first; `evaluate.py` consumes its output.
- **Training is slow**: `scripts.train` now trains three models per fold
  by default (`full`, `hierarchical`, `compressed`); the two
  gradient-boosted-tree variants (`full`, `hierarchical`) are the slower
  ones, especially after the deeper-tree hyperparameters in
  `configs/config.yaml` (`max_depth: 8`, `max_leaf_nodes: 63`,
  `n_estimators: 400`) needed to fit this dataset's real per-class overlap
  well — see `REPORT_NOTES.md` for why. For a quick check, restrict to
  just one model with `python -m scripts.train --models compressed`, or a
  single fold with `python -m scripts.train --folds 0`, or both together.

## 5. Repository layout

```
ask-the-sensors/
├── README.md                     <- you are here
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

## 6. Design summary (see `REPORT_NOTES.md` for the full mapping to the brief)

- **Grounding discipline.** The LLM is never shown raw feature vectors or
  raw waveforms. It is shown only a compact, deterministic evidence table
  (start/end timestamps in seconds-from-start, duration, modality, channel
  list, and a short deterministic natural-language description of the
  signal pattern, e.g. "high acceleration magnitude variance, dominant step
  frequency 2.6 Hz"). This guarantees the `Explanation` field is always
  tied to something the earlier, non-LLM layers actually computed — exactly
  the constraint the brief places on the interface layer.
- **Time base.** All timestamps are reported in seconds from the start of
  the recording (the same convention as the brief's own examples), computed
  from each user's `timestamp` column (`t - t_min`).
- **7 classes.** `LYING_DOWN`, `SITTING`, `OR_standing` (standing in place),
  `STANDING_AND_MOVING`, `FIX_walking`, `FIX_running`, `BICYCLING` — the
  mutually-exclusive main-activity labels the brief specifies, mapped from
  the ExtraSensory cleaned label columns. Not every user's raw file has
  every label column (a class with zero examples for a user is sometimes
  omitted from that user's header entirely), so `data_io.py` looks up each
  class per-user defensively rather than assuming a fixed column set — see
  section 1b above.
- **Orientation-dependent features excluded by default.** ExtraSensory's
  phone-only sensing means orientation-dependent features
  (`raw_acc:3d:mean_x/y/z`, `raw_acc:3d:ro_xy/xz/yz`, and the `proc_gyro`
  equivalents) conflate phone placement with body posture — direct
  inspection of real data showed one user's `raw_acc:3d:mean_z` values were
  clearly bimodal within their own SITTING examples, degrading separation
  from LYING_DOWN. `preprocessing.py::_is_orientation_dependent` excludes
  these 12 (of 52) features by default in favor of the remaining
  orientation-invariant magnitude-based features, configurable via
  `preprocessing.exclude_orientation_dependent_features` in
  `configs/config.yaml`.
- **Derived cross-sensor features.** `preprocessing.py::add_derived_features`
  adds 4 engineered features on top of the raw ExtraSensory columns: the
  gyro-to-accel magnitude ratio (mean and std), the accelerometer spectral
  high/low energy ratio, and the accelerometer magnitude interquartile
  range. These specifically target the FIX_walking / FIX_running /
  BICYCLING confusion observed in this project's real evaluation results —
  those three activities have similar raw motion *intensity* but differ in
  motion *character* (rotational, heel-strike vs. smooth pedaling), which
  no single raw ExtraSensory feature captures on its own. Configurable via
  `preprocessing.add_derived_features`.
- **Three recognition-model operating points.** `recognition_model.py`
  trains: (1) `full`, a flat gradient-boosted-tree classifier over all
  classes at once; (2) `hierarchical`, a two-stage classifier that first
  splits examples into a coarse stationary-vs-moving group, then resolves
  the fine-grained class within that group with a second classifier — this
  directly targets the dominant confusion pattern seen in this project's
  real evaluation results (confusion is concentrated *within* the
  stationary group and *within* the moving group, almost never across
  them), a technique with precedent in the HAR literature (e.g. HHAR-Net);
  and (3) `compressed`, a small quantized MLP for the accuracy-vs-overhead
  edge comparison. All three additionally use SMOTE-based synthetic
  minority oversampling (Chawla et al. 2002) for rare classes (typically
  FIX_running, <1% of examples) rather than naive duplication, plus
  class-balanced loss weighting, to address ExtraSensory's severe class
  imbalance (SITTING:FIX_running is roughly 110:1 in a typical fold).
- **Edge efficiency / extra credit.** `efficiency.py` measures model size
  on disk, parameter count, peak memory during inference, and per-query
  latency for all three operating points; `figures.py` plots the Pareto
  accuracy-vs-overhead curve across all three (Figure 4 in the brief).