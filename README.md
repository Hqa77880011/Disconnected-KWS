# Disconnected-KWS

Implementation of **Disconnected-mode personalization for ear-worn keyword spotting.**

## Install

Use Python 3.11 in an isolated environment. Run from the repository root:

```bash
python -m pip install -e .
```

## Data

Use [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) or the paper's spoken-word resources. CSV columns are `audio,label,speaker,split`, with relative WAV/FLAC paths, zero-based vocabulary IDs and roles `train,dev,anchor,support,accept,query,background`. Preserve original-utterance identity and keep enrollment speakers outside source training/development. The nominal enrollment uses 24 support and 16 distinct acceptance recordings plus 32 development anchors. Query labels never select candidates. Background rows are complete audio streams linked to their evaluated speaker. A workload CSV supplies `time_seconds,slack_ms,workspace_bytes`.

## Workflow

```bash
python -m earadapt prepare --input data/speech.csv
python -m earadapt train
python -m earadapt run
python -m earadapt evaluate
python -m earadapt plot
```

Main commands accept `--config config.json` and `--output runs/default`. Each output directory records `settings.json`. Keep independent seeds, datasets and conditions in separate directories.

## Configuration

`rank,amplitude` define the bounded low-rank logit adapter. `steps,adapter_learning_rate,anchor_weight,regularization` define candidate fitting. `acceptance_gain,anchor_tolerance` define finite-set activation. `quantum_ms,guard_ms,quantum_energy_j,check_energy_j,feature_energy_j,energy_j` define replay admission and reserved cost. `wake_threshold,wake_ticks,refractory_seconds,hop_seconds` define complete wake events. Additional optimizer, batch-size and model-width defaults are in `config.json`.

## Outputs

`base.pt` stores the frozen 64-dimensional speech detector. Accepted `adapter_<speaker>.pt` files store personalized states. `schedule.csv` logs admitted/deferred quanta and charged resources. `speakers.csv` includes rejected/postponed enrollment outcomes, query recall, cache/workspace bytes and background exposure. `metrics.csv` reports mean/lower-decile recall, false wakes/hour, acceptance/postponement, modeled resource charges and scheduling failures.

## Method and comparisons

Mono audio is resampled to 16 kHz and converted to log-mel input. The adapter is `base + delta*tanh(feature V U^T + b)`. Active, candidate and rollback copies are retained separately. Candidate SGD fits support CE plus frozen-anchor KL. Only acceptance/anchor checks can activate it. The scheduler replays declared quantum costs against foreground slack/space/energy; host wall time is recorded separately. False-wake rate is derived from full threshold/persistence/refractory events and actual audio duration.

Controls: `full`, `frozen`, `unbounded`, `no-anchor`, `no-acceptance`, `immediate`. Choose `--variant` and use a separate training/output directory.

```bash
python -m earadapt.compare --inputs "runs/*/metrics.csv" --reference full --metrics recall lower_decile_recall charged_energy_j --unit seed
```

Comparison averages rows within independent units before bootstrap intervals and paired Wilcoxon/Holm inference. For participant/client analyses choose those independent unit columns instead of counting recordings as replicates.

