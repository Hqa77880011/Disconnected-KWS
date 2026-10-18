# Disconnected-KWS

Implementation of **Disconnected-mode personalization for ear-worn keyword spotting.**

## Installation

Requires Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3 and SciPy 1.15.3. Verification covers Python syntax, JSON/TOML and README CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Use [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) or the paper's spoken-word resources. CSV columns are `audio,label,speaker,split`, with relative WAV/FLAC paths, zero-based vocabulary IDs and roles `train,dev,anchor,support,accept,query,background`. Preserve original-utterance identity and keep enrollment speakers outside source training/development. The nominal enrollment uses 24 support and 16 distinct acceptance recordings plus 32 development anchors. Query labels never select candidates. Background rows are complete audio streams linked to their evaluated speaker. Create the workload CSV at `data/workload.csv`, the path read from `config.json` (`workload`). Its columns are `time_seconds,slack_ms,workspace_bytes`; times are seconds, scheduling slack is milliseconds and workspace is bytes.

```bash
python -m earadapt prepare --input data/speech.csv --config config.json --output runs/full
```

## Training and inference

```bash
python -m earadapt train --config config.json --output runs/full
python -m earadapt run --config config.json --output runs/full
```

`config.json` records the method settings and seed `13`; the resolved configuration is saved as `runs/full/settings.json`. Keep separate output directories for different seeds and variants.

## Evaluation

```bash
python -m earadapt evaluate --config config.json --output runs/full
```

## Ablations

Supported controls: `full`, `frozen`, `unbounded`, `no-anchor`, `no-acceptance`, `immediate`.

Repeat the same data and configuration in a separate directory, for example:

```bash
python -m earadapt prepare --input data/speech.csv --config config.json --output runs/frozen --variant frozen
python -m earadapt train --config config.json --output runs/frozen --variant frozen
python -m earadapt run --config config.json --output runs/frozen --variant frozen
python -m earadapt evaluate --config config.json --output runs/frozen --variant frozen
```

## Outputs

All main outputs are written below `runs/full/`.

`base.pt` stores the frozen 64-dimensional speech detector. Accepted `adapter_<speaker>.pt` files store personalized states. `schedule.csv` logs admitted/deferred quanta and charged resources. `speakers.csv` includes rejected/postponed enrollment outcomes, query recall, cache/workspace bytes and background exposure. `metrics.csv` reports mean/lower-decile recall, false wakes/hour, acceptance/postponement, modeled resource charges and scheduling failures.

```bash
python -m earadapt plot --config config.json --output runs/full
```
