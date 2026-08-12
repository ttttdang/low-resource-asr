# Robust and Efficient Automatic Speech Recognition for Low-Resource Languages

## Overview

This repository contains code and configurations for the MPhil thesis in Machine Learning and Machine Intelligence at the University of Cambridge, titled "Robust and Efficient Automatic Speech Recognition for Low-Resource Languages".

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## Usage

Run all commands **from the repository root** — config paths are relative and
resolved at runtime.

### 1. Build the manifests

Place the downloaded corpora under `./data` and then build the per-language
manifests `./manifests/{language}/{split}.csv`:

```bash
python preprocessing/build_manifest.py --language ast_es --split all
python preprocessing/build_manifest.py --language es_419 --split all --num-proc 8
```

### 2. Train and evaluate

Each run trains, validates every epoch, and evaluates the best checkpoint on the test set, writing per-utterance predictions to `./results/{run_name}/`.

```bash
# Full fine-tune
python src/train.py --config configs/fullft_ast_medium_fleurs.yaml

# Adapter Tuning
python src/train.py --config configs/adapter_tuning_ast_fleurs.yaml

# Sequential Stacking, train the Spanish source adapter first, then stack a fresh Asturian adapter on top of it
python src/train.py --config configs/method2_es_source_adapter.yaml
python src/train.py --config configs/method3_es_to_ast_stack_adapter.yaml
```

Any config value can be overridden from the command line:

```bash
python src/train.py --config configs/adapter_tuning_ast_fleurs.yaml \
                    --overrides training.lr=1e-4 training.epochs=10
```

### Logging

Runs log to `./logs/{run_name}/train.log`. Weights & Biases is optional and off
by default; enable it by adding to a config:

```yaml
wandb:
  enabled: true
  project: low-resource-asr
```

## Adding a language

Three registries must be updated together:

| File | What to add |
|---|---|
| `preprocessing/build_manifest.py` | an entry in `LANG_CODE` and the `--language` choices |
| `src/text_normalizer.py` | a normalizer class plus its `_REGISTRY` entry |
| `src/collator.py` | a `_LANG_TO_WHISPER` mapping to a Whisper language name |

For a language Whisper has no token for, map it to the closest available one —
Asturian uses `spanish` as a proxy.
