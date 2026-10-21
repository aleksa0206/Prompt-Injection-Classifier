# prompt-guard

Small, fast classifier that flags prompt-injection / jailbreak attempts before they reach an LLM.

## Setup (GPU machine)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Run

```bash
python data.py       # download + merge datasets -> data/
python baseline.py   # TF-IDF baseline
python train.py      # fine-tune ModernBERT -> model/
```

## Data

| Source | Notes |
|---|---|
| deepset/prompt-injections | small, EN + DE |
| xTRam1/safe-guard-prompt-injection | ~10k, synthetic |
| jackhhao/jailbreak-classification | role-play jailbreaks |

## Results

F1 on the held-out test set (2,283 examples), injection class:

| Test source | TF-IDF baseline | Fine-tuned ModernBERT |
|---|---|---|
| deepset (real-world, EN + DE) | 0.652 | **0.897** |
| jackhhao (jailbreaks) | 0.950 | **0.974** |
| xTRam1 (synthetic) | 0.994 | **0.999** |
| **Overall** | 0.964 | **0.987** |

The deepset split is the meaningful one: the synthetic set is easy even for TF-IDF.
Trained for 3 epochs on an RTX 4050 Laptop GPU (6 GB), ~30 min.

TODO: LLM comparison via API, latency benchmark, adversarial evaluation.
