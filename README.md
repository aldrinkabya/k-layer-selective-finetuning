# Training-Free Layer Selection for Partial Fine-Tuning of Language Models

Official implementation of "[Training-Free Layer Selection for Partial Fine-Tuning of Language Models](https://doi.org/10.1016/j.ins.2026.123204)" (*Information Sciences*, 2026).

The method scores each layer of a frozen model by the cosine similarity of its representative token ([CLS] for BERT, the last token for GPT-2) with the other layers, using a single forward pass over the training data. It then fine-tunes only the selected $k$ layers. No gradients are needed for selection.

## Setup

```bash
conda create -n layer_select python=3.10
conda activate layer_select
pip install -r requirements.txt
```

The E2E evaluation uses METEOR, which needs Java 8 or later on your `PATH`. It is not needed for the classification tasks.

## Data

```bash
python setup_data.py
```

This downloads the classification datasets (the bundle released with [LM-BFF](https://github.com/princeton-nlp/LM-BFF)) to `data/glue_data/` and the [E2E NLG Challenge](https://github.com/tuetschek/e2e-dataset) data to `data/e2e_data/`, and fetches the NLTK data used by the metrics.

## Selection strategies

| Paper | `--method` |
| :--- | :--- |
| High Scoring $k$ layers | `highest` |
| Low Scoring $k$ layers | `lowest` |
| Block-wise $k$ layers (highest / lowest per block) | `blockwise_highest` / `blockwise_lowest` |
| Random $k$ layers (baseline) | `random` |
| Full fine-tuning (baseline) | `full_ft` |

## Sentence classification (BERT)

Tasks: `sst2`, `sst5`, `mr`, `cr`, `mpqa`, `subj`, `trec`, `cola`, `mnli`, `mnli-mm`, `snli`, `qnli`, `rte`, `mrpc`, `qqp`.

```bash
# Block-wise highest on SST-2: 3 layers, one per block
python run_glue.py --dataset sst2 --method blockwise_highest --k 3 --B 3

# Low Scoring on MRPC
python run_glue.py --dataset mrpc --method lowest --k 3

# Full fine-tuning on QNLI
python run_glue.py --dataset qnli --method full_ft
```

## Data-to-text generation (GPT-2)

```bash
python run_e2e.py --method highest --k 3
python run_e2e.py --method blockwise_lowest --k 3 --B 3
```

Each run prints a summary (accuracy and macro-F1 for classification; BLEU, NIST, METEOR, ROUGE-L and CIDEr for E2E) and writes it as a CSV to `results/` or `results_e2e/`.

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--dataset` | Task name (classification only) | required |
| `--method` | Selection strategy, see above | `highest` |
| `--k` | Number of layers to fine-tune | 3 |
| `--B` | Number of blocks (block-wise methods) | 3 |
| `--batch_size` | Batch size | 32 (BERT), 8 (GPT-2) |
| `--epochs` | Training epochs | 5 |
| `--lr` | Learning rate | 2e-5 (BERT), 5e-5 (GPT-2) |
| `--seed` | Random seed | 42 |
| `--output_dir` | Where the results CSV is written | `./results` (BERT), `./results_e2e` (GPT-2) |

## Citation

```bibtex
@article{biswas2026training,
  title={Training-Free Layer Selection for Partial Fine-Tuning of Language Models},
  author={Biswas, Aldrin Kabya and Fahim, Md and Fuad, Md Tahmid Hasan and Mazumder, Akm Moshiur Rahman and Ali, Amin Ahsan and Rahman, AKM Mahbubur},
  journal={Information Sciences},
  volume={740},
  pages={123204},
  year={2026},
  publisher={Elsevier},
  doi={10.1016/j.ins.2026.123204}
}
```
