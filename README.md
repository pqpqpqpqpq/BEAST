# Data-efficient and Interpretable Nanopore Signal Prediction via Base-resolved Mapping of Atomic-level Spatio-temporal Translocation Dynamics (BEAST)

This repository contains the PyTorch implementation of [Data-efficient and Interpretable Nanopore Signal Prediction via Base-resolved Mapping of Atomic-level Spatio-temporal Translocation Dynamics (BEAST)] Jiayao Hu, Jintao Zhu, Xuyang Zhao, Qingyuan Fan, Junyao Li, Luping Fang, Qing Pan,and Yi Li.

---

## Introduction

BEAST is an efficient spatio-temporal graph neural network for predicting nanopore sequencing current signals. By integrating atomic-level structural encoding with temporal modeling, BEAST accurately predicts nanopore current features from k-mer sequences, capturing key chemical structures such as the methyl group in 5mC. It achieves strong generalization even under few-shot conditions and can be applied to basecalling, SNP detection, and modification analysis, providing a scalable and interpretable framework for nanopore signal modeling.

---

## Installation

### Requirements
- Python 3.8
- PyTorch v2.4.1
- Other dependencies listed in `requirements.txt`

### Setup
It is recommended to use conda to create a virtual environment:

```bash
conda create -n BEAST python=3.8
conda activate BEAST
git clone https://github.com/pqpqpqpqpq/BEAST.git
cd BEAST
pip install -r requirements.txt
```

The installation should take less than 10 minutes on a typical desktop pc. T

## Training

### 1. Prepare k-mer Data
Prepare the k-mer samples used for training or prediction.

For modified bases, use the following symbols:

| Modification | Symbol |
|---|---|
| 5mC | M |
| 5hmC | K |
| m6A_RNA | X |

Canonical RNA 5-mers/9-mers use `U` instead of `T`. The repository bundles example tables under `kmer_models/`:
R9.4.1 DNA 6-mer (`Canonical.model`, `5mC_OnlyM.model`, `5hmC_OnlyK.model`), RNA004 5-mer/9-mer
(`RNA004-Canonical-5mer.model`, `RNA004-Canonical-9mer.model`) and an m6A-modified RNA 5-mer table
(`RNA004-m6A.model`). For every table, the k-mer length and nucleotide type are selected with
`--kmer-len` and `--n-type` (see the training and prediction sections).

---

### 2. Run Training Scripts

```bash
# Single k-mer model training (with defaults)
python Train/train_fixed_kmer.py

# Mixed k-mer model training (with defaults)
python Train/train_mixed_kmer.py
```


#### Single k-mer Model Training (`train_fixed_kmer.py`)

**Description**  
Train the BEAST model using a single k-mer model.

**Sampling Strategy**  
The script progressively downsamples the input k-mer model samples from 10% to 90% (typically with a 10% step size) and sequentially feeds them into the BEAST architecture.

**Output**  
For every training proportion `<train>-<test>` (e.g. `0.1-0.9`) and every fold, the script saves:
- model weights: `<model_fold>/<proportion>/fold_<i>_best.pth`
- dataset splits: `<model_fold>/dataset/<proportion>/fold_<i>_{train,val,test}_kmers.npy`
- metrics: `<result_fold>/model_weight.npy_fold_<i>_train_size_<train>_test_size_<test>.npy` (r/RMSE per fold, plus `model_weight.npy` with all folds)

**Command-line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fn` | `./kmer_models/Canonical.model` | Path to the k-mer model file |
| `--model_fold` | `../train_modified_kmer` | Directory to save model weights and dataset splits |
| `--result_fold` | `../train_modified_kmer/result` | Directory to save CV results |
| `--device` | `0` | GPU device index (e.g. `0`, `1`) or `cpu` (falls back to CPU automatically) |
| `--kmer-len` | `6` | k-mer length used to build the model (e.g. `5`, `6`, `9`) |
| `--n-type` | `DNA` | Nucleotide type: `DNA` or `RNA` |

**Examples**

```bash
# Use custom k-mer file and save results to custom directory
python Train/train_fixed_kmer.py \
    --fn ./kmer_models/Canonical.model \
    --model_fold ./output/weights \
    --result_fold ./output/results \
    --device 0

# Run on CPU
python Train/train_fixed_kmer.py --device cpu

# Run on GPU 0
python Train/train_fixed_kmer.py --device 0

# Train a 9-mer DNA model (e.g. R10.4.1 / RNA004 canonical 9-mer table)
python Train/train_fixed_kmer.py \
    --fn ./kmer_models/RNA004-Canonical-9mer.model \
    --kmer-len 9 --n-type RNA --device 0

```


#### Mixed k-mer Model Training (`train_mixed_kmer.py`)

**Description**  
Train the BEAST model using two different k-mer models simultaneously.

**Input Requirements**
- one Canonical k-mer model
- one Modified k-mer model

**Sampling Strategy**  
The Canonical k-mer model is always fully retained, while the Modified k-mer model is progressively downsampled from 10% to 90%. The mixed data is then used for BEAST training.

**Output**  
For every mixture proportion `<train_split>` and every fold:
- model weights: `<model_fold>/<train_split>/fold_<i>_best.pth`
- dataset splits: `<model_fold>/dataset/<train_split>/fold_<i>_{train,val,test}_kmers.npy`
- metrics: `<result_fold>/model_weight.npy_fold_<i>_train_split_<train_split>.npy`

**Command-line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fn` | `./kmer_models/Canonical.model` | Path to the Canonical k-mer model file |
| `--fn_M` | `./kmer_models/5mC_OnlyM.model` | Path to the Modified k-mer model file |
| `--model_fold` | `../train_mixed_kmer` | Directory to save model weights and dataset splits |
| `--result_fold` | `../train_mixed_kmer/result` | Directory to save CV results |
| `--device` | `0` | GPU device index (e.g. `0`, `1`) or `cpu` (falls back to CPU automatically) |
| `--kmer-len` | `6` | k-mer length used to build the model (e.g. `5`, `6`, `9`) |
| `--n-type` | `DNA` | Nucleotide type: `DNA` or `RNA` |

**Examples**

```bash
# Use custom modified model and save to custom directory
python Train/train_mixed_kmer.py \
    --fn ./kmer_models/Canonical.model \
    --fn_M ./kmer_models/5hmC_OnlyK.model \
    --model_fold ./output/weights \
    --result_fold ./output/results \
    --device 0

# Run on CPU
python Train/train_mixed_kmer.py --device cpu

# Run on GPU 0
python Train/train_mixed_kmer.py --device 0

# Train an m6A-modified RNA 5-mer model (X = m6A)
python Train/train_mixed_kmer.py \
    --fn ./kmer_models/RNA004-Canonical-5mer.model \
    --fn_M ./kmer_models/RNA004-m6A.model \
    --kmer-len 5 --n-type RNA --device 0

```

---

---

#### Context-held-out training (`train_context_holdout_kmer.py`)

**Description**  
Each canonical 6-mer and all of its modification-pattern variants form one *context cluster*. Splits are made at the
cluster (canonical backbone) level, so a modified test k-mer never shares its canonical backbone with any training
entry (Supplementary Table S3 and Fig. S17b). The training set contains the modified k-mers of the training clusters
plus their canonical counterparts; 10% of those modified k-mers are held out for validation.

**Output**  
- model weights: `<model_fold>/<train_fraction>/fold_<i>_best.pth`
- dataset splits: `<model_fold>/dataset/<train_fraction>/fold_<i>_{train,val,test}_kmers.npy`
- metrics: `<result_fold>/model_weight.npy_fold_<i>_train_split_<train_fraction>.npy`

**Command-line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fn` | `./kmer_models/Canonical.model` | Canonical k-mer model file |
| `--fn_M` | `./kmer_models/5mC_OnlyM.model` | Modified k-mer model file (5mC or 5hmC) |
| `--model_fold` | `../train_context_holdout` | Directory to save weights and dataset splits |
| `--result_fold` | `../train_context_holdout/result` | Directory to save CV results |
| `--dataset_dir` | — | Optional pre-saved splits `<dir>/<train_fraction>/fold_<i>_*.npy`; if omitted, splits are generated |
| `--train_splits` | `0.1,...,0.9` | Comma-separated training fractions of the modified table |
| `--device` | `0` | GPU device index (e.g. `0`, `1`) or `cpu` (falls back to CPU automatically) |

**Examples**

```bash
# 5mC context-held-out training (splits generated on the fly)
python Train/train_context_holdout_kmer.py     --fn ./kmer_models/Canonical.model     --fn_M ./kmer_models/5mC_OnlyM.model     --model_fold ./output/context --result_fold ./output/context/result --device 0

# Only the 40% training coverage used in the manuscript
python Train/train_context_holdout_kmer.py --train_splits 0.4 --device 0

# Reuse pre-saved context splits
python Train/train_context_holdout_kmer.py --dataset_dir ./output/context/dataset --device 0
```

#### Zero-shot / few-shot training (`train_zero_few_shot_kmer.py`)

**Description**  
Leave-modification-out evaluation (Supplementary Figs. S13-S15). Zero-shot modes train on the canonical table plus the
*other* modification and evaluate on the complete held-out target table; few-shot modes additionally include a fraction
of the target table and evaluate on the remaining target entries.

| `--mode` | Experiment |
|---|---|
| `a` | zero-shot 5mC (train: canonical + 5hmC) |
| `b` | zero-shot 5hmC (train: canonical + 5mC) |
| `c` | few-shot 5mC (train: canonical + 5hmC + a fraction of 5mC) |
| `d` | few-shot 5hmC (train: canonical + 5mC + a fraction of 5hmC) |

**Output**  
Same layout as above, with `<key>` = `zero_shot_5mC` / `zero_shot_5hmC` (zero-shot) or the few-shot fraction (e.g. `0.1`):
weights `<model_fold>/<key>/fold_<i>_best.pth`, splits `<model_fold>/dataset/<key>/fold_<i>_*.npy`,
metrics `<result_fold>/model_weight.npy_fold_<i>_<key>.npy`.

**Command-line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fn` | `./kmer_models/Canonical.model` | Canonical k-mer model file |
| `--fn_M` | `./kmer_models/5mC_OnlyM.model` | 5mC k-mer model file |
| `--fn_K` | `./kmer_models/5hmC_OnlyK.model` | 5hmC k-mer model file |
| `--mode` | `a` | `a`/`b` zero-shot 5mC/5hmC, `c`/`d` few-shot 5mC/5hmC |
| `--model_fold` | `../train_zero_few_shot` | Directory to save weights and dataset splits |
| `--result_fold` | `../train_zero_few_shot/result` | Directory to save CV results |
| `--train_splits` | `0.1,...,0.9` | Few-shot training fractions (ignored in zero-shot modes) |
| `--device` | `0` | GPU device index (e.g. `0`, `1`) or `cpu` (falls back to CPU automatically) |

**Examples**

```bash
# Zero-shot 5mC and few-shot 5mC
python Train/train_zero_few_shot_kmer.py --mode a --device 0
python Train/train_zero_few_shot_kmer.py --mode c --train_splits 0.1 --device 0

# Zero-shot 5hmC with custom output directories
python Train/train_zero_few_shot_kmer.py     --fn ./kmer_models/Canonical.model     --fn_M ./kmer_models/5mC_OnlyM.model     --fn_K ./kmer_models/5hmC_OnlyK.model     --mode b --model_fold ./output/zero_shot --result_fold ./output/zero_shot/result --device 0
```

Both scripts above train 6-mer DNA models with the same protocol as `train_mixed_kmer.py`
(AdamW, up to 400 epochs, early stopping patience 15) and use batched inference.

---

## Predict k-mer Models Using BEAST

```bash
python kmer_models/pred_kmer_model.py \
    --model-weight ./10%_model_weight/Canonical/Canonical_BEAST.pth \
    --kmer-model-file ./kmer_models/r9.4_450bps.nucleotide.6mer.template.model \
    --fn ./kmer_models/Canonical.model \
    --output-path ../output_results/pred.model \
    --device 0
```

This step performs BEAST inference to predict k-mer-level mean current values.
`--kmer-model-file` is the **template model** whose `level_mean` column is replaced by the predictions; the output
therefore has the same columns, k-mer order and row count as the template (`kmer, level_mean, level_stdv, sd_mean, sd_stdv, weight`)
with `level_mean` substituted by the BEAST predictions.
The output keeps the comment lines, k-mer order and columns of the template, with `level_mean` replaced by the
predicted value; the remaining columns (`level_stdv, sd_mean, sd_stdv, weight`) are copied from the template.


#### Performance

- Inference typically finishes in **less than 60 seconds** on a standard desktop computer.


#### Input Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-weight` | Yes | — | Path to the trained BEAST model weights (`.pth`) |
| `--kmer-model-file` | Yes | — | Path to the template k-mer model file |
| `--fn` | Yes | — | Path to the input k-mer model file |
| `--output-path` | No | `../pred.model` | Path to save the predicted model |
| `--device` | No | `0` | GPU device index (e.g. `0`, `1`) or `cpu` (falls back to CPU automatically) |
| `--kmer-len` | No | `6` | k-mer length used to build the model (e.g. `5`, `6`, `9`); must match the checkpoint |
| `--n-type` | No | `DNA` | Nucleotide type: `DNA` or `RNA`; must match the checkpoint |

The script prints a run summary (input file, checkpoint, device, number of k-mers processed, template file and output path).

**Examples**

```bash
# Predict a complete canonical 6-mer table
python kmer_models/pred_kmer_model.py \
    --model-weight ./10%_model_weight/Canonical/Canonical_BEAST.pth \
    --kmer-model-file ./kmer_models/r9.4_450bps.nucleotide.6mer.template.model \
    --fn ./kmer_models/Canonical.model \
    --output-path ./output_results/pred.model --device 0

# Predict an m6A-modified RNA 5-mer table (X = m6A)
python kmer_models/pred_kmer_model.py \
    --model-weight ./10%_model_weight/m6A_RNA/m6A_RNA_BEAST.pth \
    --kmer-model-file ./kmer_models/RNA004-m6A.model \
    --fn ./kmer_models/RNA004-m6A.model \
    --kmer-len 5 --n-type RNA \
    --output-path ./output_results/m6A_RNA_pred.model --device 0

```


#### Troubleshooting

- **No GPU / CPU-only machine**: pass `--device cpu`, or omit it — the script falls back to CPU automatically when CUDA is unavailable.
- **Checkpoint loading error (`module.` prefix, `Missing keys`/`Unexpected keys`)**: handled automatically; the loader strips or adds the `module.` prefix so both DataParallel and plain checkpoints load on GPU or CPU.
- **`k-mer length mismatch` / shape errors**: `--kmer-len` and `--n-type` must match the checkpoint (6-mer DNA weights are provided in `10%_model_weight/`); the `num_frame`/`num_joints` used for 5-mer/9-mer and RNA are derived from these two arguments.
- **`rdkit` import error**: install it with `pip install rdkit` (already in `requirements.txt`).
- **Output row count differs from the template**: `--fn` and `--kmer-model-file` must contain the same k-mers in the same order.
- **Out-of-memory during training**: use another GPU with `--device 1`, or train on CPU with `--device cpu`.


---


### Tools for Downstream Analysis
Downstream analyses use [Squigulator](https://github.com/nanoporetech/squigulator), [Clair3](https://github.com/HKU-BAL/Clair3), [RTG-ToolS](https://github.com/RealTimeGenomics/rtg-tools), [f5c](https://github.com/nanoporetech/f5c), and [DeepSME](https://github.com/sparkcyf/DeepSME).
You can follow their respective instructions to perform downstream tasks using the predicted k-mer models.
