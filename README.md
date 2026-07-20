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

The installation should take less than 10 minutes on a typical desktop pc. The final output is a .model file containing k-mers and their corresponding current levels.

## Training

### 1. Prepare k-mer Data
Prepare the k-mer samples used for training or prediction.

For modified bases, use the following symbols:

| Modification | Symbol |
|---|---|
| 5mC | M |
| 5hmC | K |

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
Exports trained BEAST model weights for each sampling ratio.

**Command-line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fn` | `../kmer_models/Canonical.model` | Path to the k-mer model file |
| `--model_fold` | `../train_modified_kmer` | Directory to save model weights |
| `--result_fold` | `../train_modified_kmer/result` | Directory to save CV results |
| `--device` | `0` | GPU device index (e.g. `0`, `1`) or `cpu` |

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
Exports trained BEAST model weights under different modification mixture proportions.

**Command-line Arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--fn` | `./kmer_models/Canonical.model` | Path to the Canonical k-mer model file |
| `--fn_M` | `../kmer_models/5mC_OnlyM.model` | Path to the Modified k-mer model file |
| `--model_fold` | `../train_mixed_kmer` | Directory to save model weights |
| `--result_fold` | `../train_mixed_kmer/result` | Directory to save CV results |
| `--device` | `0` | GPU device index (e.g. `0`, `1`) or `cpu` |

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
```

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

This step performs BEAST inference to predict k-mer-level mean values. Example output files are provided in `kmer_models/r9.4_450bps.nucleotide.6mer.template.model`


#### Performance

- Inference typically finishes in **less than 60 seconds** on a standard desktop computer.


#### Input Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model-weight` | Yes | — | Path to the trained BEAST model weights (`.pth`) |
| `--kmer-model-file` | Yes | — | Path to the template k-mer model file |
| `--fn` | Yes | — | Path to the input k-mer model file |
| `--output-path` | No | `../pred.model` | Path to save the predicted model |
| `--device` | No | `0` | GPU device index (e.g. `0`, `1`) or `cpu` |

---


### Tools for Downstream Analysis
Downstream analyses use [Squigulator](https://github.com/nanoporetech/squigulator), [Clair3](https://github.com/HKU-BAL/Clair3), [RTG-ToolS](https://github.com/RealTimeGenomics/rtg-tools), [f5c](https://github.com/nanoporetech/f5c), and [DeepSME](https://github.com/sparkcyf/DeepSME).
You can follow their respective instructions to perform downstream tasks using the predicted k-mer models.
