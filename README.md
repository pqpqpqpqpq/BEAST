# Data-efficient and Interpretable Nanopore Signal Prediction via Base-resolved Mapping of Atomic-level Spatio-temporal Translocation Dynamics (BEAST)

This repository contains the PyTorch implementation of [Data-efficient and Interpretable Nanopore Signal Prediction via Base-resolved Mapping of Atomic-level Spatio-temporal Translocation Dynamics (BEAST)] Jiayao Hu, Jintao Zhu, Xuyang Zhao, Qingyuan Fan, Junyao Li, Luping Fang, Qing Pan,and Yi Li.

---

## Introduction

BEAST is an efficient spatio-temporal graph neural network for predicting nanopore sequencing current signals. By integrating atomic-level structural encoding with temporal modeling, BEAST accurately predicts nanopore current features from k-mer sequences, capturing key chemical structures such as the methyl group in 5mC. It achieves strong generalization even under few-shot conditions and can be applied to basecalling, SNP detection, and modification analysis, providing a scalable and interpretable framework for nanopore signal modeling.

---

## Installation

### Requirements
- Python 3.8
- PyTorch v2.0.1
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
python Train/train_fixed_kmer.py   # Train on a single k-mer model
python Train/train_mixed_kmer.py   # Train on mixed k-mer models
```


#### Single k-mer Model Training (`train_fixed_kmer.py`)

**Description**  
Train the BEAST model using a single k-mer model.

**Sampling Strategy**  
The script progressively downsamples the input k-mer model samples from 10% to 90% (typically with a 10% step size) and sequentially feeds them into the BEAST architecture.

**Output**  
Exports trained BEAST model weights for each sampling ratio.


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

---

## Predict k-mer Models Using BEAST

```bash
python pred_kmer_model.py \
    --model-weight ../10%_model_weight/Canonical/Canonical_BEAST.pth \
    --kmer-model-file r9.4_450bps.nucleotide.6mer.template.model \
    --fn Canonical.model \
    --output-path ../output_results/pred.model
```

This step performs BEAST inference to predict k-mer-level mean values.


#### Performance

- Inference typically finishes in **less than 60 seconds** on a standard desktop computer.


#### Input Arguments

- `--model-weight`：Path to the trained BEAST model weights (`.pth`).

- `--kmer-model-file`：Path to the template k-mer model file.

- `--fn`：Path to the input k-mer model file.

- `--output-path`：Path to save the predicted model. Default: `../pred.model`.

---


### Tools for Downstream Analysis
Downstream analyses use [Squigulator](https://github.com/nanoporetech/squigulator), [Clair3](https://github.com/HKU-BAL/Clair3), [RTG-ToolS](https://github.com/RealTimeGenomics/rtg-tools), [f5c](https://github.com/nanoporetech/f5c), and [DeepSME](https://github.com/sparkcyf/DeepSME).
You can follow their respective instructions to perform downstream tasks using the predicted k-mer models.
