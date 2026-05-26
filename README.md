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

### Training
1. Prepare the k-mer data samples that you want to predict.
If your k-mer contains modified bases, use the following letters to represent them:
- 5mC → M
- 5hmC → K

2. Run one of the following training scripts:

```bash
python Train/train_fixed_kmer.py   # Train on a single k-mer model
python Train/train_mixed_kmer.py   # Train on a mixed k-mer model
```

---

# Detailed Training Mechanism

## 1. Single k-mer Model Training (`train_fixed_kmer.py`)

### Description
This script is dedicated to training the BEAST model on a single k-mer model.

### Sampling Strategy
During the training process, the script automatically performs progressive down-sampling on the input k-mer model samples from 10% to 90% (typically with a step size of 10%) and feeds them sequentially into the BEAST architecture.

### Output
The script exports the trained BEAST model weights corresponding to each sampling ratio.

---

## 2. Mixed k-mer Model Training (`train_mixed_kmer.py`)

### Description
This script is designed for mixed k-mer model training, which requires two distinct k-mer model files as input simultaneously.

### Input Requirements
Typically, you need to provide an Canonical k-mer model and a Modified k-mer model.

### Sampling Strategy
In this mode, the system keeps the Canonical k-mer model fully involved. Meanwhile, it sequentially downsamples the Modified k-mer model from 10% to 90% and blends it with the Canonical data before feeding the mixture into the BEAST model.

### Output
The script outputs trained BEAST model weights under different modification mixture proportions.

3.Predict k-mer models using BEAST:

This section describes how to run the model inference to predict k-mer level means using the trained BEAST architecture.

### Performance
* **Inference Time:** The entire inference process typically takes **less than 60 seconds** on a standard desktop computer.

---

### Input Parameters & Arguments

Running the inference script (`pred_kmer_model.py`) requires specifying the following command-line arguments:

| Argument | Type | Required | Description |
| :--- | :---: | :---: | :--- |
| `--model-weight` | `str` | **Yes** | Path to the trained BEAST model weights (`.pth` file). |
| `--kmer-model-file` | `str` | **Yes** | Path to the template k-mer model file (corresponds to `kmer_model_file`). |
| `--fn` | `str` | **Yes** | Path to the k-mer data source file (corresponds to `fn`). |
| `--output-path` | `str` | No | Path to save the predicted output model. (Default: `../pred.model`). |

---

### Usage Example

You can execute the inference by running the following command in your terminal:

```bash
python pred_kmer_model.py \
    --model-weight ../10%_model_weight/Canonical/Canonical_BEAST.pth \
    --kmer-model-file r9.4_450bps.nucleotide.6mer.template.model \
    --fn Canonical.model \
    --output-path ../output_results/pred.model

### Tools for Downstream Analysis
Downstream analyses use [Squigulator](https://github.com/nanoporetech/squigulator), [Clair3](https://github.com/HKU-BAL/Clair3), [RTG-ToolS](https://github.com/RealTimeGenomics/rtg-tools), [f5c](https://github.com/nanoporetech/f5c), and [DeepSME](https://github.com/sparkcyf/DeepSME).
You can follow their respective instructions to perform downstream tasks using the predicted k-mer models.
