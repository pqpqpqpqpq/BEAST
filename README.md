# BEAST: A Graph-based Transformer Framework for Generalizable Nanopore Signal Modeling under Data Scarcity

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

The installation should take less than 10 minutes on a typical desktop pc. Finally, a .model file containing k-mers and information about the horizontal current will be output.

### Training
1. Prepare the k-mer data samples that you want to predict.
If your k-mer contains modified bases, use the following letters to represent them:
- 5mC → M
- 5hmC → K
- 6mA → Q

2. Run one of the following training scripts:

```bash
python Train/train_fixed_kmer.py   # Train on a single k-mer model
python Train/train_mixed_kmer.py   # Train on a mixed k-mer model
```
3.Predict k-mer models using BEAST:

```bash
python kmer_models/pred_kmer_model.py
```

The inference should take less 60 seconds on a typical desktop computer.
### Tools for Downstream Analysis
Downstream analyses use [Squigulator](https://github.com/nanoporetech/squigulator), [Clair3](https://github.com/HKU-BAL/Clair3), [f5c](https://github.com/nanoporetech/f5c), and [DeepSME](https://github.com/sparkcyf/DeepSME).
You can follow their respective instructions to perform downstream tasks using the predicted k-mer models.
