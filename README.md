
Official implementation of:

**Robust Emotion Recognition via Bi-Level Self-Supervised Continual Learning on Unlabeled EEG Data**

## Overview

SSOCL is a self-supervised online continual learning framework for EEG-based emotion recognition under sequential unlabeled subject streams. The framework combines:

- Self-supervised temporal predictive learning
- Cluster-based pseudo-label generation
- Memory replay with entropy-aware sample selection
- Online continual adaptation for cross-subject EEG streams

The method is evaluated on:
- DEAP
- AMIGOS
- PPB-EMO

---

## Installation

Create environment:

```bash
conda create -n torchlab python=3.10
conda activate torchlab
```

Install dependencies


```bash
pip install -r requirements.txt
```

## Training

Run SSOCL

```bash
python main.py
```

Run with Weights & Biases logging
```bash
python main.py --use_wandb
```

Example

```bash
python main.py \
    --dataset AMIGOS \
    --source_data PPB_EMO \
    --buffer_size 200 \
    --temperature 10 \
    --use_wandb
```
##Datasets
The following datasets are used:

DEAP
AMIGOS
PPB-EMO


## Citation
@article{ssocl,
  title={Robust Emotion Recognition via Bi-Level Self-Supervised Continual Learning on Unlabeled EEG Data},
  author={...},
  journal={},
  year={}
}


Please download the datasets from their official sources and place them in the appropriate dataset directory.
