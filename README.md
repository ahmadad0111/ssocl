
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
conda activate torchlab```

