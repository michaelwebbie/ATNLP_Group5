# ATNLP_Group5 — SCAN Transformer Reimplementation (KU 2025)

## Authors
Michael Webster  
Joana Rio Maior  
Margherita Grosso  
Nikolay Varamezov  

---

Group 5 reimplementation project for Advanced Topics in Natural Language Processing (ATNLP) at the University of Copenhagen.  
This project attempts to reimplement experiments 1, 2 and 3 from the paper *Generalization without Systematicity* by Lake and Baroni.  
This repository contains a modular PyTorch implementation of a Transformer model and full reproductions of the SCAN experiments.

---

## Contents

This repository includes:

- Modular Transformer implementation (`Scripts/transformer.py`)
- SCAN data loader utilities (`Scripts/dataloader.py`)
- Training, decoding, and evaluation pipelines
- Reproductions of SCAN Experiments 1a, 1b, 2, and 3
- A Jupyter notebook (`ATNLP_Group5_main.ipynb`) that runs all experiments end-to-end
- Plots and aggregated accuracy reporting
- `train.py` — Training loop with batching and gradient clipping  
- `decoder.py` — Greedy decoding
- `evaluate.py` — Evaluation functions for Exp1a/1b, Exp2, Exp3  
- `metrics.py` — Token-level and sequence-level accuracy metrics  
- `requirements.txt` — Python dependencies  

The original SCAN splits are cloned from [`brendenlake/SCAN`](https://github.com/brendenlake/SCAN) at the top of the notebook.

---

## Running in Google Colab

Add this cell at the top of the notebook:

```bash
!git clone https://github.com/michaelwebbie/ATNLP_Group5
%cd ATNLP_Group5
```

---

## Running Experiments

All reimplementations of experiments are run directly from:

`ATNLP_Group5_main.ipynb`

The notebook includes:

- **Experiment 1a** — Full supervision  
- **Experiment 1b** — Limited supervision subsets  
- **Experiment 2** — Length generalization  
- **Experiment 3** — Primitive recombination (*jump*, *turn_left*) and compositional generalization  

---

## Resources

- Lake & Baroni (2018), *Generalization without Systematicity*:  
  https://arxiv.org/pdf/1711.00350  

- Original SCAN dataset:  
  https://github.com/brendenlake/SCAN

