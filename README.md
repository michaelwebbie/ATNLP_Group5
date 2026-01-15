# ATNLP — SCAN Reimplementation with Transformers, Curriculum Learning,and Chain-of-Thought Fine-Tuning (KU 2025)

## Author
Michael Webster (part of group 5 with Joana Rio Maior, Margherita Grosso and Nikolay Varamezov)

---

Group 5 reimplementation project for Advanced Topics in Natural Language Processing (ATNLP) at the University of Copenhagen.  
This project attempts to reimplement experiments 1, 2 and 3 from the paper *Generalization without Systematicity* by Lake and Baroni.  
This repository contains a modular PyTorch implementation of a Transformer model and full reproductions of the SCAN experiments.

The final submission is a five-page report (*SCAN Reimplementation with Transformers, Curriculum Learning,and Chain-of-Thought Fine-Tuning*)

---

## Contents

This repository includes:

- Modular Transformer implementation (`Scripts/transformer.py`)
- SCAN data loader utilities (`Scripts/dataloader.py`)
- Training, decoding, and evaluation pipelines
- Reproductions of SCAN Experiments 1a, 1b, 2, and 3
- Jupyter notebook (`ATNLP_Group5_main.ipynb`) that runs all reimplementation experiments end-to-end
- Jupyter notebook (`Assignment3_Michael_Webster.ipynb`)
- Plots and aggregated accuracy reporting
- `train.py` — Training loop 
- `decoder.py` — Greedy decoding at inference
- `evaluate.py` — Evaluation functions for Exp1a/1b, Exp2, Exp3  
- `metrics.py` — Token-level and sequence-level accuracy metrics  
- `requirements.txt` — Python dependencies
- Report for submission (`SCAN Reimplementation with Transformers, Curriculum Learning,and Chain-of-Thought Fine-Tuning`)

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

Additional experiments on experiment 1b and 2 are run from `Assignment3_Michael_Webster.ipynb`

This notebook includes
- **Experiment 1b with T5-small finetuning (on the 1% split)**
- **Experiment 2 with T5-small finetuning**
- **Experiment 2 with T5-small and curriculum learning finetuning**
- **Experiment 2 with T5-small, curriculum learning and CoT finetuning**

---

## Resources

- Lake & Baroni (2018), *Generalization without Systematicity*:  
  https://arxiv.org/pdf/1711.00350  

- Original SCAN dataset:  
  https://github.com/brendenlake/SCAN

