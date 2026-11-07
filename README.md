# Project 7: Do LLMs Know When They're Making Things Up?
## NLP CS60075 / Assignment 2 — Hallucination Detection on HaluEval QA

This repository contains the complete, reproducible implementation for Assignment 2 (Project 7). It evaluates internal token-level probability signals (Task 1), black-box sampling consistency via SelfCheckGPT (Task 2), and a supervised stacked ensemble meta-classifier (Task 3) on the **HaluEval QA** benchmark.

---

## 1. Directory & Code Organization

The code is organized to mirror the Task 1 / Task 2 / Task 3 structure:

```
23CS30073_NLP_Assignment_2/
├── code/
│   ├── task1/
│   │   └── task1.py              # Task 1: Forced decoding & token log-probs/entropy
│   ├── task2/
│   │   └── task2.py              # Task 2: SelfCheckGPT (resampling, BERTScore, DeBERTa NLI)
│   ├── task3/
│   │   └── task3.py              # Task 3: Stacked ensemble meta-classifier & bootstrap
│   ├── prepare_offline_cache.py  # Dataset sampling and offline Hugging Face caching
│   └── run_all.sh                # End-to-end sequential execution pipeline
├── data/
│   ├── halueval_qa_raw.json      # 10,000 raw QA pairs from HaluEval
│   ├── halueval_qa_sampled.json  # 400 sampled QA pairs (seed 42)
│   └── train_test_split.json     # 320 train / 80 test split (seed 42)
├── checkpoints/                  # Intermediate feature caches (omitted from submission zip)
│   ├── task1_features.json
│   ├── task2_resamples.json
│   └── task2_scores.json
├── outputs/                      # Metric reports, CSVs, and diagnostic figures
│   ├── task1_results.json
│   ├── task1_plots.png
│   ├── task2_results.json
│   ├── task2_plots.png
│   ├── task3_results.json
│   ├── task3_pr_curves.png
│   ├── final_comparison.csv
│   └── task3_meta_classifier.joblib
├── requirements.txt              # Pinned pip dependency manifest
├── README.md                     # Reproducibility guide & instructions
└── report.pdf                    # 8-12 page final technical report
```

---

## 2. Environment Setup

### Prerequisites
- Linux OS with Python 3.10
- NVIDIA GPU with $\ge 12$ GB VRAM (e.g. Tesla P100 16GB, Tesla V100 16GB, or A100)
- CUDA $\ge 11.8$ / $12.1$

### Installation
```bash
# 1. Create and activate a clean virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install pinned requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Pretrained Models & Sources

All models are open-weights and sourced directly from the Hugging Face Hub:

| Model Role | Hugging Face Checkpoint ID | Backbone / Architecture | Source |
| :--- | :--- | :--- | :--- |
| **Generative LLM** (Task 1 & 2) | `google/gemma-2-2b-it` | Gemma-2 (2.6B params, bfloat16) | [google/gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it) |
| **BERTScore Backbone** (Task 2) | `roberta-large` | RoBERTa (355M params) | [roberta-large](https://huggingface.co/roberta-large) |
| **NLI Classifier** (Task 2) | `potsawee/deberta-v3-large-mnli` | DeBERTa-v3 (434M params) | [potsawee/deberta-v3-large-mnli](https://huggingface.co/potsawee/deberta-v3-large-mnli) |

---

## 4. Dataset Preparation

The dataset used is the **HaluEval QA** subset (Li et al., 2023):
- Source URL: `https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json`
- Contains 10,000 questions, each with one human-verified factual answer and one ChatGPT-generated hallucinated answer.

To download and prepare the dataset and pre-cache models:
```bash
python prepare_offline_cache.py
```
This script:
1. Downloads the raw JSONL data into `data/halueval_qa_raw.json`.
2. Samples $400$ questions ($800$ scoring instances) with fixed random seed `42` $\to$ `data/halueval_qa_sampled.json`.
3. Splits questions $80/20$ ($320$ train questions $= 640$ instances; $80$ test questions $= 160$ instances) with seed `42` $\to$ `data/train_test_split.json`.

---

## 5. Execution Instructions

### Option A: Run Entire Pipeline End-to-End
To run Tasks 1, 2, and 3 sequentially:
```bash
bash run_all.sh
```

### Option B: Run Tasks Independently

#### Task 1: Probability-Based Hallucination Detection
```bash
python -u task1.py
```
- **Output Checkpoint:** `checkpoints/task1_features.json`
- **Results:** `outputs/task1_results.json`, `outputs/task1_plots.png`
- **Runtime:** ~1-2 minutes on GPU.

#### Task 2: SelfCheckGPT (Sampling Consistency)
```bash
python -u task2.py
```
- **Output Checkpoints:** `checkpoints/task2_resamples.json`, `checkpoints/task2_scores.json`
- **Results:** `outputs/task2_results.json`, `outputs/task2_plots.png`
- **Runtime:** ~15-20 minutes on GPU ($N=20$ resamples $\times$ 400 questions).

#### Task 3: Stacked Ensemble Meta-Classifier
```bash
python -u task3.py
```
- **Output Model:** `outputs/task3_meta_classifier.joblib`
- **Results:** `outputs/final_comparison.csv`, `outputs/task3_results.json`, `outputs/task3_pr_curves.png`
- **Runtime:** ~30 seconds.

---

## 6. Reproducibility & Fixed Random Seeds

All stochastic operations use explicit, fixed random seeds:
- **Dataset Sampling Seed:** `42`
- **Train/Test Split Seed:** `42`
- **LLM Resampling Seeds:** `42 + sample_id` (deterministic per question)
- **GroupKFold Cross-Validation:** Grouped strictly by `sample_id` to prevent data leakage between train and test pairs.
- **Bootstrap Significance Test:** `2,000` iterations with fixed seed `42`.

---

## 7. Results Summary (Test Partition: 160 Instances)

| Method | AUC-PR (Test) | 95% Bootstrap CI | Pairwise Acc. |
| :--- | :---: | :---: | :---: |
| **Random Baseline** | 0.5000 | — | 50.0% |
| **Task 1: Mean Entropy** | 0.8572 | — | 90.0% |
| **Task 1: Max Entropy** | 0.9319 | — | 95.6% |
| **Task 1: Negative Sum Log-Prob** | 0.9794 | — | 98.8% |
| **Task 1: Answer Token Count** | 0.9801 | [0.9598, 0.9957] | 98.8% |
| **Task 2: SelfCheck-BERTScore (+Context)** | 0.7116 | — | 82.5% |
| **Task 2: SelfCheck-NLI (Inverted Entailment)** | 0.7001 | — | 73.8% |
| **Task 3: Meta-Classifier (Logistic Regression)** | **0.9985** | **[0.9951, 1.0000]** | **—** |
| **Task 3: Meta-Classifier (Random Forest)** | **0.9957** | **[0.9875, 1.0000]** | **—** |
| **Task 3: Meta-Classifier (Gradient Boosting)** | **0.9954** | **[0.9863, 1.0000]** | **—** |

