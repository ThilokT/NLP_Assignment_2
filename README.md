# Project 7: Do LLMs Know When They're Making Things Up?
## NLP CS60075 / Assignment 2 — Hallucination Detection on HaluEval QA
**Student Roll:** 23CS30073  
**Repository:** [https://github.com/ThilokT/NLP_Assignment_2](https://github.com/ThilokT/NLP_Assignment_2)

---

## Overview

This repository provides an end-to-end, reproducible implementation for detecting LLM hallucinations on the **HaluEval QA** benchmark. It systematically evaluates and combines three paradigms:

1. **Task 1 — White-Box Internal Signals:** Forced decoding using `google/gemma-2-2b-it` conditioned on factual knowledge context (`"Context: {knowledge}\nQuestion: {question}\nAnswer: "`). Extracts a 9-feature suite including token log-probabilities, predictive entropy, sequence length, and perplexity.
2. **Task 2 — Black-Box Sampling Consistency (SelfCheckGPT):** Stochastic resampling ($N=20, T=1.2$) combined with context-prepended BERTScore (`roberta-large`) and 3-class Natural Language Inference (`microsoft/deberta-large-mnli`), measuring contradiction and inverted entailment.
3. **Task 3 — Stacked Ensemble Meta-Classifier:** Supervised meta-classifiers (Logistic Regression, Random Forest, Gradient Boosting) trained on extracted signals with 5-fold cross-validation (`GroupKFold` grouped by question) and evaluated via 2,000 cluster-bootstrap iterations on held-out test data.

---

## Repository Structure

```
NLP_Assignment_2/
├── task1.py                 # Task 1: Knowledge-conditioned forced decoding & feature extraction
├── task2.py                 # Task 2: SelfCheckGPT (Resampling + BERTScore + DeBERTa NLI)
├── task3.py                 # Task 3: Supervised stacked ensemble meta-classifier & bootstrap
├── tasks_colab.ipynb        # Interactive 1-click execution notebook for Google Colab
├── tasks_Kaggle.ipynb       # Interactive 1-click execution notebook for Kaggle Notebooks
├── data/
│   ├── halueval_qa_sampled.json  # 400 sampled QA pairs (seed 42)
│   └── train_test_split.json     # 320 train / 80 test question split (seed 42)
├── checkpoints/             # Incremental feature caches (auto-resumes if interrupted)
│   ├── task1_features.json
│   ├── task2_resamples.json
│   └── task2_scores.json
├── outputs/                 # Evaluation reports, trained models, plots, CSV tables
│   ├── task1_plots.png
│   ├── task1_results.json
│   ├── task2_plots.png
│   ├── task2_results.json
│   ├── task3_pr_curves.png
│   ├── task3_results.json
│   ├── task3_meta_classifier.joblib
│   └── final_comparison.csv
├── .gitignore               # Ignored cache, model weight, and virtual environment files
└── README.md                # Comprehensive documentation and execution guide
```

---

## Pretrained Models & Hardware

All models are open-weights and loaded directly from Hugging Face:

| Model Role | Hugging Face Hub ID | Backbone / Architecture | Purpose |
| :--- | :--- | :--- | :--- |
| **Generative LLM** | `google/gemma-2-2b-it` | Gemma-2 (2.6B params, float16) | Forced decoding (Task 1) & Stochastic Resampling (Task 2) |
| **BERTScore Scorer** | `roberta-large` | RoBERTa (355M params) | Semantic similarity against resamples (Task 2) |
| **NLI Classifier** | `microsoft/deberta-large-mnli` | DeBERTa-large (434M params) | 3-class cross-encoder contradiction/entailment (Task 2) |

> [!NOTE]
> `google/gemma-2-2b-it` is a gated model. You must accept the license terms on its [Hugging Face page](https://huggingface.co/google/gemma-2-2b-it) and generate a free Read Token.

**Hardware Recommendation:** NVIDIA GPU with $\ge 12$ GB VRAM (e.g., Tesla T4, Tesla P100, V100, or A100). Execution on CPU is supported as fallback.

---

## Running on Google Colab

Use the provided notebook [`tasks_colab.ipynb`](file:///d:/Academics/sem7/NLP/asgn-2/tasks_colab.ipynb) for a seamless 1-click execution pipeline with permanent Google Drive storage.

### Step 1: Open Notebook in Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Select **Upload** and upload `tasks_colab.ipynb` (or open from GitHub).

### Step 2: Configure Hardware Accelerator
1. In the top menu, go to **Runtime → Change runtime type**.
2. Set **Hardware accelerator** to **T4 GPU** (Standard free tier).
3. Click **Save**.

### Step 3: Run the Cells Sequentially
* **Cell 1 (GPU Verification):** Runs `nvidia-smi` and checks PyTorch CUDA availability.
* **Cell 2 (Google Drive Setup & Repo Sync):** Mounts Google Drive, creates `/content/drive/MyDrive/NLP_Assignment_2`, and synchronizes the latest repo scripts:
  ```python
  from google.colab import drive
  import os
  drive.mount('/content/drive')
  %cd /content/drive/MyDrive/NLP_Assignment_2
  if not os.path.exists(".git"):
      !git init
      !git remote add origin https://github.com/ThilokT/NLP_Assignment_2.git
  else:
      !git remote set-url origin https://github.com/ThilokT/NLP_Assignment_2.git 2>/dev/null || true
  !git fetch origin main
  !git reset --hard origin/main
  ```
* **Cell 3 (Dependencies):** Installs required packages:
  ```bash
  !pip install -q transformers torch bert-score huggingface_hub accelerate pandas scikit-learn matplotlib joblib tqdm
  ```
* **Cell 4 (Hugging Face Login):** Authenticates with your Hugging Face access token.
* **Cell 5 (Run Task 1):** Executes forced decoding (~1-2 minutes):
  ```bash
  !python -u task1.py
  ```
* **Cell 6 (Run Task 2):** Executes resampling ($N=20$) and SelfCheckGPT scoring (~15-18 minutes):
  ```bash
  !python -u task2.py
  ```
  *(Checkpoints are saved every 25 questions into Google Drive; if Colab disconnects, re-running automatically resumes where it left off).*
* **Cell 7 (Run Task 3):** Trains and evaluates meta-classifiers (~30 seconds):
  ```bash
  !python -u task3.py
  ```
* **Cell 8 (Results Display):** Displays `final_comparison.csv` and renders PR curves and diagnostic figures directly in the notebook.
* **Cell 9 (Drive Verification):** Confirms all checkpoints, outputs, and joblib models are saved permanently in Google Drive.

---

## Running on Kaggle Notebooks

Use [`tasks_Kaggle.ipynb`](file:///d:/Academics/sem7/NLP/asgn-2/tasks_Kaggle.ipynb) for fast dual-T4 or P100 execution with 1-click ZIP download of results.

### Step 1: Create Notebook on Kaggle
1. Go to [kaggle.com](https://www.kaggle.com/) and click **Create → New Notebook**.
2. Click **File → Import Notebook** and upload `tasks_Kaggle.ipynb`.

### Step 2: Enable GPU and Internet Access (Critical)
In the right sidebar under **Notebook settings**:
1. **Accelerator:** Select **GPU T4 x2** (or GPU P100).
2. **Internet:** Toggle to **Internet ON** (Required to clone GitHub and download Hugging Face models).

> [!IMPORTANT]
> Kaggle notebooks disable internet by default. Ensure **Internet ON** is enabled in the sidebar, or `git fetch` and model downloads will fail.

### Step 3: Run the Cells
* **Cell 1:** Verifies dual Tesla T4 GPUs.
* **Cell 2:** Sets up `/kaggle/working/NLP_Assignment_2` and clones the repository.
* **Cell 3:** Installs dependencies (`bert-score`, `transformers`, etc.).
* **Cell 4:** Authenticates Hugging Face. (You can also save your token in Kaggle **Add-ons → Secrets** with label `HF_TOKEN`).
* **Cells 5, 6, 7:** Execute Tasks sequentially:
  ```bash
  !python -u task1.py
  !python -u task2.py
  !python -u task3.py
  ```
* **Cell 8:** Displays the final comparison table and precision-recall curves inline.
* **Cell 9:** Packages all checkpoints, outputs, and trained models into `nlp_assignment_2_results.zip` with a 1-click browser download link.

---

## Methodological Summary

### Task 1: Knowledge-Conditioned Probability & Entropy
* **Prompt Template:** `"Context: {knowledge}\nQuestion: {question}\nAnswer: "`
* **Evaluated Signals (9 Features):**
  * `kc_mean_log_prob`: Token-averaged log-likelihood
  * `kc_mean_entropy`: Token-averaged predictive entropy
  * `kc_sum_log_prob`: Total sequence log-probability
  * `kc_log_prob_variance`: Variance of token log-probabilities
  * `kc_entropy_variance`: Variance of token entropies
  * `kc_min_log_prob`: Minimum token log-probability (bottleneck token)
  * `kc_max_entropy`: Peak token entropy
  * `kc_perplexity`: Sequence perplexity $\exp(-\text{mean\_lp})$
  * `kc_answer_token_count`: Number of tokens in the generated answer
* **Outputs:** [`outputs/task1_plots.png`](outputs/task1_plots.png), [`outputs/task1_results.json`](outputs/task1_results.json).

### Task 2: SelfCheckGPT (Resampling Consistency)
* **Sampling Protocol:** $N=20$ stochastic resamples per question with temperature $T=1.2$, $\text{top\_p}=0.95$, $\text{max\_new\_tokens}=64$.
* **Scorers & Signals (8 Features):**
  * `bertscore_inconsistency`: $1 - \text{mean}(F_1)$ bare answer comparison (`roberta-large`)
  * `bertscore_inconsistency_ctx`: $1 - \text{mean}(F_1)$ with question context prepended
  * `bertscore_inconsistency_max`: $1 - \max(F_1)$ across resamples
  * `nli_contradiction`: Mean $P(\text{contradiction})$ bare answer (`microsoft/deberta-large-mnli`)
  * `nli_contradiction_ctx`: Mean $P(\text{contradiction})$ with question context
  * `nli_entailment`: $1 - \text{mean}(P(\text{entailment}))$ (inverted entailment)
  * `nli_max_contradiction`: Peak contradiction probability across all $N$ resamples
  * `semantic_entropy_proxy`: Standard deviation of contradiction probabilities across resamples
* **Outputs:** [`outputs/task2_plots.png`](outputs/task2_plots.png), [`outputs/task2_results.json`](outputs/task2_results.json).

### Task 3: Stacked Ensemble Meta-Classifier
* **Feature Integration:** Concatenates all 17 extracted Task 1 and Task 2 signals into a unified feature matrix.
* **Cross-Validation:** 5-fold `GroupKFold` grouped strictly by `sample_id` on the train partition (640 instances) to eliminate data leakage.
* **Models Evaluated:**
  * Logistic Regression (L2 regularized, tuned over $C \in [0.01, 100]$)
  * Random Forest (`n_estimators` $\in [200, 500]$, `max_depth` $\in [3, 5, 7]$)
  * Gradient Boosting (`n_estimators` $\in [100, 200]$, `max_depth` $\in [2, 3]$, `learning_rate` $\in [0.05, 0.1]$)
* **Significance Testing:** Cluster bootstrap with $B=2,000$ resamples at the question cluster level, reporting 95% confidence intervals.
* **Outputs:** [`outputs/task3_meta_classifier.joblib`](outputs/task3_meta_classifier.joblib), [`outputs/final_comparison.csv`](outputs/final_comparison.csv), [`outputs/task3_pr_curves.png`](outputs/task3_pr_curves.png), [`outputs/task3_results.json`](outputs/task3_results.json).

---

## Experimental Results

Evaluated on the held-out test split (80 questions, 160 instances: 80 factual, 80 hallucinated):

### Individual Feature Baselines (Test Partition)

| Feature / Signal | AUC-PR (Test) | Pairwise Accuracy | Key Observation |
| :--- | :---: | :---: | :--- |
| **`answer_token_count`** | **0.9801** | **98.8%** | Hallucinations generate significantly more elaboration |
| **`neg_sum_log_prob`** | **0.9794** | **98.8%** | Total sequence unlikelihood correlates strongly with hallucination |
| **`max_entropy`** | **0.9319** | **95.6%** | Peak uncertainty pinpoints hallucinated factual tokens |
| **`entropy_variance`** | **0.9027** | **95.0%** | Non-uniform uncertainty across token positions |
| **`mean_entropy`** | **0.8572** | **90.0%** | Knowledge conditioning dramatically boosts entropy signal |
| **`SelfCheck-BERTScore (+Context)`** | **0.7116** | **82.5%** | Prepending question context substantially improves semantic consistency |
| **`SelfCheck-NLI (Inverted Entailment)`**| **0.7001** | **73.8%** | Factual answers consistently entail stochastic samples |
| **`SelfCheck-NLI (Max Contradiction)`** | 0.5612 | 45.0% | Peak contradiction across $N=20$ resamples |
| **`Random Baseline`** | 0.5000 | 50.0% | Theoretical uninformative baseline |

### Final Meta-Classifier Comparison (with 95% Bootstrap CI)

| Method | Test AUC-PR | 95% Bootstrap CI | Description |
| :--- | :---: | :---: | :--- |
| **Best Single Feature (`answer_token_count`)** | 0.9801 | [0.9598, 0.9957] | Length / total uncertainty heuristic |
| **Gradient Boosting Classifier (All features)** | 0.9954 | [0.9863, 1.0000] | Non-linear ensemble over 17 signals |
| **Random Forest Classifier (All features)** | 0.9957 | [0.9875, 1.0000] | Bagged decision tree ensemble |
| **Logistic Regression (All features)** | **0.9985** | **[0.9951, 1.0000]** | Calibrated linear ensemble over all 17 signals |
| **★ Best Overall Ensemble (LR)** | **0.9985** | **[0.9951, 1.0000]** | Near-perfect separation on held-out test split |

---

## Reproducibility Checklist

- [x] **Deterministic Splits:** Fixed random seed `42` (`data/train_test_split.json`).
- [x] **No Data Leakage:** Grouped CV (`GroupKFold`) ensures factual and hallucinated counterparts of any question are strictly in the same split.
- [x] **Self-Contained Data Loading:** Automatically downloads and prepares HaluEval QA if data files are missing.
- [x] **Incremental Checkpointing:** Checkpoint files save every 25 questions; long runs resume seamlessly.
- [x] **Pre-trained Serialized Models:** Trained meta-classifier is saved as a ready-to-deploy Joblib pipeline in `outputs/task3_meta_classifier.joblib`.
- [x] **Cloud GPU Verification:** Fully tested and verified on Google Colab (Tesla T4) and Kaggle Notebooks (GPU T4 x2 / P100).
