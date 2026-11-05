# NLP Assignment 2 — Cluster Workflow & Execution Guide

This document explains the system architecture, the offline execution strategy, and the step-by-step workflow for running **Project 7 (HaluEval QA)** across the cluster nodes.

---

## 1. System Architecture & Context

The high-performance computing (HPC) cluster operates with a **head-node / compute-node separation**:

```
 ┌─────────────────────────────────────────────────────────────┐
 │                     MASTER (Login Node)                     │
 │  - Has full external internet access                        │
 │  - Has CPU & general system tools                           │
 │  - No dedicated GPU                                         │
 │  - Purpose: Code editing, dataset downloads, model caching  │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                 NFS Shared Storage (/home/rs/...)
                 Everything cached here is visible everywhere
                                │
 ┌──────────────────────────────┴──────────────────────────────┐
 │               COMPUTE NODES (e.g. gnode5)                   │
 │  - 2 × NVIDIA Tesla P100 (16 GB VRAM each)                  │
 │  - Isolated network: NO DIRECT INTERNET / NO DNS ACCESS     │
 │  - Managed via Slurm Workload Manager                       │
 │  - Purpose: GPU compute, model inference, heavy training    │
 └─────────────────────────────────────────────────────────────┘
```

### Why Compute Nodes Cannot Download Directly
Worker nodes like `gnode5` sit in an internal private cluster network. Any direct request to GitHub, Hugging Face, or external websites throws `urllib.error.URLError: [Errno -5] No address associated with hostname`.

### The Solution: Shared NFS Cache
Because `/home/rs/26CS91R12/` is mounted across all nodes via NFS:
1. Datasets and Hugging Face models are downloaded **once on `master`** into `data/` and `~/.cache/huggingface/`.
2. Compute nodes (`gnode5`) access these files locally and run **100% offline without network errors**.

---

## 2. Dataset & Offline Model Preparation

Before running GPU tasks, `prepare_offline_cache.py` is executed on `master`:

```bash
# On master node:
source .venv/bin/activate
python prepare_offline_cache.py
```

### What It Sets Up:
1. **Dataset (`data/halueval_qa_raw.json`)**:
   - Parses 10,000 raw QA pairs from HaluEval.
   - Extracts a reproducible 400-question sample (`data/halueval_qa_sampled.json`, seed `42`).
   - Creates an 80/20 train/test partition (`data/train_test_split.json`: 320 train pairs $\to$ 640 instances, 80 test pairs $\to$ 160 instances).
2. **Pre-Cached Models (stored in `~/.cache/huggingface/hub/`)**:
   - `google/gemma-2-2b-it`: 2B parameter open LLM (used in Task 1 & Task 2).
   - `roberta-large`: Backbone for SelfCheck-BERTScore (Task 2).
   - `potsawee/deberta-v3-large-mnli`: Backbone for SelfCheck-NLI (Task 2).

---

## 3. The Three Tasks Explained

### Task 1: Probability-Based Hallucination Detection (`task1.py`)
* **Objective:** Detect hallucinations using internal model token probabilities without external verification.
* **Method (Forced Decoding):**
  - Prompt: `Question: <question>\nAnswer: `
  - Force `gemma-2-2b-it` to decode the candidate answer token-by-token.
  - Compute two features for each candidate:
    1. **Mean Token Log-Probability**: $\frac{1}{T}\sum \log P(w_t \mid w_{<t})$ (lower $\to$ more hallucinated).
    2. **Mean Predictive Entropy**: $\frac{1}{T}\sum \mathcal{H}(P_t)$ (higher $\to$ more hallucinated).
* **Scoring Instances:** 800 total (each question has 1 faithful answer [label 0] and 1 hallucinated answer [label 1]).
* **Evaluation Metric:** AUC-PR on the 160 test instances.
* **Auto-Checkpoints:** Saves progress to `checkpoints/task1_logprobs_entropy.json` every 25 questions.

---

### Task 2: SelfCheckGPT (`task2.py`)
* **Objective:** Detect hallucinations by measuring self-consistency across multiple stochastic samples.
* **Method:**
  1. **Resampling:** Draw $N = 20$ stochastic samples per question at `temperature = 1.0` using `gemma-2-2b-it`.
  2. **Memory Cleanup:** Frees LLM weights from GPU memory (`del model`, `torch.cuda.empty_cache()`).
  3. **SelfCheck-BERTScore:** $1 - \text{mean}(F_1)$ against resamples using `roberta-large`.
  4. **SelfCheck-NLI:** Mean contradiction probability $P(\text{contradiction})$ using `potsawee/deberta-v3-large-mnli`.
* **Evaluation Metric:** AUC-PR and pairwise accuracy on the test partition.
* **Auto-Checkpoints:** Saves resamples to `checkpoints/task2_resamples.json` and scores to `checkpoints/task2_selfcheck_scores.json`.

---

### Task 3: Stacked Ensemble Meta-Classifier (`task3.py`)
* **Objective:** Combine signals from Task 1 and Task 2 into a single meta-detector.
* **Method:**
  - Standardizes features: `mean_log_prob`, `mean_entropy`, `bertscore_inconsistency`, `nli_contradiction`.
  - Fits regularized Logistic Regression (`L2`) on the 80% train split.
  - Tunes hyperparameter $C \in [0.01, 100]$ using **5-Fold Grouped Cross-Validation** (grouped by question to avoid data leakage).
  - Evaluates AUC-PR on the untouched 20% test split.
  - Runs **Bootstrap (B = 2,000)** resampling to compute 95% confidence intervals and paired significance tests.
  - Conducts an ablation study (leave-one-out and extended variants).

---

## 4. Background Execution & Logging Mechanics

### Why `nohup` + `python -u`?
When running long jobs on remote clusters:
* `nohup`: Prevents the process from getting killed if SSH disconnects or the terminal closes.
* `python -u` & `sys.stdout.reconfigure(line_buffering=True)`: Disables output buffering. Without this, outputs remain stuck in Python's internal buffer for minutes before writing to disk.
* `> file.txt 2>&1 &`: Redirects both stdout and stderr to the log file and backgrounds the process.

### Runtime Tracking
Each script automatically records start time and prints total elapsed time at completion:
```text
============================================================
⏱ TOTAL EXECUTION TIME: 0h 24m 12s (1452.80 seconds)
============================================================
```

---

## 5. End-to-End Command Cheat Sheet

### Step 1: Connect to GPU Node
```bash
srun -p gpupart_p100 -w gnode5 --gres=gpu:1 --pty bash
cd ~/23CS30073_NLP_Assignment_2
source .venv/bin/activate
```

### Step 2: Launch Tasks

* **Run Task 1 in Background:**
  ```bash
  nohup python -u task1.py > task1_output.txt 2>&1 &
  ```

* **Run Task 2 in Background (after Task 1):**
  ```bash
  nohup python -u task2.py > task2_output.txt 2>&1 &
  ```

* **Run Task 3 in Background (after Task 1 & 2):**
  ```bash
  nohup python -u task3.py > task3_output.txt 2>&1 &
  ```

* **Or Run the Entire Pipeline Sequentially:**
  ```bash
  nohup bash run_all.sh > pipeline_output.txt 2>&1 &
  ```

---

## 6. Monitoring & Inspection

```bash
# Follow live output stream
tail -f task1_output.txt

# Check if script process is alive
ps aux | grep task1.py

# Monitor GPU memory and compute load
watch -n 2 nvidia-smi

# Check Slurm job queue
squeue -u $USER
```

---

## 7. Deliverables Generated

| File | Produced By | Purpose |
| :--- | :--- | :--- |
| `data/halueval_qa_sampled.json` | Data Setup | 400 sampled QA pairs (800 scoring instances) |
| `data/train_test_split.json` | Data Setup | 80/20 train/test split indices |
| `checkpoints/task1_logprobs_entropy.json` | Task 1 | Extracted log-probabilities and entropy |
| `outputs/task1_plots.png` | Task 1 | Score distribution histograms and PR curves |
| `outputs/task1_results.json` | Task 1 | AUC-PR metrics and summary statistics |
| `checkpoints/task2_resamples.json` | Task 2 | 20 stochastic samples generated per question |
| `checkpoints/task2_selfcheck_scores.json` | Task 2 | SelfCheck-BERTScore and SelfCheck-NLI scores |
| `outputs/task2_plots.png` | Task 2 | Consistency distribution and PR curves |
| `outputs/task2_results.json` | Task 2 | Task 2 AUC-PR and pairwise accuracy |
| `outputs/task3_meta_classifier.joblib` | Task 3 | Trained Logistic Regression model |
| `outputs/final_comparison.csv` | Task 3 | Final test AUC-PR comparison table + 95% CIs |
| `outputs/task3_pr_curves.png` | Task 3 | Precision-Recall curve comparison across methods |
| `outputs/task3_results.json` | Task 3 | Complete ensemble results, weights, and ablation |

