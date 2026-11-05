# Execution Commands Guide

This document lists all terminal commands to run, monitor, and manage **Task 1**, **Task 2**, and **Task 3** separately in the background on **Node 5 (`gnode5`)**.

---

## 0. Initial Setup & Node Allocation

If you ever need to reconnect or start a fresh session on Node 5:

```bash
# 1. Allocate an interactive session on Node 5 with 1 GPU (from master node)
srun -p gpupart_p100 -w gnode5 --gres=gpu:1 --pty bash

# 2. Navigate to project repository
cd ~/23CS30073_NLP_Assignment_2

# 3. Activate the virtual environment
source .venv/bin/activate
```

---

## 1. Task 1: Probability-Based Hallucination Detection

Computes **mean token log-probability** and **mean predictive entropy** using forced decoding on `google/gemma-2-2b-it`.

### Run in Background
```bash
nohup python -u task1.py > task1_output.txt 2>&1 &
```

### Monitor Live Output
```bash
tail -f task1_output.txt
```
*(Press `Ctrl + C` anytime to exit log viewing; the job keeps running).*

### Check Process Status
```bash
ps aux | grep task1.py
```

### Stop / Cancel Task 1 (If needed)
```bash
pkill -f task1.py
```

### Deliverables & Output Files
* **Log file**: `task1_output.txt` (includes runtime duration at the end)
* **Checkpoint**: `checkpoints/task1_logprobs_entropy.json` (auto-saves every 25 questions)
* **Metrics**: `outputs/task1_results.json`
* **Diagnostic plots**: `outputs/task1_plots.png`

---

## 2. Task 2: SelfCheckGPT (BERTScore & NLI)

Generates $N=20$ stochastic resamples per question with `gemma-2-2b-it`, then scores inconsistency with `roberta-large` (BERTScore) and `deberta-v3-large-mnli` (NLI).

> **Prerequisite:** Can run after Task 1 (or concurrently, but uses same GPU).

### Run in Background
```bash
nohup python -u task2.py > task2_output.txt 2>&1 &
```

### Monitor Live Output
```bash
tail -f task2_output.txt
```
*(Press `Ctrl + C` to exit log viewing).*

### Check Process Status
```bash
ps aux | grep task2.py
```

### Stop / Cancel Task 2 (If needed)
```bash
pkill -f task2.py
```

### Deliverables & Output Files
* **Log file**: `task2_output.txt` (includes runtime duration at the end)
* **Resamples checkpoint**: `checkpoints/task2_resamples.json`
* **Scoring checkpoint**: `checkpoints/task2_selfcheck_scores.json`
* **Metrics**: `outputs/task2_results.json`
* **Diagnostic plots**: `outputs/task2_plots.png`

---

## 3. Task 3: Stacked Ensemble Meta-Classifier

Fits a regularized Logistic Regression classifier using features from Task 1 and Task 2, and evaluates AUC-PR with 95% bootstrap confidence intervals on the 160-instance test set.

> **Prerequisite:** Requires `task1.py` and `task2.py` to be completed first. Runs on CPU in ~1–2 minutes.

### Run in Background
```bash
nohup python -u task3.py > task3_output.txt 2>&1 &
```

### Monitor Live Output
```bash
tail -f task3_output.txt
```

### Check Process Status
```bash
ps aux | grep task3.py
```

### Deliverables & Output Files
* **Log file**: `task3_output.txt` (includes runtime duration at the end)
* **Trained model**: `outputs/task3_meta_classifier.joblib`
* **Comparison table**: `outputs/final_comparison.csv`
* **Ablation & metrics**: `outputs/task3_results.json`
* **PR curves**: `outputs/task3_pr_curves.png`

---

## 4. Run All Tasks End-to-End (Full Pipeline)

If you want to run Task 1 $\to$ Task 2 $\to$ Task 3 sequentially in a single command:

### Run in Background
```bash
nohup bash run_all.sh > pipeline_output.txt 2>&1 &
```

### Monitor Live Output
```bash
tail -f pipeline_output.txt
```

---

## 5. Helpful Cluster Monitoring Commands

```bash
# Check GPU memory and compute usage
nvidia-smi

# Watch GPU usage update every 2 seconds
watch -n 2 nvidia-smi

# Check your Slurm jobs from any terminal
squeue -u $USER

# Check cluster node availability
sinfo
```

