#!/usr/bin/env python3
"""
Task 1: Probability-Based Hallucination Detection
NLP Assignment 2 - Project 7 (HaluEval QA)
Model: google/gemma-2-2b-it

Optimized for execution on cluster GPU nodes (e.g. gnode5 / Tesla P100).
"""

import json
import os
import random
import sys
import time
import urllib.request

# Ensure immediate unbuffered output to log files
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TOTAL_START_TIME = time.time()

import matplotlib
# Use headless backend for matplotlib on Linux server without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# ============================================================
# 1. DIRECTORY & ENVIRONMENT SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Working Directory: {BASE_DIR}")
print(f"Data Directory:    {DATA_DIR}")
print(f"Checkpoints:       {CKPT_DIR}")
print(f"Outputs:           {OUT_DIR}")


# ============================================================
# 2. AUTOMATIC DATA SETUP (Self-contained prerequisite check)
# ============================================================
sampled_path = os.path.join(DATA_DIR, "halueval_qa_sampled.json")
split_path   = os.path.join(DATA_DIR, "train_test_split.json")
raw_path     = os.path.join(DATA_DIR, "halueval_qa_raw.json")

RANDOM_SEED = 42
SAMPLE_SIZE = 400

if not os.path.exists(sampled_path) or not os.path.exists(split_path):
    print("\n[Data Setup] Required dataset files not found. Initializing dataset...")
    
    # 2.1 Download or load raw HaluEval QA dataset
    if os.path.exists(raw_path):
        print(f"  Loading raw data from cached {raw_path}...")
        with open(raw_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content.startswith("["):
            raw_data = json.loads(content)
        else:
            raw_data = []
            for idx, line in enumerate(content.splitlines()):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    raw_data.append({
                        "id": idx,
                        "knowledge": item.get("knowledge", ""),
                        "question": item["question"],
                        "right_answer": item["right_answer"],
                        "hallucinated_answer": item["hallucinated_answer"],
                    })
    else:
        url = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
        print(f"  Downloading HaluEval QA dataset from GitHub: {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            raw_text = response.read().decode("utf-8")

        raw_json_data = []
        for line in raw_text.strip().splitlines():
            line = line.strip()
            if line:
                raw_json_data.append(json.loads(line))

        raw_data = []
        for idx, item in enumerate(raw_json_data):
            raw_data.append({
                "id": idx,
                "knowledge": item.get("knowledge", ""),
                "question": item["question"],
                "right_answer": item["right_answer"],
                "hallucinated_answer": item["hallucinated_answer"],
            })

        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved raw data ({len(raw_data)} pairs) to: {raw_path}")

    # 2.2 Sample 400 questions with fixed seed
    random.seed(RANDOM_SEED)
    sampled_data = random.sample(raw_data, SAMPLE_SIZE)
    for i, item in enumerate(sampled_data):
        item["sample_id"] = i

    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved sampled data ({len(sampled_data)} pairs) to: {sampled_path}")

    # 2.3 Create 80/20 train/test split (320 train, 80 test)
    indices = list(range(len(sampled_data)))
    random.shuffle(indices)
    split_point = int(0.8 * len(sampled_data))
    train_ids = sorted(indices[:split_point])
    test_ids  = sorted(indices[split_point:])

    split_info = {
        "random_seed": RANDOM_SEED,
        "sample_size": len(sampled_data),
        "train_size": len(train_ids),
        "test_size": len(test_ids),
        "train_sample_ids": train_ids,
        "test_sample_ids": test_ids,
    }

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved 80/20 train/test split to: {split_path}")
else:
    print("\n✓ Found existing sampled data and train/test split.")
    with open(sampled_path, "r", encoding="utf-8") as f:
        sampled_data = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    train_ids = split_info["train_sample_ids"]
    test_ids  = split_info["test_sample_ids"]

print(f"  Total QA pairs: {len(sampled_data)}")
print(f"  Train: {len(train_ids)} questions → {len(train_ids)*2} scoring instances")
print(f"  Test:  {len(test_ids)} questions → {len(test_ids)*2} scoring instances")


# ============================================================
# 3. HUGGING FACE AUTHENTICATION & MODEL LOADING
# ============================================================
token_file = os.path.join(BASE_DIR, "token.txt")
hf_token = None
if os.path.exists(token_file):
    with open(token_file, "r") as f:
        hf_token = f.read().strip()

if not hf_token:
    # Fallback to token configured in notebook
    hf_token = "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"

try:
    login(token=hf_token)
    print("✓ Hugging Face authenticated.")
except Exception as e:
    print(f"Hugging Face login note: {e}")

MODEL_NAME = "google/gemma-2-2b-it"

# Check GPU hardware
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✓ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Total:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    device = torch.device("cpu")
    print("\n⚠ WARNING: No GPU detected. Running on CPU (will be significantly slower).")

print(f"\nLoading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
if not torch.cuda.is_available():
    model = model.to(device)

model.eval()
print(f"✓ Model loaded successfully!")
if torch.cuda.is_available():
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")


# ============================================================
# 4. FORCED DECODING FUNCTIONS
# ============================================================
def compute_forced_decoding_features(question, candidate_answer, model, tokenizer, device=None):
    """
    Run forced decoding on a (question, candidate_answer) pair.
    Returns:
        mean_log_prob (float): Mean token log-probability of the candidate answer.
        mean_entropy  (float): Mean predictive entropy at each answer position.
    """
    if device is None:
        device = model.device

    prompt = f"Question: {question}\nAnswer: "

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    answer_ids = tokenizer.encode(candidate_answer, add_special_tokens=False, return_tensors="pt")

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)
    num_prompt_tokens = prompt_ids.shape[1]
    num_answer_tokens = answer_ids.shape[1]

    if num_answer_tokens == 0:
        return 0.0, 0.0

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Position t-1 predicts position t
    start_pos = num_prompt_tokens - 1
    end_pos   = input_ids.shape[1] - 1

    answer_logits = logits[0, start_pos:end_pos, :]      # (num_answer_tokens, vocab_size)
    answer_token_ids = input_ids[0, num_prompt_tokens:]  # (num_answer_tokens,)

    # Feature 1: Mean Token Log-Probability
    log_probs_all = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=answer_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    mean_log_prob = token_log_probs.mean().item()

    # Feature 2: Mean Predictive Entropy
    probs = F.softmax(answer_logits, dim=-1)
    log_probs_for_entropy = torch.log(probs.clamp(min=1e-10))
    entropy_per_token = -(probs * log_probs_for_entropy).sum(dim=-1)
    mean_entropy = entropy_per_token.mean().item()

    return mean_log_prob, mean_entropy


# Sanity check on sample 0
sample0 = sampled_data[0]
print("\nRunning quick sanity check on Sample 0...")
mlp, me = compute_forced_decoding_features(sample0["question"], sample0["right_answer"], model, tokenizer)
mlp_h, me_h = compute_forced_decoding_features(sample0["question"], sample0["hallucinated_answer"], model, tokenizer)
print(f"  Right answer:        Mean Log-Prob = {mlp:.4f}, Mean Entropy = {me:.4f}")
print(f"  Hallucinated answer: Mean Log-Prob = {mlp_h:.4f}, Mean Entropy = {me_h:.4f}")
print(f"  Log-Prob check: {'PASS (right > hall)' if mlp > mlp_h else 'NOTE (hall >= right)'}")
print(f"  Entropy check:  {'PASS (right < hall)' if me < me_h else 'NOTE (right >= hall)'}")


# ============================================================
# 5. FORCED DECODING OVER ALL QA PAIRS (WITH CHECKPOINTING)
# ============================================================
TASK1_CKPT = os.path.join(CKPT_DIR, "task1_logprobs_entropy.json")

if os.path.exists(TASK1_CKPT):
    with open(TASK1_CKPT, "r") as f:
        task1_results = json.load(f)
    print(f"\n✓ Resuming from checkpoint: {len(task1_results)}/{len(sampled_data)} questions already processed.")
else:
    task1_results = {}
    print(f"\nStarting fresh Task 1 forced decoding for {len(sampled_data)} QA pairs...")

start_time = time.time()
checkpoint_interval = 25

for i, item in enumerate(tqdm(sampled_data, desc="Forced Decoding")):
    sid = str(item["sample_id"])
    if sid in task1_results:
        continue

    q = item["question"]
    right_mlp, right_me = compute_forced_decoding_features(q, item["right_answer"], model, tokenizer)
    hall_mlp, hall_me   = compute_forced_decoding_features(q, item["hallucinated_answer"], model, tokenizer)

    task1_results[sid] = {
        "sample_id": item["sample_id"],
        "original_id": item["id"],
        "right_answer": {
            "mean_log_prob": right_mlp,
            "mean_entropy": right_me,
            "label": 0,
        },
        "hallucinated_answer": {
            "mean_log_prob": hall_mlp,
            "mean_entropy": hall_me,
            "label": 1,
        },
    }

    if (i + 1) % checkpoint_interval == 0:
        with open(TASK1_CKPT, "w") as f:
            json.dump(task1_results, f, indent=2)
        elapsed = time.time() - start_time
        processed = len(task1_results)
        rate = elapsed / max(processed, 1)
        remaining = (len(sampled_data) - processed) * rate
        print(f"  Checkpoint saved ({processed}/{len(sampled_data)}). Est. remaining: {remaining/60:.1f} min")

# Final checkpoint save
with open(TASK1_CKPT, "w") as f:
    json.dump(task1_results, f, indent=2)

print(f"\n✓ Forced decoding complete in {(time.time() - start_time)/60:.1f} minutes.")
print(f"  Saved results to: {TASK1_CKPT}")


# ============================================================
# 6. EVALUATE AUC-PR ON TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("   EVALUATING AUC-PR ON TEST SET")
print("=" * 60)

test_labels = []
test_scores_neg_logprob = []
test_scores_entropy = []

for sid in test_ids:
    res = task1_results[str(sid)]
    # Right answer (label 0)
    test_labels.append(0)
    test_scores_neg_logprob.append(-res["right_answer"]["mean_log_prob"])
    test_scores_entropy.append(res["right_answer"]["mean_entropy"])
    # Hallucinated answer (label 1)
    test_labels.append(1)
    test_scores_neg_logprob.append(-res["hallucinated_answer"]["mean_log_prob"])
    test_scores_entropy.append(res["hallucinated_answer"]["mean_entropy"])

test_labels = np.array(test_labels)
test_scores_neg_logprob = np.array(test_scores_neg_logprob)
test_scores_entropy = np.array(test_scores_entropy)

auc_pr_logprob = average_precision_score(test_labels, test_scores_neg_logprob)
auc_pr_entropy = average_precision_score(test_labels, test_scores_entropy)

# Combined z-score
test_scores_combined = (
    test_scores_neg_logprob / test_scores_neg_logprob.std()
    + test_scores_entropy / test_scores_entropy.std()
) / 2
auc_pr_combined = average_precision_score(test_labels, test_scores_combined)

print(f"  Test instances        : {len(test_labels)} ({sum(test_labels==0)} faithful + {sum(test_labels==1)} hallucinated)")
print(f"  AUC-PR (neg log-prob) : {auc_pr_logprob:.4f}")
print(f"  AUC-PR (entropy)      : {auc_pr_entropy:.4f}")
print(f"  AUC-PR (combined z)   : {auc_pr_combined:.4f}")

# Train split for reference
train_labels = []
train_scores_neg_logprob = []
train_scores_entropy = []
for sid in train_ids:
    res = task1_results[str(sid)]
    train_labels.append(0)
    train_scores_neg_logprob.append(-res["right_answer"]["mean_log_prob"])
    train_scores_entropy.append(res["right_answer"]["mean_entropy"])
    train_labels.append(1)
    train_scores_neg_logprob.append(-res["hallucinated_answer"]["mean_log_prob"])
    train_scores_entropy.append(res["hallucinated_answer"]["mean_entropy"])

train_auc_logprob = average_precision_score(np.array(train_labels), np.array(train_scores_neg_logprob))
train_auc_entropy = average_precision_score(np.array(train_labels), np.array(train_scores_entropy))
print(f"  [Ref] Train AUC-PR (neg log-prob): {train_auc_logprob:.4f}")
print(f"  [Ref] Train AUC-PR (entropy)     : {train_auc_entropy:.4f}")
print("=" * 60)


# ============================================================
# 7. GENERATE & SAVE PLOTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Neg Log-Prob distribution
ax = axes[0, 0]
ax.hist(test_scores_neg_logprob[test_labels == 0], bins=30, alpha=0.6, label="Faithful (label=0)", color="green")
ax.hist(test_scores_neg_logprob[test_labels == 1], bins=30, alpha=0.6, label="Hallucinated (label=1)", color="red")
ax.set_xlabel("-Mean Log-Probability")
ax.set_ylabel("Count")
ax.set_title(f"Neg Log-Prob Distribution\n(AUC-PR = {auc_pr_logprob:.4f})")
ax.legend()

# Plot 2: Entropy distribution
ax = axes[0, 1]
ax.hist(test_scores_entropy[test_labels == 0], bins=30, alpha=0.6, label="Faithful (label=0)", color="green")
ax.hist(test_scores_entropy[test_labels == 1], bins=30, alpha=0.6, label="Hallucinated (label=1)", color="red")
ax.set_xlabel("Mean Predictive Entropy")
ax.set_ylabel("Count")
ax.set_title(f"Entropy Distribution\n(AUC-PR = {auc_pr_entropy:.4f})")
ax.legend()

# Plot 3: PR Curve (neg log-prob)
ax = axes[1, 0]
precision_lp, recall_lp, _ = precision_recall_curve(test_labels, test_scores_neg_logprob)
ax.plot(recall_lp, precision_lp, color="blue", lw=2)
ax.fill_between(recall_lp, precision_lp, alpha=0.1, color="blue")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR Curve — Neg Log-Prob\n(AUC-PR = {auc_pr_logprob:.4f})")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

# Plot 4: PR Curve (entropy)
ax = axes[1, 1]
precision_e, recall_e, _ = precision_recall_curve(test_labels, test_scores_entropy)
ax.plot(recall_e, precision_e, color="orange", lw=2)
ax.fill_between(recall_e, precision_e, alpha=0.1, color="orange")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"PR Curve — Entropy\n(AUC-PR = {auc_pr_entropy:.4f})")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "task1_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n✓ Diagnostic plots saved to: {plot_path}")


# ============================================================
# 8. SAVE FINAL SUMMARY RESULTS
# ============================================================
task1_final = {
    "model": MODEL_NAME,
    "dataset": "HaluEval QA (Li et al., 2023)",
    "sample_size": len(sampled_data),
    "random_seed": split_info["random_seed"],
    "test_size": len(test_ids),
    "test_scoring_instances": len(test_labels),
    "metrics": {
        "AUC-PR (neg_log_prob)": round(float(auc_pr_logprob), 4),
        "AUC-PR (entropy)": round(float(auc_pr_entropy), 4),
        "AUC-PR (combined_z)": round(float(auc_pr_combined), 4),
    },
    "train_reference_metrics": {
        "AUC-PR (neg_log_prob)": round(float(train_auc_logprob), 4),
        "AUC-PR (entropy)": round(float(train_auc_entropy), 4),
    },
    "feature_statistics": {
        "test_neg_logprob_mean": round(float(test_scores_neg_logprob.mean()), 4),
        "test_neg_logprob_std": round(float(test_scores_neg_logprob.std()), 4),
        "test_entropy_mean": round(float(test_scores_entropy.mean()), 4),
        "test_entropy_std": round(float(test_scores_entropy.std()), 4),
    },
}

results_path = os.path.join(OUT_DIR, "task1_results.json")
with open(results_path, "w") as f:
    json.dump(task1_final, f, indent=2)

print(f"✓ Summary results saved to:  {results_path}")
print(f"✓ Checkpoint file saved to:  {TASK1_CKPT}")

total_elapsed = time.time() - TOTAL_START_TIME
hours = int(total_elapsed // 3600)
minutes = int((total_elapsed % 3600) // 60)
seconds = int(total_elapsed % 60)

print("\n" + "=" * 60)
print(f"⏱ TOTAL EXECUTION TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.2f} seconds)")
print("=" * 60)
print("\n✓ Task 1 completed successfully!")

