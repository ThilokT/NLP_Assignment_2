#!/usr/bin/env python3
"""
Task 1: Knowledge-Conditioned Probability-Based Hallucination Detection
Adapted for Google Colab / Kaggle / Local
NLP Assignment 2 - Project 7 (HaluEval QA)
Model: google/gemma-2-2b-it

Optimizations:
1. Knowledge-Conditioned Prompt: "Context: {knowledge}\nQuestion: {question}\nAnswer: "
2. Expanded Feature Suite (9 signals):
   mean_log_prob, mean_entropy, sum_log_prob, log_prob_variance,
   entropy_variance, min_log_prob, max_entropy, perplexity, answer_token_count
3. Online HuggingFace Model & Dataset Loading (No offline caching restrictions)
4. Checkpointing to checkpoints/task1_features.json every 25 questions
5. Comprehensive test-set evaluation (AUC-PR, Pairwise Accuracy, 3x3 diagnostic plots)
"""

import datetime
import json
import math
import os
import random
import sys
import time
import urllib.request

# Ensure immediate unbuffered output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TOTAL_START_TIME = time.time()

import matplotlib
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
# 1. DIRECTORY SETUP
# ============================================================
# Find project root whether running from repo root or a subdirectory
_curr = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.path.dirname(_curr)) == "code":
    BASE_DIR = os.path.abspath(os.path.join(_curr, "..", ".."))
elif os.path.basename(_curr) == "code":
    BASE_DIR = os.path.abspath(os.path.join(_curr, ".."))
else:
    BASE_DIR = _curr

DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 75)
print("   TASK 1: KNOWLEDGE-CONDITIONED HALLUCINATION DETECTION")
print("=" * 75)
print(f"Working Directory: {BASE_DIR}")
print(f"Data Directory:    {DATA_DIR}")
print(f"Checkpoints:       {CKPT_DIR}")
print(f"Outputs:           {OUT_DIR}")
print(f"Timestamp:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# 2. AUTOMATIC DATA SETUP (Self-contained)
# ============================================================
sampled_path = os.path.join(DATA_DIR, "halueval_qa_sampled.json")
split_path   = os.path.join(DATA_DIR, "train_test_split.json")
raw_path     = os.path.join(DATA_DIR, "halueval_qa_raw.json")

RANDOM_SEED = 42
SAMPLE_SIZE = 400

if not os.path.exists(sampled_path) or not os.path.exists(split_path):
    print("\n[Data Setup] Required dataset files not found. Initializing dataset...")

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

    random.seed(RANDOM_SEED)
    sampled_data = random.sample(raw_data, SAMPLE_SIZE)
    for i, item in enumerate(sampled_data):
        item["sample_id"] = i

    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved sampled data ({len(sampled_data)} pairs) to: {sampled_path}")

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
hf_token = os.environ.get("HF_TOKEN", None)
if not hf_token:
    token_file = os.path.join(BASE_DIR, "token.txt")
    if os.path.exists(token_file):
        with open(token_file, "r") as f:
            hf_token = f.read().strip()

if not hf_token:
    hf_token = "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"

try:
    login(token=hf_token)
    print("✓ Hugging Face authenticated.")
except Exception as e:
    print(f"Hugging Face login notice: {e}")

MODEL_NAME = "google/gemma-2-2b-it"

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"\n✓ GPU Detected: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM Total:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    device = torch.device("cpu")
    print("\n⚠ WARNING: No GPU detected! Running on CPU will be slow.")

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
print("✓ Model loaded successfully!")
if torch.cuda.is_available():
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")


# ============================================================
# 4. FORCED DECODING FUNCTIONS (KNOWLEDGE-CONDITIONED)
# ============================================================
def compute_forced_decoding_features(knowledge, question, candidate_answer, model, tokenizer, device=None):
    if device is None:
        device = model.device

    context_str = str(knowledge).strip() if knowledge else ""
    question_str = str(question).strip() if question else ""
    prompt = f"Context: {context_str}\nQuestion: {question_str}\nAnswer: "

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    answer_ids = tokenizer.encode(str(candidate_answer), add_special_tokens=False, return_tensors="pt")

    num_prompt_tokens = prompt_ids.shape[1]
    num_answer_tokens = answer_ids.shape[1]

    if num_answer_tokens == 0:
        return {
            "mean_log_prob": 0.0,
            "mean_entropy": 0.0,
            "sum_log_prob": 0.0,
            "log_prob_variance": 0.0,
            "entropy_variance": 0.0,
            "min_log_prob": 0.0,
            "max_entropy": 0.0,
            "perplexity": 0.0,
            "answer_token_count": 0,
        }

    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    start_pos = num_prompt_tokens - 1
    end_pos   = input_ids.shape[1] - 1

    answer_logits = logits[0, start_pos:end_pos, :]
    answer_token_ids = input_ids[0, num_prompt_tokens:]

    # Upcast logits to float32 for complete numerical stability in fp16
    answer_logits_f32 = answer_logits.float()

    log_probs_all = F.log_softmax(answer_logits_f32, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=answer_token_ids.unsqueeze(-1)
    ).squeeze(-1)

    mean_log_prob = float(token_log_probs.mean().item())
    sum_log_prob  = float(token_log_probs.sum().item())
    min_log_prob  = float(token_log_probs.min().item())

    # PyTorch's native categorical entropy directly from float32 logits (avoids NaN)
    entropy_per_token = torch.distributions.Categorical(logits=answer_logits_f32).entropy()

    mean_entropy = float(entropy_per_token.mean().item())
    max_entropy  = float(entropy_per_token.max().item())

    if num_answer_tokens > 1:
        log_prob_variance = float(token_log_probs.var(unbiased=False).item())
        entropy_variance  = float(entropy_per_token.var(unbiased=False).item())
    else:
        log_prob_variance = 0.0
        entropy_variance  = 0.0

    try:
        if -mean_log_prob > 23.02585:
            perplexity = 1e10
        elif -mean_log_prob < -23.02585:
            perplexity = 0.0
        else:
            perplexity = float(min(math.exp(-mean_log_prob), 1e10))
    except OverflowError:
        perplexity = 1e10

    answer_token_count = int(num_answer_tokens)

    return {
        "mean_log_prob": mean_log_prob,
        "mean_entropy": mean_entropy,
        "sum_log_prob": sum_log_prob,
        "log_prob_variance": log_prob_variance,
        "entropy_variance": entropy_variance,
        "min_log_prob": min_log_prob,
        "max_entropy": max_entropy,
        "perplexity": perplexity,
        "answer_token_count": answer_token_count,
    }


# Quick sanity check on sample 0
sample0 = sampled_data[0]
print("\nRunning sanity check on Sample 0...")
s0_k = sample0.get("knowledge", "")
s0_q = sample0["question"]
feat_r0 = compute_forced_decoding_features(s0_k, s0_q, sample0["right_answer"], model, tokenizer)
feat_h0 = compute_forced_decoding_features(s0_k, s0_q, sample0["hallucinated_answer"], model, tokenizer)

print(f"  Right answer:        Mean LP={feat_r0['mean_log_prob']:.4f}, Entropy={feat_r0['mean_entropy']:.4f}, PPL={feat_r0['perplexity']:.4f}")
print(f"  Hallucinated answer: Mean LP={feat_h0['mean_log_prob']:.4f}, Entropy={feat_h0['mean_entropy']:.4f}, PPL={feat_h0['perplexity']:.4f}")


# ============================================================
# 5. FORCED DECODING OVER ALL QA PAIRS
# ============================================================
TASK1_CKPT = os.path.join(CKPT_DIR, "task1_features.json")
# Fallback to legacy checkpoint if task1_features.json does not exist yet
if not os.path.exists(TASK1_CKPT) and os.path.exists(os.path.join(CKPT_DIR, "task1_v2_features.json")):
    TASK1_CKPT = os.path.join(CKPT_DIR, "task1_v2_features.json")

if os.path.exists(TASK1_CKPT):
    with open(TASK1_CKPT, "r") as f:
        task1_results = json.load(f)
    
    # Check if checkpoint from a prior run contains corrupt NaNs
    has_nan = any(
        math.isnan(float(r["right_answer"].get("mean_entropy", 0.0))) or
        math.isnan(float(r["hallucinated_answer"].get("mean_entropy", 0.0)))
        for r in task1_results.values()
    )
    if has_nan:
        print("\n⚠ Detected NaN values in existing checkpoint from previous run.")
        print("  Invalidating cache and re-running forced decoding with float32 Categorical entropy...")
        task1_results = {}
    else:
        print(f"\n✓ Resuming from checkpoint: {len(task1_results)}/{len(sampled_data)} already processed.")
else:
    task1_results = {}
    print(f"\nStarting fresh forced decoding for {len(sampled_data)} QA pairs...")

start_time = time.time()
checkpoint_interval = 25

for i, item in enumerate(tqdm(sampled_data, desc="Task 1 Forced Decoding")):
    sid = str(item["sample_id"])
    if sid in task1_results:
        continue

    k = item.get("knowledge", "")
    q = item["question"]
    right_feat = compute_forced_decoding_features(k, q, item["right_answer"], model, tokenizer)
    hall_feat  = compute_forced_decoding_features(k, q, item["hallucinated_answer"], model, tokenizer)

    task1_results[sid] = {
        "sample_id": item["sample_id"],
        "original_id": item["id"],
        "right_answer": {
            **right_feat,
            "label": 0,
        },
        "hallucinated_answer": {
            **hall_feat,
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

with open(TASK1_CKPT, "w") as f:
    json.dump(task1_results, f, indent=2)

print(f"\n✓ Forced decoding complete in {(time.time() - start_time)/60:.1f} minutes.")
print(f"  Saved features to: {TASK1_CKPT}")


# ============================================================
# 6. EVALUATE AUC-PR & PAIRWISE ACCURACY
# ============================================================
print("\n" + "=" * 75)
print("   EVALUATING AUC-PR & PAIRWISE ACCURACY (TEST SET: 80 PAIRS, 160 INSTANCES)")
print("=" * 75)

FEATURES_CONFIG = [
    ("neg_mean_log_prob",  "mean_log_prob",       True,  "-mean_log_prob"),
    ("mean_entropy",       "mean_entropy",        False, "mean_entropy"),
    ("perplexity",         "perplexity",          False, "perplexity = exp(-mean_lp)"),
    ("neg_sum_log_prob",   "sum_log_prob",        True,  "-sum_log_prob"),
    ("neg_min_log_prob",   "min_log_prob",        True,  "-min_log_prob (worst token)"),
    ("max_entropy",        "max_entropy",         False, "max_entropy (peak uncertainty)"),
    ("log_prob_variance",  "log_prob_variance",   False, "var(token_log_probs)"),
    ("entropy_variance",   "entropy_variance",    False, "var(entropy_per_token)"),
    ("answer_token_count", "answer_token_count",  False, "num_answer_tokens"),
]

def evaluate_feature(results_dict, sample_ids, feat_key, invert=False):
    labels, scores, pairwise_correct = [], [], []

    for sid in sample_ids:
        res = results_dict[str(sid)]
        r_val = res["right_answer"][feat_key]
        h_val = res["hallucinated_answer"][feat_key]

        r_score = -float(r_val) if invert else float(r_val)
        h_score = -float(h_val) if invert else float(h_val)

        labels.append(0)
        scores.append(r_score)

        labels.append(1)
        scores.append(h_score)

        if h_score > r_score:
            pairwise_correct.append(1.0)
        elif h_score == r_score:
            pairwise_correct.append(0.5)
        else:
            pairwise_correct.append(0.0)

    labels = np.array(labels)
    scores = np.nan_to_num(np.array(scores, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    auc_pr = float(average_precision_score(labels, scores))
    pairwise_acc = float(np.mean(pairwise_correct))

    return {
        "labels": labels,
        "scores": scores,
        "auc_pr": auc_pr,
        "pairwise_acc": pairwise_acc,
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }

test_eval_results = {}
train_eval_results = {}

for name, key, invert, desc in FEATURES_CONFIG:
    test_eval_results[name] = evaluate_feature(task1_results, test_ids, key, invert=invert)
    train_eval_results[name] = evaluate_feature(task1_results, train_ids, key, invert=invert)

test_neg_lp = test_eval_results["neg_mean_log_prob"]["scores"]
test_ent    = test_eval_results["mean_entropy"]["scores"]
test_neg_lp_std = test_neg_lp.std() if test_neg_lp.std() > 0 else 1.0
test_ent_std    = test_ent.std() if test_ent.std() > 0 else 1.0

test_comb_z = np.nan_to_num((
    (test_neg_lp - test_neg_lp.mean()) / test_neg_lp_std +
    (test_ent - test_ent.mean()) / test_ent_std
) / 2.0, nan=0.0)

auc_pr_combined_test = float(average_precision_score(test_eval_results["neg_mean_log_prob"]["labels"], test_comb_z))

comb_pairwise = []
for i in range(len(test_ids)):
    r_comb = test_comb_z[2 * i]
    h_comb = test_comb_z[2 * i + 1]
    if h_comb > r_comb:
        comb_pairwise.append(1.0)
    elif h_comb == r_comb:
        comb_pairwise.append(0.5)
    else:
        comb_pairwise.append(0.0)
pairwise_acc_combined_test = float(np.mean(comb_pairwise))

test_eval_results["combined_z"] = {
    "labels": test_eval_results["neg_mean_log_prob"]["labels"],
    "scores": test_comb_z,
    "auc_pr": auc_pr_combined_test,
    "pairwise_acc": pairwise_acc_combined_test,
    "mean_score": float(np.mean(test_comb_z)),
    "std_score": float(np.std(test_comb_z)),
}

print(f"{'Feature Name':<22} | {'Test AUC-PR':<11} | {'Test Pairwise Acc':<17} | {'Train AUC-PR':<12} | {'Description'}")
print("-" * 88)
for name, key, invert, desc in FEATURES_CONFIG:
    t_auc = test_eval_results[name]["auc_pr"]
    t_acc = test_eval_results[name]["pairwise_acc"] * 100.0
    tr_auc = train_eval_results[name]["auc_pr"]
    print(f"{name:<22} | {t_auc:<11.4f} | {t_acc:>6.2f}% ({test_eval_results[name]['pairwise_acc']*len(test_ids):.1f}/{len(test_ids)}) | {tr_auc:<12.4f} | {desc}")

print("-" * 88)
print(f"{'combined_z':<22} | {auc_pr_combined_test:<11.4f} | {pairwise_acc_combined_test*100.0:>6.2f}% ({pairwise_acc_combined_test*len(test_ids):.1f}/{len(test_ids)}) | {'N/A':<12} | z(-mean_lp) + z(entropy)")
print("=" * 88)


# ============================================================
# 7. GENERATE DIAGNOSTIC PLOTS
# ============================================================
print("\nGenerating diagnostic plots...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
test_labels = test_eval_results["neg_mean_log_prob"]["labels"]

# Plot 1: Neg Mean Log-Prob
ax = axes[0, 0]
scores_nlp = test_eval_results["neg_mean_log_prob"]["scores"]
ax.hist(scores_nlp[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c")
ax.hist(scores_nlp[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728")
ax.set_title(f"Neg Mean Log-Prob (AUC-PR: {test_eval_results['neg_mean_log_prob']['auc_pr']:.4f})")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 2: Mean Predictive Entropy
ax = axes[0, 1]
scores_ent = test_eval_results["mean_entropy"]["scores"]
ax.hist(scores_ent[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c")
ax.hist(scores_ent[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728")
ax.set_title(f"Mean Predictive Entropy (AUC-PR: {test_eval_results['mean_entropy']['auc_pr']:.4f})")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 3: Perplexity
ax = axes[0, 2]
scores_ppl = test_eval_results["perplexity"]["scores"]
ppl_cap = float(np.percentile(scores_ppl, 98))
scores_ppl_plot = np.clip(scores_ppl, 0, ppl_cap)
ax.hist(scores_ppl_plot[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c")
ax.hist(scores_ppl_plot[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728")
ax.set_title(f"Perplexity clipped (AUC-PR: {test_eval_results['perplexity']['auc_pr']:.4f})")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 4: Neg Min Log-Prob
ax = axes[1, 0]
scores_min_lp = test_eval_results["neg_min_log_prob"]["scores"]
ax.hist(scores_min_lp[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c")
ax.hist(scores_min_lp[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728")
ax.set_title(f"Neg Min Log-Prob (AUC-PR: {test_eval_results['neg_min_log_prob']['auc_pr']:.4f})")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 5: Max Entropy
ax = axes[1, 1]
scores_max_ent = test_eval_results["max_entropy"]["scores"]
ax.hist(scores_max_ent[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c")
ax.hist(scores_max_ent[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728")
ax.set_title(f"Max Entropy (AUC-PR: {test_eval_results['max_entropy']['auc_pr']:.4f})")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 6: PR Curves Comparison
ax = axes[1, 2]
curve_features = [
    ("neg_mean_log_prob", "blue",   "-Mean Log-Prob"),
    ("perplexity",        "purple", "Perplexity"),
    ("mean_entropy",      "orange", "Mean Entropy"),
    ("neg_min_log_prob",  "brown",  "-Min Log-Prob"),
    ("combined_z",        "red",    "Combined Z-Score"),
]
for feat_name, col, lbl in curve_features:
    p_curve, r_curve, _ = precision_recall_curve(test_labels, test_eval_results[feat_name]["scores"])
    ax.plot(r_curve, p_curve, color=col, lw=2, label=f"{lbl} ({test_eval_results[feat_name]['auc_pr']:.3f})")

ax.axhline(0.50, color="gray", linestyle="--", label="Random (0.50)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("PR Curves Comparison")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 7: Pairwise Scatter Mean Log-Prob
ax = axes[2, 0]
right_lp = [task1_results[str(sid)]["right_answer"]["mean_log_prob"] for sid in test_ids]
hall_lp  = [task1_results[str(sid)]["hallucinated_answer"]["mean_log_prob"] for sid in test_ids]
correct_lp = [r > h for r, h in zip(right_lp, hall_lp)]
ax.scatter([r for r, c in zip(right_lp, correct_lp) if c],
           [h for h, c in zip(hall_lp, correct_lp) if c],
           color="#2ca02c", alpha=0.7, label=f"Right > Hall ({sum(correct_lp)})", s=30)
ax.scatter([r for r, c in zip(right_lp, correct_lp) if not c],
           [h for h, c in zip(hall_lp, correct_lp) if not c],
           color="#d62728", alpha=0.7, label=f"Hall >= Right ({len(correct_lp) - sum(correct_lp)})", s=30)
min_v = min(min(right_lp), min(hall_lp)) - 0.5
max_v = max(max(right_lp), max(hall_lp)) + 0.5
ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--")
ax.set_title(f"Pairwise Mean Log-Prob ({test_eval_results['neg_mean_log_prob']['pairwise_acc']*100:.1f}%)")
ax.set_xlim([min_v, max_v])
ax.set_ylim([min_v, max_v])
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 8: Pairwise Scatter Mean Entropy
ax = axes[2, 1]
right_ent = [task1_results[str(sid)]["right_answer"]["mean_entropy"] for sid in test_ids]
hall_ent  = [task1_results[str(sid)]["hallucinated_answer"]["mean_entropy"] for sid in test_ids]
correct_ent = [h > r for r, h in zip(right_ent, hall_ent)]
ax.scatter([r for r, c in zip(right_ent, correct_ent) if c],
           [h for h, c in zip(hall_ent, correct_ent) if c],
           color="#2ca02c", alpha=0.7, label=f"Hall > Right ({sum(correct_ent)})", s=30)
ax.scatter([r for r, c in zip(right_ent, correct_ent) if not c],
           [h for h, c in zip(hall_ent, correct_ent) if not c],
           color="#d62728", alpha=0.7, label=f"Right >= Hall ({len(correct_ent) - sum(correct_ent)})", s=30)
min_e = min(min(right_ent), min(hall_ent)) - 0.2
max_e = max(max(right_ent), max(hall_ent)) + 0.2
ax.plot([min_e, max_e], [min_e, max_e], color="black", linestyle="--")
ax.set_title(f"Pairwise Mean Entropy ({test_eval_results['mean_entropy']['pairwise_acc']*100:.1f}%)")
ax.set_xlim([min_e, max_e])
ax.set_ylim([min_e, max_e])
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 9: Performance Bar Chart
ax = axes[2, 2]
all_names = [cfg[0] for cfg in FEATURES_CONFIG] + ["combined_z"]
all_aucs  = [test_eval_results[n]["auc_pr"] for n in all_names]
all_accs  = [test_eval_results[n]["pairwise_acc"] for n in all_names]
y_positions = np.arange(len(all_names))
bar_width = 0.35

ax.barh(y_positions + bar_width/2, all_aucs, height=bar_width, label="AUC-PR", color="#1f77b4", alpha=0.85)
ax.barh(y_positions - bar_width/2, all_accs, height=bar_width, label="Pairwise Acc", color="#ff7f0e", alpha=0.85)
ax.axvline(0.50, color="gray", linestyle="--", label="Random (0.5)")
ax.set_yticks(y_positions)
ax.set_yticklabels(all_names, fontsize=8)
ax.set_title("Test Set Feature Performance")
ax.set_xlim([0.0, 1.05])
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "task1_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ Diagnostic plots saved to: {plot_path}")


# ============================================================
# 8. SAVE FINAL SUMMARY RESULTS
# ============================================================
summary_auc_pr = {}
summary_pairwise_acc = {}

for name in [cfg[0] for cfg in FEATURES_CONFIG] + ["combined_z"]:
    summary_auc_pr[name] = round(float(test_eval_results[name]["auc_pr"]), 4)
    summary_pairwise_acc[name] = round(float(test_eval_results[name]["pairwise_acc"]), 4)

task1_final = {
    "task": "Task 1: Knowledge-Conditioned Hallucination Detection",
    "model": MODEL_NAME,
    "prompt_template": "Context: {knowledge}\\nQuestion: {question}\\nAnswer: ",
    "sample_size": len(sampled_data),
    "random_seed": split_info["random_seed"],
    "metrics_auc_pr_test": summary_auc_pr,
    "metrics_pairwise_accuracy_test": summary_pairwise_acc,
    "checkpoint_file": TASK1_CKPT,
    "plot_file": plot_path,
}

results_path = os.path.join(OUT_DIR, "task1_results.json")
with open(results_path, "w") as f:
    json.dump(task1_final, f, indent=2)

print(f"\n✓ Summary results saved to: {results_path}")
print(f"✓ Checkpoint file saved to: {TASK1_CKPT}")

total_elapsed = time.time() - TOTAL_START_TIME
print(f"\n⏱ TOTAL EXECUTION TIME: {int(total_elapsed//60)}m {int(total_elapsed%60)}s")
print("✓ Task 1 complete!")

