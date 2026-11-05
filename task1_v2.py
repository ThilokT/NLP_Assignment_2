#!/usr/bin/env python3
"""
Task 1 v2: Knowledge-Conditioned Probability-Based Hallucination Detection
NLP Assignment 2 - Project 7 (HaluEval QA)
Model: google/gemma-2-2b-it

Optimizations over original Task 1:
1. Knowledge-Conditioned Prompt: Conditions on context ("Context: {knowledge}\\nQuestion: {question}\\nAnswer: ")
   rather than question alone, significantly enhancing discriminative token probabilities.
2. Expanded Feature Set: Extracts 9 token-level probability/entropy signals:
   - mean_log_prob, mean_entropy (original baseline signals)
   - sum_log_prob (unnormalized cumulative log-probability)
   - log_prob_variance (variance of per-token log-probs)
   - entropy_variance (variance of predictive entropy across token positions)
   - min_log_prob (worst-case token log probability)
   - max_entropy (peak predictive uncertainty position)
   - perplexity (exp(-mean_log_prob), clamped)
   - answer_token_count (length of candidate answer in tokens)
3. Dedicated Checkpointing: Saved to checkpoints/task1_v2_features.json every 25 questions.
4. Comprehensive Evaluation: Individual AUC-PR and Pairwise Ranking Accuracy on test partition.
5. Rich Diagnostic Visualizations: 3x3 diagnostic panel saved to outputs/task1_v2_plots.png.
6. Summary Results: Structured JSON exported to outputs/task1_v2_results.json.

Optimized for execution on cluster GPU nodes (e.g., Tesla P100 16GB).
"""

# ============================================================
# STANDARD LIBRARY IMPORTS
# ============================================================
import datetime
import json
import math
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

# ============================================================
# SCIENTIFIC & VISUALIZATION IMPORTS
# ============================================================
import matplotlib
# Headless backend for cluster nodes without active X display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

# ============================================================
# PYTORCH & TRANSFORMERS IMPORTS
# ============================================================
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login


# ============================================================
# 1. DIRECTORY & ENVIRONMENT SETUP
# ============================================================
# Enforce offline mode on cluster compute nodes without outbound internet
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("TASK 1 v2: KNOWLEDGE-CONDITIONED HALLUCINATION DETECTION")
print("=" * 70)
print(f"Working Directory: {BASE_DIR}")
print(f"Data Directory:    {DATA_DIR}")
print(f"Checkpoints:       {CKPT_DIR}")
print(f"Outputs:           {OUT_DIR}")
print(f"Timestamp:         {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


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
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
except Exception as e:
    print(f"  [Offline load notice] local_files_only raised ({e}), attempting standard load...")
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
# 4. OPTIMIZED FORCED DECODING FUNCTIONS
# ============================================================
def compute_forced_decoding_features(knowledge, question, candidate_answer, model, tokenizer, device=None):
    """
    Run knowledge-conditioned forced decoding on a (knowledge, question, candidate_answer) triplet.

    Prompt format (Optimization 1):
        "Context: {knowledge}\\nQuestion: {question}\\nAnswer: "

    Returns a dict with 9 features (Optimization 2):
        - mean_log_prob: Mean token log-probability of candidate answer tokens
        - mean_entropy: Mean predictive entropy across candidate answer positions
        - sum_log_prob: Cumulative log-probability of candidate answer tokens
        - log_prob_variance: Variance of token log probabilities (0.0 if num_tokens <= 1)
        - entropy_variance: Variance of predictive entropy across positions (0.0 if num_tokens <= 1)
        - min_log_prob: Minimum token log probability (highest uncertainty token)
        - max_entropy: Maximum predictive entropy across positions (peak uncertainty)
        - perplexity: exp(-mean_log_prob), clamped to 1e10 to avoid overflow
        - answer_token_count: Number of candidate answer tokens
    """
    if device is None:
        device = model.device

    context_str = str(knowledge).strip() if knowledge else ""
    question_str = str(question).strip() if question else ""
    prompt = f"Context: {context_str}\nQuestion: {question_str}\nAnswer: "

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    answer_ids = tokenizer.encode(str(candidate_answer), add_special_tokens=False, return_tensors="pt")

    num_prompt_tokens = prompt_ids.shape[1]
    num_answer_tokens = answer_ids.shape[1]

    # Handle edge case where candidate answer yields 0 tokens
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
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Position t-1 predicts position t
    start_pos = num_prompt_tokens - 1
    end_pos   = input_ids.shape[1] - 1

    answer_logits = logits[0, start_pos:end_pos, :]      # (num_answer_tokens, vocab_size)
    answer_token_ids = input_ids[0, num_prompt_tokens:]  # (num_answer_tokens,)

    # Log probabilities of actual candidate tokens
    log_probs_all = F.log_softmax(answer_logits, dim=-1)
    token_log_probs = log_probs_all.gather(
        dim=-1,
        index=answer_token_ids.unsqueeze(-1)
    ).squeeze(-1)  # shape: (num_answer_tokens,)

    # Feature 1: Mean Token Log-Probability
    mean_log_prob = float(token_log_probs.mean().item())

    # Feature 2: Sum Log-Probability (unnormalized cumulative total)
    sum_log_prob = float(token_log_probs.sum().item())

    # Feature 3: Min Log-Probability (most uncertain token)
    min_log_prob = float(token_log_probs.min().item())

    # Predictive entropy at each answer position
    probs = F.softmax(answer_logits, dim=-1)
    log_probs_for_entropy = torch.log(probs.clamp(min=1e-10))
    entropy_per_token = -(probs * log_probs_for_entropy).sum(dim=-1)  # shape: (num_answer_tokens,)

    # Feature 4: Mean Predictive Entropy
    mean_entropy = float(entropy_per_token.mean().item())

    # Feature 5: Max Predictive Entropy (most uncertain position)
    max_entropy = float(entropy_per_token.max().item())

    # Features 6 & 7: Variances (handle edge case where num_answer_tokens == 1)
    if num_answer_tokens > 1:
        log_prob_variance = float(token_log_probs.var(unbiased=False).item())
        entropy_variance = float(entropy_per_token.var(unbiased=False).item())
    else:
        log_prob_variance = 0.0
        entropy_variance = 0.0

    # Feature 8: Perplexity = exp(-mean_log_prob), clamped to avoid overflow
    try:
        if -mean_log_prob > 23.02585:  # ln(1e10) ≈ 23.02585
            perplexity = 1e10
        elif -mean_log_prob < -23.02585:
            perplexity = 0.0
        else:
            perplexity = float(min(math.exp(-mean_log_prob), 1e10))
    except OverflowError:
        perplexity = 1e10

    # Feature 9: Actual answer token count
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
print("\nRunning quick sanity check on Sample 0 (Knowledge-Conditioned)...")
s0_k = sample0.get("knowledge", "")
s0_q = sample0["question"]
feat_r0 = compute_forced_decoding_features(s0_k, s0_q, sample0["right_answer"], model, tokenizer)
feat_h0 = compute_forced_decoding_features(s0_k, s0_q, sample0["hallucinated_answer"], model, tokenizer)

print(f"  Right answer:        Mean LP={feat_r0['mean_log_prob']:.4f}, Entropy={feat_r0['mean_entropy']:.4f}, PPL={feat_r0['perplexity']:.4f}, Tokens={feat_r0['answer_token_count']}")
print(f"  Hallucinated answer: Mean LP={feat_h0['mean_log_prob']:.4f}, Entropy={feat_h0['mean_entropy']:.4f}, PPL={feat_h0['perplexity']:.4f}, Tokens={feat_h0['answer_token_count']}")
print(f"  Log-Prob check: {'PASS (right > hall)' if feat_r0['mean_log_prob'] > feat_h0['mean_log_prob'] else 'NOTE (hall >= right)'}")
print(f"  Entropy check:  {'PASS (right < hall)' if feat_r0['mean_entropy'] < feat_h0['mean_entropy'] else 'NOTE (right >= hall)'}")
print(f"  PPL check:      {'PASS (right < hall)' if feat_r0['perplexity'] < feat_h0['perplexity'] else 'NOTE (right >= hall)'}")


# ============================================================
# 5. FORCED DECODING OVER ALL QA PAIRS (WITH CHECKPOINTING)
# ============================================================
TASK1_V2_CKPT = os.path.join(CKPT_DIR, "task1_v2_features.json")

if os.path.exists(TASK1_V2_CKPT):
    with open(TASK1_V2_CKPT, "r") as f:
        task1_v2_results = json.load(f)
    print(f"\n✓ Resuming from checkpoint: {len(task1_v2_results)}/{len(sampled_data)} questions already processed.")
else:
    task1_v2_results = {}
    print(f"\nStarting fresh Task 1 v2 forced decoding for {len(sampled_data)} QA pairs...")

start_time = time.time()
checkpoint_interval = 25

for i, item in enumerate(tqdm(sampled_data, desc="Forced Decoding v2")):
    sid = str(item["sample_id"])
    if sid in task1_v2_results:
        continue

    k = item.get("knowledge", "")
    q = item["question"]
    right_feat = compute_forced_decoding_features(k, q, item["right_answer"], model, tokenizer)
    hall_feat  = compute_forced_decoding_features(k, q, item["hallucinated_answer"], model, tokenizer)

    task1_v2_results[sid] = {
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
        with open(TASK1_V2_CKPT, "w") as f:
            json.dump(task1_v2_results, f, indent=2)
        elapsed = time.time() - start_time
        processed = len(task1_v2_results)
        rate = elapsed / max(processed, 1)
        remaining = (len(sampled_data) - processed) * rate
        print(f"  Checkpoint saved ({processed}/{len(sampled_data)}). Est. remaining: {remaining/60:.1f} min")

# Final checkpoint save
with open(TASK1_V2_CKPT, "w") as f:
    json.dump(task1_v2_results, f, indent=2)

print(f"\n✓ Task 1 v2 forced decoding complete in {(time.time() - start_time)/60:.1f} minutes.")
print(f"  Saved features to: {TASK1_V2_CKPT}")


# ============================================================
# 6. EVALUATE AUC-PR & PAIRWISE ACCURACY ON TEST SPLIT
# ============================================================
print("\n" + "=" * 75)
print("   EVALUATING AUC-PR & PAIRWISE ACCURACY (TEST SET: 80 QA PAIRS, 160 INSTANCES)")
print("=" * 75)

# Definitions of all features to evaluate:
# (feature_display_name, dict_key, invert_score_for_hallucination_detection, formula_description)
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
    """
    Extracts binary labels and detection scores across sample_ids.
    Also computes pairwise accuracy (does score(hallucinated) > score(right)?).
    """
    labels = []
    scores = []
    pairwise_correct = []
    pairwise_ties = []

    for sid in sample_ids:
        res = results_dict[str(sid)]
        r_val = res["right_answer"][feat_key]
        h_val = res["hallucinated_answer"][feat_key]

        r_score = -float(r_val) if invert else float(r_val)
        h_score = -float(h_val) if invert else float(h_val)

        # Right answer (label 0)
        labels.append(0)
        scores.append(r_score)

        # Hallucinated answer (label 1)
        labels.append(1)
        scores.append(h_score)

        # Pairwise comparison: higher score indicates hallucination
        if h_score > r_score:
            pairwise_correct.append(1.0)
            pairwise_ties.append(0)
        elif h_score == r_score:
            pairwise_correct.append(0.5)
            pairwise_ties.append(1)
        else:
            pairwise_correct.append(0.0)
            pairwise_ties.append(0)

    labels = np.array(labels)
    scores = np.array(scores)
    auc_pr = float(average_precision_score(labels, scores))
    pairwise_acc = float(np.mean(pairwise_correct))

    return {
        "labels": labels,
        "scores": scores,
        "auc_pr": auc_pr,
        "pairwise_acc": pairwise_acc,
        "ties": int(sum(pairwise_ties)),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
    }

test_eval_results = {}
train_eval_results = {}

for name, key, invert, desc in FEATURES_CONFIG:
    test_eval_results[name] = evaluate_feature(task1_v2_results, test_ids, key, invert=invert)
    train_eval_results[name] = evaluate_feature(task1_v2_results, train_ids, key, invert=invert)

# Evaluation of Combined Z-Score: z(-mean_log_prob) + z(mean_entropy)
test_neg_lp = test_eval_results["neg_mean_log_prob"]["scores"]
test_ent    = test_eval_results["mean_entropy"]["scores"]

test_neg_lp_std = test_neg_lp.std() if test_neg_lp.std() > 0 else 1.0
test_ent_std    = test_ent.std() if test_ent.std() > 0 else 1.0

test_comb_z = (
    (test_neg_lp - test_neg_lp.mean()) / test_neg_lp_std +
    (test_ent - test_ent.mean()) / test_ent_std
) / 2.0

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

# Reference Train Combined Z-Score
train_neg_lp = train_eval_results["neg_mean_log_prob"]["scores"]
train_ent    = train_eval_results["mean_entropy"]["scores"]
train_neg_lp_std = train_neg_lp.std() if train_neg_lp.std() > 0 else 1.0
train_ent_std    = train_ent.std() if train_ent.std() > 0 else 1.0

train_comb_z = (
    (train_neg_lp - train_neg_lp.mean()) / train_neg_lp_std +
    (train_ent - train_ent.mean()) / train_ent_std
) / 2.0
auc_pr_combined_train = float(average_precision_score(train_eval_results["neg_mean_log_prob"]["labels"], train_comb_z))

test_eval_results["combined_z"] = {
    "labels": test_eval_results["neg_mean_log_prob"]["labels"],
    "scores": test_comb_z,
    "auc_pr": auc_pr_combined_test,
    "pairwise_acc": pairwise_acc_combined_test,
    "ties": 0,
    "mean_score": float(np.mean(test_comb_z)),
    "std_score": float(np.std(test_comb_z)),
}
train_eval_results["combined_z"] = {
    "auc_pr": auc_pr_combined_train,
}

# Print formatted summary table
print(f"{'Feature Name':<22} | {'Test AUC-PR':<11} | {'Test Pairwise Acc':<17} | {'Train AUC-PR':<12} | {'Description'}")
print("-" * 88)
for name, key, invert, desc in FEATURES_CONFIG:
    t_auc = test_eval_results[name]["auc_pr"]
    t_acc = test_eval_results[name]["pairwise_acc"] * 100.0
    tr_auc = train_eval_results[name]["auc_pr"]
    print(f"{name:<22} | {t_auc:<11.4f} | {t_acc:>6.2f}% ({test_eval_results[name]['pairwise_acc']*len(test_ids):.1f}/{len(test_ids)}) | {tr_auc:<12.4f} | {desc}")

print("-" * 88)
print(f"{'combined_z':<22} | {auc_pr_combined_test:<11.4f} | {pairwise_acc_combined_test*100.0:>6.2f}% ({pairwise_acc_combined_test*len(test_ids):.1f}/{len(test_ids)}) | {auc_pr_combined_train:<12.4f} | z(-mean_lp) + z(entropy)")
print("=" * 88)


# ============================================================
# 7. GENERATE DIAGNOSTIC PLOTS (3x3 VISUALIZATION PANEL)
# ============================================================
print("\nGenerating diagnostic plots...")
fig, axes = plt.subplots(3, 3, figsize=(18, 14))

test_labels = test_eval_results["neg_mean_log_prob"]["labels"]

# Plot 1: Neg Mean Log-Prob Distribution
ax = axes[0, 0]
scores_nlp = test_eval_results["neg_mean_log_prob"]["scores"]
ax.hist(scores_nlp[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c", edgecolor="black", linewidth=0.5)
ax.hist(scores_nlp[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_xlabel("-Mean Token Log-Probability")
ax.set_ylabel("Count")
ax.set_title(f"Neg Mean Log-Prob Distribution\nAUC-PR: {test_eval_results['neg_mean_log_prob']['auc_pr']:.4f} | Pairwise: {test_eval_results['neg_mean_log_prob']['pairwise_acc']*100:.1f}%")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 2: Mean Predictive Entropy Distribution
ax = axes[0, 1]
scores_ent = test_eval_results["mean_entropy"]["scores"]
ax.hist(scores_ent[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c", edgecolor="black", linewidth=0.5)
ax.hist(scores_ent[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Mean Predictive Entropy")
ax.set_ylabel("Count")
ax.set_title(f"Mean Predictive Entropy Distribution\nAUC-PR: {test_eval_results['mean_entropy']['auc_pr']:.4f} | Pairwise: {test_eval_results['mean_entropy']['pairwise_acc']*100:.1f}%")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 3: Perplexity Distribution (capped at 98th percentile for clear plotting)
ax = axes[0, 2]
scores_ppl = test_eval_results["perplexity"]["scores"]
ppl_cap = float(np.percentile(scores_ppl, 98))
scores_ppl_plot = np.clip(scores_ppl, 0, ppl_cap)
ax.hist(scores_ppl_plot[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c", edgecolor="black", linewidth=0.5)
ax.hist(scores_ppl_plot[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Perplexity (exp(-mean_log_prob))")
ax.set_ylabel("Count")
ax.set_title(f"Perplexity Distribution (clipped at 98%ile)\nAUC-PR: {test_eval_results['perplexity']['auc_pr']:.4f} | Pairwise: {test_eval_results['perplexity']['pairwise_acc']*100:.1f}%")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 4: Neg Min Log-Prob Distribution (Worst Token Uncertainty)
ax = axes[1, 0]
scores_min_lp = test_eval_results["neg_min_log_prob"]["scores"]
ax.hist(scores_min_lp[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c", edgecolor="black", linewidth=0.5)
ax.hist(scores_min_lp[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_xlabel("-Min Token Log-Probability (Worst Token)")
ax.set_ylabel("Count")
ax.set_title(f"Neg Min Log-Prob Distribution\nAUC-PR: {test_eval_results['neg_min_log_prob']['auc_pr']:.4f} | Pairwise: {test_eval_results['neg_min_log_prob']['pairwise_acc']*100:.1f}%")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 5: Max Entropy Distribution (Peak Uncertainty Position)
ax = axes[1, 1]
scores_max_ent = test_eval_results["max_entropy"]["scores"]
ax.hist(scores_max_ent[test_labels == 0], bins=25, alpha=0.6, label="Faithful (0)", color="#2ca02c", edgecolor="black", linewidth=0.5)
ax.hist(scores_max_ent[test_labels == 1], bins=25, alpha=0.6, label="Hallucinated (1)", color="#d62728", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Max Predictive Entropy (Peak Token)")
ax.set_ylabel("Count")
ax.set_title(f"Max Predictive Entropy Distribution\nAUC-PR: {test_eval_results['max_entropy']['auc_pr']:.4f} | Pairwise: {test_eval_results['max_entropy']['pairwise_acc']*100:.1f}%")
ax.legend(loc="upper right")
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 6: Multi-Feature Precision-Recall Curves
ax = axes[1, 2]
curve_features = [
    ("neg_mean_log_prob", "blue",       "-Mean Log-Prob"),
    ("perplexity",        "purple",     "Perplexity"),
    ("mean_entropy",      "orange",     "Mean Entropy"),
    ("neg_min_log_prob",  "brown",      "-Min Log-Prob"),
    ("combined_z",        "red",        "Combined Z-Score"),
]
for feat_name, col, lbl in curve_features:
    p_curve, r_curve, _ = precision_recall_curve(test_labels, test_eval_results[feat_name]["scores"])
    ax.plot(r_curve, p_curve, color=col, lw=2, label=f"{lbl} ({test_eval_results[feat_name]['auc_pr']:.3f})")

ax.axhline(0.50, color="gray", linestyle="--", label="Random (0.50)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves Comparison")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="lower left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 7: Pairwise Scatter: Right vs Hallucinated Mean Log-Prob
ax = axes[2, 0]
right_lp = [task1_v2_results[str(sid)]["right_answer"]["mean_log_prob"] for sid in test_ids]
hall_lp  = [task1_v2_results[str(sid)]["hallucinated_answer"]["mean_log_prob"] for sid in test_ids]
correct_lp = [r > h for r, h in zip(right_lp, hall_lp)]
ax.scatter([r for r, c in zip(right_lp, correct_lp) if c],
           [h for h, c in zip(hall_lp, correct_lp) if c],
           color="#2ca02c", alpha=0.7, label=f"Right > Hall ({sum(correct_lp)})", s=30)
ax.scatter([r for r, c in zip(right_lp, correct_lp) if not c],
           [h for h, c in zip(hall_lp, correct_lp) if not c],
           color="#d62728", alpha=0.7, label=f"Hall >= Right ({len(correct_lp) - sum(correct_lp)})", s=30)
min_v = min(min(right_lp), min(hall_lp)) - 0.5
max_v = max(max(right_lp), max(hall_lp)) + 0.5
ax.plot([min_v, max_v], [min_v, max_v], color="black", linestyle="--", label="Diagonal (y = x)")
ax.set_xlabel("Right Answer Mean Log-Prob")
ax.set_ylabel("Hallucinated Answer Mean Log-Prob")
ax.set_title(f"Pairwise Mean Log-Prob\nAccuracy: {test_eval_results['neg_mean_log_prob']['pairwise_acc']*100:.1f}%")
ax.set_xlim([min_v, max_v])
ax.set_ylim([min_v, max_v])
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 8: Pairwise Scatter: Right vs Hallucinated Mean Entropy
ax = axes[2, 1]
right_ent = [task1_v2_results[str(sid)]["right_answer"]["mean_entropy"] for sid in test_ids]
hall_ent  = [task1_v2_results[str(sid)]["hallucinated_answer"]["mean_entropy"] for sid in test_ids]
correct_ent = [h > r for r, h in zip(right_ent, hall_ent)]
ax.scatter([r for r, c in zip(right_ent, correct_ent) if c],
           [h for h, c in zip(hall_ent, correct_ent) if c],
           color="#2ca02c", alpha=0.7, label=f"Hall > Right ({sum(correct_ent)})", s=30)
ax.scatter([r for r, c in zip(right_ent, correct_ent) if not c],
           [h for h, c in zip(hall_ent, correct_ent) if not c],
           color="#d62728", alpha=0.7, label=f"Right >= Hall ({len(correct_ent) - sum(correct_ent)})", s=30)
min_e = min(min(right_ent), min(hall_ent)) - 0.2
max_e = max(max(right_ent), max(hall_ent)) + 0.2
ax.plot([min_e, max_e], [min_e, max_e], color="black", linestyle="--", label="Diagonal (y = x)")
ax.set_xlabel("Right Answer Mean Entropy")
ax.set_ylabel("Hallucinated Answer Mean Entropy")
ax.set_title(f"Pairwise Mean Entropy\nAccuracy: {test_eval_results['mean_entropy']['pairwise_acc']*100:.1f}%")
ax.set_xlim([min_e, max_e])
ax.set_ylim([min_e, max_e])
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)

# Plot 9: Horizontal Bar Chart of All Features (AUC-PR & Pairwise Accuracy)
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
ax.set_xlabel("Score")
ax.set_title("Test Set Feature Performance")
ax.set_xlim([0.0, 1.05])
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "task1_v2_plots.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"✓ Diagnostic plots saved to: {plot_path}")


# ============================================================
# 8. SAVE FINAL SUMMARY RESULTS
# ============================================================
summary_auc_pr = {}
summary_pairwise_acc = {}
summary_train_auc_pr = {}
feature_stats = {}

for name in [cfg[0] for cfg in FEATURES_CONFIG] + ["combined_z"]:
    summary_auc_pr[name] = round(float(test_eval_results[name]["auc_pr"]), 4)
    summary_pairwise_acc[name] = round(float(test_eval_results[name]["pairwise_acc"]), 4)
    summary_train_auc_pr[name] = round(float(train_eval_results[name]["auc_pr"]), 4)
    feature_stats[f"{name}_test_mean"] = round(float(test_eval_results[name]["mean_score"]), 4)
    feature_stats[f"{name}_test_std"]  = round(float(test_eval_results[name]["std_score"]), 4)

task1_v2_final = {
    "task": "Task 1 v2: Knowledge-Conditioned Probability-Based Hallucination Detection",
    "model": MODEL_NAME,
    "dataset": "HaluEval QA (Li et al., 2023)",
    "prompt_template": "Context: {knowledge}\\nQuestion: {question}\\nAnswer: ",
    "sample_size": len(sampled_data),
    "random_seed": split_info["random_seed"],
    "train_size": len(train_ids),
    "test_size": len(test_ids),
    "test_scoring_instances": len(test_labels),
    "metrics_auc_pr_test": summary_auc_pr,
    "metrics_pairwise_accuracy_test": summary_pairwise_acc,
    "train_reference_auc_pr": summary_train_auc_pr,
    "feature_statistics_test": feature_stats,
    "checkpoint_file": TASK1_V2_CKPT,
    "plot_file": plot_path,
}

results_path = os.path.join(OUT_DIR, "task1_v2_results.json")
with open(results_path, "w") as f:
    json.dump(task1_v2_final, f, indent=2)

print(f"\n✓ Summary results saved to:  {results_path}")
print(f"✓ Checkpoint file saved to:  {TASK1_V2_CKPT}")

total_elapsed = time.time() - TOTAL_START_TIME
hours = int(total_elapsed // 3600)
minutes = int((total_elapsed % 3600) // 60)
seconds = int(total_elapsed % 60)

print("\n" + "=" * 70)
print(f"⏱ TOTAL EXECUTION TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.2f} seconds)")
print("=" * 70)
print("\n✓ Task 1 v2 completed successfully!")
