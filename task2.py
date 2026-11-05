#!/usr/bin/env python3
"""
Task 2: SelfCheckGPT (SelfCheck-BERTScore & SelfCheck-NLI)
NLP Assignment 2 - Project 7 (HaluEval QA)
LLM: google/gemma-2-2b-it
Scorers: roberta-large (BERTScore), potsawee/deberta-v3-large-mnli (NLI)

Optimized for execution on cluster GPU nodes (e.g. gnode5 / Tesla P100).
"""

import gc
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
# Headless backend for Linux cluster without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login


# ============================================================
# 1. DIRECTORY & ENVIRONMENT SETUP
# ============================================================
# Enable offline mode for Hugging Face on compute nodes without internet
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
# 2. DATASET VERIFICATION & AUTOMATIC FALLBACK
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

    random.seed(RANDOM_SEED)
    sampled_data = random.sample(raw_data, SAMPLE_SIZE)
    for i, item in enumerate(sampled_data):
        item["sample_id"] = i

    with open(sampled_path, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

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
else:
    print("\n✓ Found existing sampled data and train/test split.")
    with open(sampled_path, "r", encoding="utf-8") as f:
        sampled_data = json.load(f)
    with open(split_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)
    train_ids = split_info["train_sample_ids"]
    test_ids  = split_info["test_sample_ids"]

all_ids = sorted(train_ids + test_ids)
print(f"  Total QA pairs: {len(sampled_data)} | Train: {len(train_ids)} | Test: {len(test_ids)}")


# ============================================================
# 3. HUGGING FACE AUTHENTICATION
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


# ============================================================
# 4. STEP 1: GENERATE STOCHASTIC RESAMPLES (N = 20)
# ============================================================
RES_CKPT = os.path.join(CKPT_DIR, "task2_resamples.json")
resamples = json.load(open(RES_CKPT)) if os.path.exists(RES_CKPT) else {}

N_SAMPLES      = 20
TEMPERATURE    = 1.0
TOP_P          = 1.0
TOP_K          = 0       # 0 = disabled -> true sampling from the full distribution
MAX_NEW_TOKENS = 48
GEN_SEED       = 42
MODEL_NAME     = "google/gemma-2-2b-it"

if len(resamples) < len(sampled_data):
    print(f"\n[Step 1] Loading LLM ({MODEL_NAME}) for resampling...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        local_files_only=True,
    )
    model.eval()
    if torch.cuda.is_available():
        print(f"✓ LLM loaded on GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def build_prompt(question):
        msg = ("Answer the following question with a short, direct answer "
               "(a phrase or one sentence). Do not explain.\n\n"
               f"Question: {question}")
        return tokenizer.apply_chat_template([{"role": "user", "content": msg}],
                                             tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def resample_answers(question, seed):
        enc = tokenizer(build_prompt(question), return_tensors="pt", add_special_tokens=False).to(model.device)
        torch.manual_seed(seed)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=MAX_NEW_TOKENS,
            num_return_sequences=N_SAMPLES,
            pad_token_id=tokenizer.pad_token_id,
        )
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        cleaned = []
        for t in texts:
            t = t.strip().split("\n")[0].strip()
            cleaned.append(t if t else "N/A")
        return cleaned

    print(f"\nResampling {len(sampled_data)} questions (N={N_SAMPLES} each)...")
    print(f"  Resuming with {len(resamples)} already done.")
    t0 = time.time()
    for i, item in enumerate(tqdm(sampled_data, desc="Resampling")):
        sid = str(item["sample_id"])
        if sid in resamples:
            continue
        resamples[sid] = resample_answers(item["question"], GEN_SEED + item["sample_id"])
        if (i + 1) % 25 == 0:
            with open(RES_CKPT, "w", encoding="utf-8") as f:
                json.dump(resamples, f, ensure_ascii=False)
            print(f"  Resample checkpoint saved ({len(resamples)}/{len(sampled_data)})")

    with open(RES_CKPT, "w", encoding="utf-8") as f:
        json.dump(resamples, f, ensure_ascii=False)
    print(f"✓ Resampling completed in {(time.time()-t0)/60:.1f} minutes | Saved: {RES_CKPT}")

    # Free LLM memory before loading BERTScore and NLI models
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  GPU memory after freeing LLM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    print(f"\n✓ Found complete resamples checkpoint ({len(resamples)} questions): {RES_CKPT}")


# ============================================================
# 5. STEP 2: LOAD SCORERS (BERTScore & DeBERTa NLI)
# ============================================================
device_str = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[Step 2] Loading BERTScore (roberta-large) & NLI models on {device_str}...")

from bert_score import BERTScorer

bert_scorer = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=False,
    device=device_str,
)

NLI_NAME = "potsawee/deberta-v3-large-mnli"
nli_tok   = AutoTokenizer.from_pretrained(NLI_NAME, local_files_only=True, use_fast=False)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_NAME, local_files_only=True).to(device_str).eval()

id2label = {int(k): v.lower() for k, v in nli_model.config.id2label.items()}
ENT = next((k for k, v in id2label.items() if v.startswith("entail")), 0)
CON = next((k for k, v in id2label.items() if v.startswith("contra")), 2)
print(f"✓ Scorers loaded. NLI indices -> Entailment: {ENT}, Contradiction: {CON}")


@torch.no_grad()
def p_contradiction(premises, hypotheses, bs=32):
    out = []
    vocab_size = getattr(nli_model.config, "vocab_size", 128100)
    for i in range(0, len(premises), bs):
        enc = nli_tok(
            premises[i:i+bs],
            hypotheses[i:i+bs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_token_type_ids=False,
        )
        enc.pop("token_type_ids", None)
        if "input_ids" in enc:
            enc["input_ids"] = torch.clamp(enc["input_ids"], min=0, max=vocab_size - 1)
        enc = {k: v.to(device_str) for k, v in enc.items()}
        logits = nli_model(**enc).logits
        num_classes = logits.shape[-1]
        ent_idx = min(ENT, num_classes - 1)
        con_idx = min(CON, num_classes - 1)
        two = torch.stack([logits[:, ent_idx], logits[:, con_idx]], dim=-1)
        contra_probs = torch.softmax(two, dim=-1)[:, 1]
        out.extend(contra_probs.cpu().tolist())
    return out


def bertscore_f1(cands, refs):
    _, _, F = bert_scorer.score(cands, refs, batch_size=64)
    return F.tolist()


def score_candidate(question, cand, samples):
    f1 = bertscore_f1([cand] * len(samples), samples)
    nli_bare = p_contradiction(samples, [cand] * len(samples))
    nli_ctx  = p_contradiction(
        [f"{question} {s}" for s in samples],
        [f"{question} {cand}"] * len(samples)
    )
    return {
        "bertscore_inconsistency": 1 - float(np.mean(f1)),
        "bertscore_inconsistency_max": 1 - float(np.max(f1)),
        "nli_contradiction": float(np.mean(nli_bare)),
        "nli_contradiction_ctx": float(np.mean(nli_ctx)),
    }


# ============================================================
# 6. STEP 3: SCORE ALL CANDIDATES AGAINST RESAMPLES
# ============================================================
SC_CKPT = os.path.join(CKPT_DIR, "task2_selfcheck_scores.json")
task2_scores = json.load(open(SC_CKPT)) if os.path.exists(SC_CKPT) else {}

print(f"\n[Step 3] SelfCheck scoring ({len(task2_scores)} already completed)...")
t0 = time.time()

for i, item in enumerate(tqdm(sampled_data, desc="SelfCheck scoring")):
    sid = str(item["sample_id"])
    if sid in task2_scores:
        continue
    s, q = resamples[sid], item["question"]
    r = score_candidate(q, item["right_answer"], s)
    r["label"] = 0
    h = score_candidate(q, item["hallucinated_answer"], s)
    h["label"] = 1

    task2_scores[sid] = {
        "sample_id": item["sample_id"],
        "original_id": item["id"],
        "right_answer": r,
        "hallucinated_answer": h,
    }

    if (i + 1) % 25 == 0:
        with open(SC_CKPT, "w", encoding="utf-8") as f:
            json.dump(task2_scores, f, indent=1)
        print(f"  Scoring checkpoint saved ({len(task2_scores)}/{len(sampled_data)})")

with open(SC_CKPT, "w", encoding="utf-8") as f:
    json.dump(task2_scores, f, indent=1)
print(f"✓ Scoring finished in {(time.time()-t0)/60:.1f} minutes | Saved: {SC_CKPT}")


# ============================================================
# 7. STEP 4: AUC-PR EVALUATION
# ============================================================
def collect(ids, key):
    y, s = [], []
    for sid in ids:
        r = task2_scores[str(sid)]
        for t, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            y.append(lab)
            s.append(r[t][key])
    return np.array(y), np.array(s)


def pairwise_acc(ids, key):
    return float(np.mean([
        task2_scores[str(i)]["hallucinated_answer"][key] >
        task2_scores[str(i)]["right_answer"][key]
        for i in ids
    ]))


KEYS = {
    "SelfCheck-BERTScore (mean, primary)": "bertscore_inconsistency",
    "SelfCheck-NLI (primary)":             "nli_contradiction",
    "BERTScore (max-over-samples, ablation)": "bertscore_inconsistency_max",
    "NLI + question context (ablation)":   "nli_contradiction_ctx",
}

results = {}
print("\n" + "=" * 65)
print("   TASK 2 RESULTS — AUC-PR & PAIRWISE ACCURACY")
print("=" * 65)

for split_name, ids in [("full (all 800 instances)", all_ids), ("test split (160 instances)", test_ids)]:
    print(f"\n=== {split_name} ===")
    results[split_name] = {}
    for name, key in KEYS.items():
        y, s = collect(ids, key)
        ap = average_precision_score(y, s)
        pw = pairwise_acc(ids, key)
        results[split_name][name] = {"AUC-PR": round(float(ap), 4), "pairwise_acc": round(pw, 4)}
        print(f"  {name:42s} AUC-PR = {ap:.4f}   pairwise acc = {pw:.3f}")


# ============================================================
# 8. STEP 5: DIAGNOSTIC PLOTS
# ============================================================
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for col, (name, key) in enumerate([("SelfCheck-BERTScore", "bertscore_inconsistency"), ("SelfCheck-NLI", "nli_contradiction")]):
    y, s = collect(all_ids, key)
    ax[0, col].hist(s[y == 0], bins=30, alpha=.6, color="green", label="Right (0)")
    ax[0, col].hist(s[y == 1], bins=30, alpha=.6, color="red", label="Hallucinated (1)")
    ax[0, col].set_title(f"{name} Score Distribution")
    ax[0, col].legend()

    p, r, _ = precision_recall_curve(y, s)
    ax[1, col].plot(r, p, lw=2)
    ax[1, col].axhline(0.5, ls="--", c="gray")
    ax[1, col].set_xlabel("Recall")
    ax[1, col].set_ylabel("Precision")
    ax[1, col].set_title(f"PR Curve (AUC-PR={average_precision_score(y, s):.3f})")

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "task2_plots.png")
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\n✓ Diagnostic plots saved to: {plot_path}")


# ============================================================
# 9. STEP 6: SAVE SUMMARY RESULTS
# ============================================================
final = {
    "llm": MODEL_NAME,
    "bertscore_backbone": "roberta-large",
    "nli_model": NLI_NAME,
    "resampling_protocol": {
        "N": N_SAMPLES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": f"{GEN_SEED}+sample_id",
        "prompt": "chat template, short direct answer, closed-book (no knowledge passage)",
    },
    "random_seed_split": split_info["random_seed"],
    "metrics": results,
}

res_path = os.path.join(OUT_DIR, "task2_results.json")
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)

print(f"✓ Final results saved to: {res_path}")

total_elapsed = time.time() - TOTAL_START_TIME
hours = int(total_elapsed // 3600)
minutes = int((total_elapsed % 3600) // 60)
seconds = int(total_elapsed % 60)

print("\n" + "=" * 60)
print(f"⏱ TOTAL EXECUTION TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.2f} seconds)")
print("=" * 60)
print("✓ Task 2 complete! Features ready for Task 3.")

