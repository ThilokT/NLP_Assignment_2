#!/usr/bin/env python3
"""
Task 2 V2: Optimized SelfCheckGPT (SelfCheck-BERTScore & SelfCheck-NLI)
NLP Assignment 2 - Project 7 (HaluEval QA)
LLM: google/gemma-2-2b-it
Scorers: roberta-large (BERTScore), microsoft/deberta-large-mnli or potsawee/deberta-v3-large-mnli (NLI)

Key Optimizations in V2:
1. Higher sampling temperature (T=1.2, top_p=0.95, max_new_tokens=64) for richer sample diversity.
2. 3-class NLI model (microsoft/deberta-large-mnli) with empirical auto-detection of ENT/CON classes.
   Graceful fallback to potsawee/deberta-v3-large-mnli with tokenizer and tensor safeguards.
3. Question-prepended BERTScore giving rich context to candidate and resample comparisons.
4. Expanded feature suite (8 features per candidate):
   - bertscore_inconsistency: 1 - mean(F1) bare
   - bertscore_inconsistency_ctx: 1 - mean(F1) question-prepended
   - bertscore_inconsistency_max: 1 - max(F1) bare
   - nli_contradiction: mean P(contradiction) bare
   - nli_contradiction_ctx: mean P(contradiction) with question context
   - nli_entailment: 1 - mean P(entailment) bare (inverted entailment)
   - nli_max_contradiction: max P(contradiction) across resamples
   - semantic_entropy_proxy: std of contradiction probabilities (model inconsistency)
5. Robust checkpointing every 25 questions and evaluation on held-out test split (160 instances) & full set (800 instances).

Optimized for execution on cluster GPU nodes (e.g. gnode5 / Tesla P100 16GB).
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
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login


# ============================================================
# 1. DIRECTORY & ENVIRONMENT SETUP
# ============================================================
# Offline mode for compute nodes without internet access
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Robust project root resolution (works from repo root or code/task2/)
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
print("   TASK 2 V2: OPTIMIZED SELFCHECKGPT HALLUCINATION DETECTOR")
print("=" * 75)
print(f"Working Directory: {BASE_DIR}")
print(f"Data Directory:    {DATA_DIR}")
print(f"Checkpoints:       {CKPT_DIR}")
print(f"Outputs:           {OUT_DIR}")


# ============================================================
# 2. DATASET VERIFICATION & LOADING
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
# 4. STEP 1: GENERATE STOCHASTIC RESAMPLES (N = 20, T = 1.2)
# ============================================================
RES_CKPT = os.path.join(CKPT_DIR, "task2_resamples_v2.json")
resamples = json.load(open(RES_CKPT)) if os.path.exists(RES_CKPT) else {}

N_SAMPLES      = 20
TEMPERATURE    = 1.2       # Increased from 1.0 to 1.2 to enhance resample diversity
TOP_P          = 0.95      # Nucleus sampling for high quality with variety
TOP_K          = 0         # Disabled top_k for pure nucleus sampling
MAX_NEW_TOKENS = 64        # Extended tokens to avoid truncation of short phrases
GEN_SEED       = 42
MODEL_NAME     = "google/gemma-2-2b-it"

if len(resamples) < len(sampled_data):
    print(f"\n[Step 1 / Section 4] Loading LLM ({MODEL_NAME}) for resampling...")
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

    print(f"\nResampling {len(sampled_data)} questions (N={N_SAMPLES}, T={TEMPERATURE}, top_p={TOP_P})...")
    print(f"  Resuming with {len(resamples)} already cached.")
    t0 = time.time()
    for i, item in enumerate(tqdm(sampled_data, desc="Resampling (V2)")):
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
print(f"\n[Step 2 / Section 5] Loading BERTScore (roberta-large) & NLI models on {device_str}...")

from bert_score import BERTScorer

bert_scorer = BERTScorer(
    model_type="roberta-large",
    lang="en",
    rescale_with_baseline=False,
    device=device_str,
)
print("✓ Loaded BERTScorer: roberta-large")

# Try loading microsoft/deberta-large-mnli (3-class), fallback to potsawee/deberta-v3-large-mnli
NLI_NAME = None
is_potsawee = False

try:
    NLI_CANDIDATE = "microsoft/deberta-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(NLI_CANDIDATE, local_files_only=True)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_CANDIDATE, local_files_only=True
    ).to(device_str).eval()
    NLI_NAME = NLI_CANDIDATE
    is_potsawee = False
    print(f"✓ Loaded 3-class NLI model: {NLI_NAME}")
except Exception as e:
    print(f"⚠ Could not load microsoft/deberta-large-mnli ({e})")
    NLI_CANDIDATE = "potsawee/deberta-v3-large-mnli"
    nli_tok = AutoTokenizer.from_pretrained(NLI_CANDIDATE, local_files_only=True, use_fast=False)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_CANDIDATE, local_files_only=True
    ).to(device_str).eval()
    NLI_NAME = NLI_CANDIDATE
    is_potsawee = True
    print(f"⚠ Fallback to model: {NLI_NAME}")

# Determine class indices from config
id2label = {int(k): v.lower() for k, v in nli_model.config.id2label.items()}
num_classes = nli_model.config.num_labels
print(f"  NLI num_labels: {num_classes} | Config id2label: {id2label}")

ENT = next((k for k, v in id2label.items() if "entail" in v), 0)
CON = next((k for k, v in id2label.items() if "contra" in v), 2 if num_classes > 2 else 1)

# Auto-detect / verify label mapping by testing with known ground-truth premise-hypothesis pairs
test_premises = [
    "A dog is happily running across the green lawn in the park.",
    "A dog is happily running across the green lawn in the park."
]
test_hypotheses = [
    "An animal is active and outdoors.",                          # Entailment
    "There are no animals anywhere and the park is empty."        # Contradiction
]

test_tok_args = {
    "return_tensors": "pt",
    "padding": True,
    "truncation": True,
    "max_length": 128,
}
if is_potsawee:
    test_tok_args["return_token_type_ids"] = False

test_enc = nli_tok(test_premises, test_hypotheses, **test_tok_args)
if is_potsawee:
    test_enc.pop("token_type_ids", None)
    vocab_size = getattr(nli_model.config, "vocab_size", 128100)
    if "input_ids" in test_enc:
        test_enc["input_ids"] = torch.clamp(test_enc["input_ids"], min=0, max=vocab_size - 1)

test_enc = {k: v.to(device_str) for k, v in test_enc.items()}
with torch.no_grad():
    probe_logits = nli_model(**test_enc).logits

empirical_ent = int(torch.argmax(probe_logits[0]).item())
empirical_con = int(torch.argmax(probe_logits[1]).item())
print(f"  Empirical validation -> Entailment argmax index: {empirical_ent}, Contradiction argmax index: {empirical_con}")

if empirical_ent != ENT:
    print(f"  ⚠ Updating Entailment index from config {ENT} to empirical {empirical_ent}")
    ENT = empirical_ent
if empirical_con != CON:
    print(f"  ⚠ Updating Contradiction index from config {CON} to empirical {empirical_con}")
    CON = empirical_con

print(f"✓ Confirmed active NLI indices -> ENT: {ENT} ({id2label.get(ENT, 'entailment')}), CON: {CON} ({id2label.get(CON, 'contradiction')})")


# ============================================================
# 6. STEP 3: SCORING FUNCTIONS
# ============================================================
@torch.no_grad()
def compute_nli_probs(premises, hypotheses, bs=32):
    """
    Computes P(contradiction) and P(entailment) for premise-hypothesis pairs.
    Handles both 3-class (microsoft/deberta-large-mnli) and 2-class (potsawee) models.
    Premise: sample answer (or question + sample answer)
    Hypothesis: candidate answer (or question + candidate answer)
    """
    p_contra = []
    p_entail = []
    vocab_size = getattr(nli_model.config, "vocab_size", 128100)

    for i in range(0, len(premises), bs):
        b_prem = premises[i:i+bs]
        b_hyp  = hypotheses[i:i+bs]

        tok_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 256,
        }
        if is_potsawee:
            tok_kwargs["return_token_type_ids"] = False

        enc = nli_tok(b_prem, b_hyp, **tok_kwargs)

        if is_potsawee:
            enc.pop("token_type_ids", None)
            if "input_ids" in enc:
                enc["input_ids"] = torch.clamp(enc["input_ids"], min=0, max=vocab_size - 1)

        enc = {k: v.to(device_str) for k, v in enc.items()}
        logits = nli_model(**enc).logits
        num_c = logits.shape[-1]

        if num_c >= 3:
            # Standard 3-class NLI: contradiction, neutral, entailment
            probs = torch.softmax(logits, dim=-1)
            b_con = probs[:, CON].cpu().tolist()
            b_ent = probs[:, ENT].cpu().tolist()
        else:
            # 2-class NLI: softmax normalized over entailment and contradiction
            ent_idx = min(ENT, num_c - 1)
            con_idx = min(CON, num_c - 1)
            two = torch.stack([logits[:, ent_idx], logits[:, con_idx]], dim=-1)
            probs_two = torch.softmax(two, dim=-1)
            b_ent = probs_two[:, 0].cpu().tolist()
            b_con = probs_two[:, 1].cpu().tolist()

        p_contra.extend(b_con)
        p_entail.extend(b_ent)

    return p_contra, p_entail


def bertscore_f1(cands, refs, bs=64):
    """Computes BERTScore F1 between candidates and references."""
    _, _, F = bert_scorer.score(cands, refs, batch_size=bs)
    return F.tolist()


def score_candidate(question, cand, samples):
    """
    Computes 8 expanded hallucination detection features for candidate against resamples:
    1. bertscore_inconsistency: 1 - mean(F1) bare candidate vs resamples
    2. bertscore_inconsistency_ctx: 1 - mean(F1) question-prepended comparison
    3. bertscore_inconsistency_max: 1 - max(F1) bare candidate vs resamples
    4. nli_contradiction: mean P(contradiction) bare
    5. nli_contradiction_ctx: mean P(contradiction) with question context
    6. nli_entailment: 1 - mean P(entailment) bare (inverted entailment)
    7. nli_max_contradiction: max P(contradiction) across resamples
    8. semantic_entropy_proxy: std of contradiction probabilities (model inconsistency)
    """
    n_samples = len(samples)
    cand_list = [cand] * n_samples

    # 1. Bare BERTScore comparison
    f1_bare = np.array(bertscore_f1(cand_list, samples, bs=64), dtype=np.float64)
    bs_inconsistency = float(1.0 - np.mean(f1_bare))
    bs_inconsistency_max = float(1.0 - np.max(f1_bare))

    # 2. Context-prepended BERTScore comparison
    cand_ctx = [f"{question} {cand}"] * n_samples
    refs_ctx = [f"{question} {s}" for s in samples]
    f1_ctx = np.array(bertscore_f1(cand_ctx, refs_ctx, bs=64), dtype=np.float64)
    bs_inconsistency_ctx = float(1.0 - np.mean(f1_ctx))

    # 3. Bare NLI comparison: samples as premises, candidate as hypothesis
    contra_bare, entail_bare = compute_nli_probs(samples, cand_list, bs=32)
    contra_bare = np.array(contra_bare, dtype=np.float64)
    entail_bare = np.array(entail_bare, dtype=np.float64)

    nli_contra_mean = float(np.mean(contra_bare))
    nli_contra_max = float(np.max(contra_bare))
    nli_contra_std = float(np.std(contra_bare))
    # Inverted entailment: higher score = less entailed = higher hallucination likelihood
    nli_entail_inverted = float(1.0 - np.mean(entail_bare))

    # 4. Context-prepended NLI comparison
    contra_ctx, _ = compute_nli_probs(refs_ctx, cand_ctx, bs=32)
    contra_ctx = np.array(contra_ctx, dtype=np.float64)
    nli_contra_ctx_mean = float(np.mean(contra_ctx))

    return {
        "bertscore_inconsistency": bs_inconsistency,
        "bertscore_inconsistency_ctx": bs_inconsistency_ctx,
        "bertscore_inconsistency_max": bs_inconsistency_max,
        "nli_contradiction": nli_contra_mean,
        "nli_contradiction_ctx": nli_contra_ctx_mean,
        "nli_entailment": nli_entail_inverted,
        "nli_max_contradiction": nli_contra_max,
        "semantic_entropy_proxy": nli_contra_std,
    }


# ============================================================
# 7. STEP 4: SCORE ALL CANDIDATES AGAINST RESAMPLES
# ============================================================
SC_CKPT = os.path.join(CKPT_DIR, "task2_v2_scores.json")
task2_scores = json.load(open(SC_CKPT)) if os.path.exists(SC_CKPT) else {}

print(f"\n[Step 4 / Section 7] SelfCheck V2 scoring ({len(task2_scores)} already completed)...")
t0 = time.time()

for i, item in enumerate(tqdm(sampled_data, desc="SelfCheck scoring (V2)")):
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
# 8. STEP 5: AUC-PR & PAIRWISE ACCURACY EVALUATION
# ============================================================
def collect(ids, key):
    y, s = [], []
    for sid in ids:
        r = task2_scores[str(sid)]
        for t, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            y.append(lab)
            s.append(r[t][key])
    return np.array(y, dtype=np.int32), np.array(s, dtype=np.float64)


def pairwise_acc(ids, key):
    return float(np.mean([
        task2_scores[str(i)]["hallucinated_answer"][key] >
        task2_scores[str(i)]["right_answer"][key]
        for i in ids
    ]))


KEYS = {
    "SelfCheck-BERTScore (bare, primary)":          "bertscore_inconsistency",
    "SelfCheck-BERTScore + question context":       "bertscore_inconsistency_ctx",
    "BERTScore (max-over-samples, ablation)":       "bertscore_inconsistency_max",
    "SelfCheck-NLI (mean contradiction, primary)":  "nli_contradiction",
    "SelfCheck-NLI + question context":             "nli_contradiction_ctx",
    "SelfCheck-NLI (inverted entailment)":          "nli_entailment",
    "SelfCheck-NLI (max contradiction across N)":   "nli_max_contradiction",
    "SelfCheck-NLI (semantic entropy proxy, std)":  "semantic_entropy_proxy",
}

results = {}
print("\n" + "=" * 78)
print("   TASK 2 V2 RESULTS — AUC-PR & PAIRWISE ACCURACY (EXPANDED FEATURES)")
print("=" * 78)

splits_to_eval = [
    ("test split (160 instances)", test_ids),
    ("train split (640 instances)", train_ids),
    ("full dataset (800 instances)", all_ids),
]

for split_name, ids in splits_to_eval:
    print(f"\n=== {split_name} ===")
    results[split_name] = {}
    rows = []
    for name, key in KEYS.items():
        y, s = collect(ids, key)
        ap = average_precision_score(y, s)
        pw = pairwise_acc(ids, key)
        results[split_name][name] = {"feature_key": key, "AUC-PR": round(float(ap), 4), "pairwise_acc": round(pw, 4)}
        rows.append({"Feature / Method": name, "Feature Key": key, "AUC-PR": f"{ap:.4f}", "Pairwise Acc": f"{pw:.4f}"})
        print(f"  {name:46s} AUC-PR = {ap:.4f}   pairwise acc = {pw:.3f}")

    # Print summary table
    df_res = pd.DataFrame(rows)
    print(f"\nSummary Table [{split_name}]:")
    print(df_res[["Feature / Method", "AUC-PR", "Pairwise Acc"]].to_string(index=False))


# ============================================================
# 9. STEP 6: DIAGNOSTIC PLOTS
# ============================================================
print("\n[Step 6 / Section 9] Generating diagnostic plots...")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))

# Row 0: Score distributions for primary features
dist_features = [
    ("SelfCheck-BERTScore (bare)", "bertscore_inconsistency"),
    ("SelfCheck-BERTScore (context)", "bertscore_inconsistency_ctx"),
    ("SelfCheck-NLI (contradiction bare)", "nli_contradiction"),
    ("SelfCheck-NLI (contradiction ctx)", "nli_contradiction_ctx"),
]

for col, (title, key) in enumerate(dist_features):
    ax = axes[0, col]
    y, s = collect(all_ids, key)
    ax.hist(s[y == 0], bins=25, alpha=0.6, color="forestgreen", density=True, label="Right (0)")
    ax.hist(s[y == 1], bins=25, alpha=0.6, color="crimson", density=True, label="Hallucinated (1)")
    ax.set_title(f"{title}\nDistribution (All 800)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Hallucination Score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

# Row 1, Col 0: PR Curves for BERTScore Variants on Test Split
ax_bs = axes[1, 0]
y_test_bs, _ = collect(test_ids, "bertscore_inconsistency")
ax_bs.axhline(y_test_bs.mean(), ls="--", color="gray", label=f"Random ({y_test_bs.mean():.2f})")
bs_keys = [
    ("BERTScore (bare)", "bertscore_inconsistency", "blue"),
    ("BERTScore (ctx)", "bertscore_inconsistency_ctx", "darkorange"),
    ("BERTScore (max)", "bertscore_inconsistency_max", "purple"),
]
for label, key, color in bs_keys:
    y_eval, s_eval = collect(test_ids, key)
    p, r, _ = precision_recall_curve(y_eval, s_eval)
    ap = average_precision_score(y_eval, s_eval)
    ax_bs.plot(r, p, lw=2, color=color, label=f"{label} ({ap:.3f})")
ax_bs.set_title("BERTScore Variants — Test PR Curves", fontsize=11, fontweight="bold")
ax_bs.set_xlabel("Recall")
ax_bs.set_ylabel("Precision")
ax_bs.legend(loc="lower left", fontsize=8)
ax_bs.grid(alpha=0.3)

# Row 1, Col 1: PR Curves for NLI Variants on Test Split
ax_nli = axes[1, 1]
y_test_nli, _ = collect(test_ids, "nli_contradiction")
ax_nli.axhline(y_test_nli.mean(), ls="--", color="gray", label=f"Random ({y_test_nli.mean():.2f})")
nli_keys = [
    ("NLI Contra (bare)", "nli_contradiction", "red"),
    ("NLI Contra (ctx)", "nli_contradiction_ctx", "magenta"),
    ("NLI Inverted Entail", "nli_entailment", "navy"),
    ("NLI Max Contra", "nli_max_contradiction", "brown"),
    ("Semantic Entropy (std)", "semantic_entropy_proxy", "teal"),
]
for label, key, color in nli_keys:
    y_eval, s_eval = collect(test_ids, key)
    p, r, _ = precision_recall_curve(y_eval, s_eval)
    ap = average_precision_score(y_eval, s_eval)
    ax_nli.plot(r, p, lw=1.8, color=color, label=f"{label} ({ap:.3f})")
ax_nli.set_title("NLI Variants — Test PR Curves", fontsize=11, fontweight="bold")
ax_nli.set_xlabel("Recall")
ax_nli.set_ylabel("Precision")
ax_nli.legend(loc="lower left", fontsize=7.5)
ax_nli.grid(alpha=0.3)

# Row 1, Col 2: Distribution for Inverted Entailment
ax_ent = axes[1, 2]
y, s = collect(all_ids, "nli_entailment")
ax_ent.hist(s[y == 0], bins=25, alpha=0.6, color="forestgreen", density=True, label="Right (0)")
ax_ent.hist(s[y == 1], bins=25, alpha=0.6, color="crimson", density=True, label="Hallucinated (1)")
ax_ent.set_title("NLI Inverted Entailment\n(1 - P(entailment))", fontsize=11, fontweight="bold")
ax_ent.set_xlabel("Score")
ax_ent.set_ylabel("Density")
ax_ent.legend(loc="upper left", fontsize=9)
ax_ent.grid(alpha=0.3)

# Row 1, Col 3: Distribution for Semantic Entropy Proxy (std)
ax_std = axes[1, 3]
y, s = collect(all_ids, "semantic_entropy_proxy")
ax_std.hist(s[y == 0], bins=25, alpha=0.6, color="forestgreen", density=True, label="Right (0)")
ax_std.hist(s[y == 1], bins=25, alpha=0.6, color="crimson", density=True, label="Hallucinated (1)")
ax_std.set_title("Semantic Entropy Proxy\n(Std of Contradiction Probs)", fontsize=11, fontweight="bold")
ax_std.set_xlabel("Score (Std Dev)")
ax_std.set_ylabel("Density")
ax_std.legend(loc="upper right", fontsize=9)
ax_std.grid(alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "task2_v2_plots.png")
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"✓ Diagnostic plots saved to: {plot_path}")


# ============================================================
# 10. STEP 7: SAVE SUMMARY RESULTS
# ============================================================
final = {
    "version": "v2_optimized",
    "llm": MODEL_NAME,
    "bertscore_backbone": "roberta-large",
    "nli_model": NLI_NAME,
    "is_potsawee_fallback": is_potsawee,
    "nli_label_indices": {"ENT": ENT, "CON": CON},
    "resampling_protocol": {
        "N": N_SAMPLES,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": f"{GEN_SEED}+sample_id",
        "prompt": "chat template, short direct answer, closed-book (no knowledge passage)",
        "output_checkpoint": RES_CKPT,
    },
    "feature_descriptions": {
        "bertscore_inconsistency": "1 - mean(F1) bare candidate vs resamples",
        "bertscore_inconsistency_ctx": "1 - mean(F1) question-prepended comparison",
        "bertscore_inconsistency_max": "1 - max(F1) bare candidate vs resamples",
        "nli_contradiction": "mean P(contradiction) bare candidate vs resamples",
        "nli_contradiction_ctx": "mean P(contradiction) with question prepended",
        "nli_entailment": "1 - mean P(entailment) bare candidate vs resamples (inverted)",
        "nli_max_contradiction": "max P(contradiction) across resamples",
        "semantic_entropy_proxy": "std of contradiction probabilities across resamples",
    },
    "random_seed_split": split_info["random_seed"],
    "metrics": results,
}

res_path = os.path.join(OUT_DIR, "task2_v2_results.json")
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2)

print(f"✓ Final results saved to: {res_path}")

total_elapsed = time.time() - TOTAL_START_TIME
hours = int(total_elapsed // 3600)
minutes = int((total_elapsed % 3600) // 60)
seconds = int(total_elapsed % 60)

print("\n" + "=" * 70)
print(f"⏱ TOTAL EXECUTION TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.2f} seconds)")
print("=" * 70)
print("✓ Task 2 V2 complete! Optimized features ready for Task 3 V2.")
