#!/usr/bin/env python3
"""
Pre-download datasets and Hugging Face models on the master node
so that compute nodes (like gnode5) can run 100% offline.
"""

import json
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================================
# 1. PROCESS RAW DATA & CREATE 400-SAMPLE SPLIT
# ============================================================
raw_path     = os.path.join(DATA_DIR, "halueval_qa_raw.json")
sampled_path = os.path.join(DATA_DIR, "halueval_qa_sampled.json")
split_path   = os.path.join(DATA_DIR, "train_test_split.json")

RANDOM_SEED = 42
SAMPLE_SIZE = 400

if not os.path.exists(raw_path):
    print(f"❌ Error: {raw_path} not found. Please run wget first:")
    print('   wget -O data/halueval_qa_raw.json "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"')
    sys.exit(1)

print("\n--- Step 1: Processing raw dataset ---")
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

print(f"✓ Parsed {len(raw_data)} total QA pairs from {raw_path}")

# Draw sample
random.seed(RANDOM_SEED)
sampled_data = random.sample(raw_data, SAMPLE_SIZE)
for i, item in enumerate(sampled_data):
    item["sample_id"] = i

with open(sampled_path, "w", encoding="utf-8") as f:
    json.dump(sampled_data, f, indent=2, ensure_ascii=False)
print(f"✓ Created sampled dataset (400 pairs): {sampled_path}")

# 80/20 train/test split
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
print(f"✓ Created 80/20 train/test split (320 train / 80 test): {split_path}")


# ============================================================
# 2. HUGGING FACE AUTHENTICATION & MODEL PRE-CACHING
# ============================================================
print("\n--- Step 2: Pre-caching Hugging Face Models ---")
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

token_file = os.path.join(BASE_DIR, "token.txt")
hf_token = None
if os.path.exists(token_file):
    with open(token_file, "r") as f:
        hf_token = f.read().strip()
if not hf_token:
    hf_token = "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"

login(token=hf_token)
print("✓ Hugging Face authenticated.")

# 2.1 Gemma-2-2B (Task 1 & Task 2)
MODEL_NAME = "google/gemma-2-2b-it"
print(f"\n[1/3] Downloading {MODEL_NAME}...")
AutoTokenizer.from_pretrained(MODEL_NAME)
AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print(f"✓ {MODEL_NAME} cached.")

# 2.2 RoBERTa-large (Task 2 BERTScore)
ROBERTA_NAME = "roberta-large"
print(f"\n[2/3] Downloading {ROBERTA_NAME} (BERTScore backbone)...")
AutoTokenizer.from_pretrained(ROBERTA_NAME)
AutoModelForSequenceClassification.from_pretrained(ROBERTA_NAME)
print(f"✓ {ROBERTA_NAME} cached.")

# 2.3 DeBERTa-v3-large-MNLI (Task 2 NLI)
NLI_NAME = "potsawee/deberta-v3-large-mnli"
print(f"\n[3/3] Downloading {NLI_NAME} (SelfCheck-NLI)...")
AutoTokenizer.from_pretrained(NLI_NAME)
AutoModelForSequenceClassification.from_pretrained(NLI_NAME)
print(f"✓ {NLI_NAME} cached.")

print("\n" + "=" * 65)
print("🎉 All datasets and models are fully pre-cached in ~/.cache/huggingface!")
print("Compute node gnode5 can now run completely offline with no network errors.")
print("=" * 65)

