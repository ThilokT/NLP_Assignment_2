#!/usr/bin/env python3
"""
Prepare V2 Models — Download the 3-class NLI model for Task 2 optimization.
Run this script on the MASTER (login) node ONLY, since compute nodes have no internet.

Usage:
    python prepare_v2_models.py
"""

import os
import sys

print("=" * 60)
print("  Preparing V2 Models for Optimized Tasks")
print("=" * 60)

# Check we're on a node with internet
import socket
try:
    socket.create_connection(("huggingface.co", 443), timeout=5)
    print("✓ Internet access confirmed.")
except OSError:
    print("❌ ERROR: No internet access detected!")
    print("   This script must be run on the MASTER (login) node.")
    print("   Please exit the compute node first: exit")
    sys.exit(1)

# HF Auth
from huggingface_hub import login

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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
    print(f"HF login note: {e}")

# Download microsoft/deberta-large-mnli (3-class NLI model)
print("\n[1/2] Downloading microsoft/deberta-large-mnli (3-class NLI)...")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI_MODEL = "microsoft/deberta-large-mnli"
try:
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    
    # Verify labels
    id2label = model.config.id2label
    print(f"  ✓ Model loaded: {NLI_MODEL}")
    print(f"  ✓ num_labels: {model.config.num_labels}")
    print(f"  ✓ id2label: {id2label}")
    
    # Quick sanity check
    import torch
    enc = tokenizer(
        "The cat is sleeping on the sofa.",
        "The cat is awake and running.",
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1)[0]
    labels = [id2label[i] for i in range(len(probs))]
    print(f"  ✓ Sanity check (sleeping vs awake):")
    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"    [{i}] {label}: {prob.item():.4f}")
    
    del model, tokenizer
    print(f"\n  ✓ {NLI_MODEL} cached successfully!")
    
except Exception as e:
    print(f"\n  ⚠ Failed to download {NLI_MODEL}: {e}")
    print(f"  The optimized Task 2 will fall back to potsawee/deberta-v3-large-mnli")

# Verify existing models are still cached
print("\n[2/2] Verifying existing model caches...")
EXISTING_MODELS = [
    "google/gemma-2-2b-it",
    "roberta-large",
    "potsawee/deberta-v3-large-mnli",
]

for name in EXISTING_MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(name)
        print(f"  ✓ {name} — cached and ready")
        del tok
    except Exception as e:
        print(f"  ⚠ {name} — {e}")

print("\n" + "=" * 60)
print("  ✓ V2 model preparation complete!")
print("  You can now run task1_v2.py, task2_v2.py, task3_v2.py on gnode5.")
print("=" * 60)

