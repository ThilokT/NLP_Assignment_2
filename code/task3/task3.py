#!/usr/bin/env python3
"""
Task 3: Stacked Ensemble Detector (Logistic-Regression Meta-Classifier)
NLP Assignment 2 - Project 7 (HaluEval QA)

Features:
- Task 1: mean_log_prob, mean_entropy
- Task 2: bertscore_inconsistency, nli_contradiction

Evaluates AUC-PR on the held-out 20% test partition (160 instances).
"""

import json
import os
import sys
import time

# Ensure immediate unbuffered output to log files
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TOTAL_START_TIME = time.time()

import joblib
import matplotlib
# Headless backend for Linux cluster without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DIRECTORIES & PREREQUISITE CHECK
# ============================================================
# Robust project root resolution (works from code/taskX/ or repo root)
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

sampled_path = os.path.join(DATA_DIR, "halueval_qa_sampled.json")
split_path   = os.path.join(DATA_DIR, "train_test_split.json")
t1_path      = os.path.join(CKPT_DIR, "task1_logprobs_entropy.json")
t2_path      = os.path.join(CKPT_DIR, "task2_selfcheck_scores.json")

missing = []
if not os.path.exists(sampled_path): missing.append(sampled_path)
if not os.path.exists(split_path):   missing.append(split_path)
if not os.path.exists(t1_path):      missing.append(f"{t1_path} (Run task1.py first)")
if not os.path.exists(t2_path):      missing.append(f"{t2_path} (Run task2.py first)")

if missing:
    print("\n❌ Cannot run Task 3: Missing prerequisite files:")
    for m in missing:
        print(f"   - {m}")
    print("\nPlease run task1.py and task2.py first to compute the required features.")
    sys.exit(1)

# ============================================================
# 2. LOAD SPLITS & FEATURE FILES
# ============================================================
sampled_data = json.load(open(sampled_path))
split_info   = json.load(open(split_path))
train_ids, test_ids = split_info["train_sample_ids"], split_info["test_sample_ids"]
SEED = split_info["random_seed"]

t1 = json.load(open(t1_path))
t2 = json.load(open(t2_path))

print("=" * 65)
print("   TASK 3: STACKED ENSEMBLE DETECTOR")
print("=" * 65)
print(f"Task 1 features: {len(t1)} questions")
print(f"Task 2 features: {len(t2)} questions")
print(f"Train split:     {len(train_ids)} questions ({len(train_ids)*2} instances)")
print(f"Test split:      {len(test_ids)} questions ({len(test_ids)*2} instances)")


# ============================================================
# 3. BUILD FEATURE TABLE
# ============================================================
T1_FEATS = ["mean_log_prob", "mean_entropy"]
T2_FEATS = ["bertscore_inconsistency", "nli_contradiction"]
EXTRA    = ["bertscore_inconsistency_max", "nli_contradiction_ctx"]

def build_dataframe(ids):
    rows = []
    for sid in sorted(ids):
        for t, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            r = {"sample_id": sid, "label": lab}
            r.update({k: t1[str(sid)][t][k] for k in T1_FEATS})
            r.update({k: t2[str(sid)][t][k] for k in T2_FEATS + EXTRA})
            rows.append(r)
    return pd.DataFrame(rows)

train_df = build_dataframe(train_ids)
test_df  = build_dataframe(test_ids)

assert not (set(train_df.sample_id) & set(test_df.sample_id)), "Data leakage: question appears in both train and test!"
print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")


# ============================================================
# 4. SINGLE-SIGNAL REFERENCE SCORES ON THE TEST SPLIT
# ============================================================
y_test  = test_df.label.values
y_train = train_df.label.values

raw_scores = {
    "T1: -mean log-prob":      -test_df.mean_log_prob.values,
    "T1: mean entropy":         test_df.mean_entropy.values,
    "T2: SelfCheck-BERTScore":  test_df.bertscore_inconsistency.values,
    "T2: SelfCheck-NLI":        test_df.nli_contradiction.values,
}

ap_raw = {k: average_precision_score(y_test, v) for k, v in raw_scores.items()}
print("\n--- Single-Signal Baselines (Test AUC-PR) ---")
for k, v in ap_raw.items():
    print(f"  {k:28s} AUC-PR = {v:.4f}")
print(f"  {'Random Baseline':28s} AUC-PR = {y_test.mean():.4f}")


# ============================================================
# 5. FIT LOGISTIC-REGRESSION META-CLASSIFIERS
# ============================================================
C_GRID = [0.01, 0.1, 1, 10, 100]
groups = train_df.sample_id.values

def fit_meta(feats):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
    ])
    gs = GridSearchCV(
        pipe,
        {"lr__C": C_GRID},
        scoring="average_precision",
        cv=GroupKFold(n_splits=5),
        n_jobs=-1,
        refit=True
    )
    gs.fit(train_df[feats].values, y_train, groups=groups)
    return gs

FEATURE_SETS = {
    "T1 features only (LR)":   T1_FEATS,
    "T2 features only (LR)":   T2_FEATS,
    "T3 ensemble: T1+T2 (LR)": T1_FEATS + T2_FEATS,
}

models, ap_lr, test_scores = {}, {}, {}
print("\n--- Meta-Classifier Cross-Validation & Test Performance ---")
for name, feats in FEATURE_SETS.items():
    gs = fit_meta(feats)
    models[name] = gs
    test_scores[name] = gs.predict_proba(test_df[feats].values)[:, 1]
    ap_lr[name] = average_precision_score(y_test, test_scores[name])
    print(f"  {name:26s} best C={gs.best_params_['lr__C']:<6} CV AUC-PR(train)={gs.best_score_:.4f} | TEST AUC-PR={ap_lr[name]:.4f}")


# ============================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS (B = 2000)
# ============================================================
rng = np.random.default_rng(SEED)
n_q = len(test_df) // 2
B = 2000
boot_idx = rng.integers(0, n_q, size=(B, n_q))

def boot_ap(score):
    s = score.reshape(n_q, 2)
    y = y_test.reshape(n_q, 2)
    return np.array([average_precision_score(y[ix].ravel(), s[ix].ravel()) for ix in boot_idx])

best_t1 = max(["T1: -mean log-prob", "T1: mean entropy"], key=lambda k: ap_raw[k])
best_t2 = max(["T2: SelfCheck-BERTScore", "T2: SelfCheck-NLI"], key=lambda k: ap_raw[k])

table_scores = {
    f"Task 1 alone ({best_t1[4:]})": raw_scores[best_t1],
    f"Task 2 alone ({best_t2[4:]})": raw_scores[best_t2],
    "Task 1 features, LR":           test_scores["T1 features only (LR)"],
    "Task 2 features, LR":           test_scores["T2 features only (LR)"],
    "Task 3 ensemble (T1+T2, LR)":   test_scores["T3 ensemble: T1+T2 (LR)"],
}

rows, boots = [], {}
for name, s in table_scores.items():
    b = boot_ap(s)
    boots[name] = b
    rows.append({
        "Method": name,
        "AUC-PR (test)": round(average_precision_score(y_test, s), 4),
        "95% CI low": round(np.percentile(b, 2.5), 4),
        "95% CI high": round(np.percentile(b, 97.5), 4),
    })

comparison = pd.DataFrame(rows)
print("\n" + "=" * 65)
print("   FINAL COMPARISON TABLE (TEST SET AUC-PR)")
print("=" * 65)
print(comparison.to_string(index=False))

ens = boots["Task 3 ensemble (T1+T2, LR)"]
for ref in [k for k in boots if k.startswith("Task 1 alone") or k.startswith("Task 2 alone")]:
    d = ens - boots[ref]
    print(f"\nEnsemble vs [{ref}]: mean diff = {d.mean():+.4f}, 95% CI [{np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f}], P(ensemble better) = {(d>0).mean():.2f}")


# ============================================================
# 7. FEATURE IMPORTANCE & WEIGHT ANALYSIS
# ============================================================
gs = models["T3 ensemble: T1+T2 (LR)"]
lr = gs.best_estimator_.named_steps["lr"]
coef = pd.Series(lr.coef_[0], index=T1_FEATS + T2_FEATS).sort_values(key=abs, ascending=False)
print("\n--- Standardized Logistic Regression Coefficients ---")
print(coef.round(3).to_string())
print("\nExpected signs: mean_log_prob NEGATIVE; entropy and SelfCheck POSITIVE.")

print("\n--- Feature Correlation Matrix on Train ---")
print(train_df[T1_FEATS + T2_FEATS].corr().round(2).to_string())


# ============================================================
# 8. PRECISION-RECALL CURVE PLOT
# ============================================================
plt.figure(figsize=(7, 5.5))
for name, s in table_scores.items():
    p, r, _ = precision_recall_curve(y_test, s)
    plt.plot(r, p, lw=2.2 if "Task 3" in name else 1.3, label=f"{name} ({average_precision_score(y_test, s):.3f})")

plt.axhline(y_test.mean(), ls="--", c="gray", label="Random baseline")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Task 3 — Stacked Ensemble PR Curves (160 test instances)")
plt.legend(fontsize=8)
plt.tight_layout()
pr_plot_path = os.path.join(OUT_DIR, "task3_pr_curves.png")
plt.savefig(pr_plot_path, dpi=150)
plt.close()
print(f"\n✓ PR curves plot saved to: {pr_plot_path}")


# ============================================================
# 9. ABLATION STUDY
# ============================================================
ALL4 = T1_FEATS + T2_FEATS
abl = {}
for f in ALL4:
    abl[f"All minus {f}"] = [x for x in ALL4 if x != f]
abl["Extended (+ BERTScore-max, NLI+ctx)"] = ALL4 + EXTRA

abl_rows = []
for name, feats in abl.items():
    g = fit_meta(feats)
    s = g.predict_proba(test_df[feats].values)[:, 1]
    abl_rows.append({
        "Feature set": name,
        "CV AUC-PR (train)": round(g.best_score_, 4),
        "Test AUC-PR": round(average_precision_score(y_test, s), 4),
    })

ablation = pd.DataFrame(abl_rows)
print("\n--- Ablation Study ---")
print(ablation.to_string(index=False))


# ============================================================
# 10. SAVE DELIVERABLES
# ============================================================
model_out = os.path.join(OUT_DIR, "task3_meta_classifier.joblib")
joblib.dump(models["T3 ensemble: T1+T2 (LR)"].best_estimator_, model_out)

csv_out = os.path.join(OUT_DIR, "final_comparison.csv")
comparison.to_csv(csv_out, index=False)

final = {
    "split_seed": SEED,
    "train_questions": len(train_ids),
    "test_questions": len(test_ids),
    "train_instances": len(train_df),
    "test_instances": len(test_df),
    "features": T1_FEATS + T2_FEATS,
    "meta_classifier": "StandardScaler + LogisticRegression(L2), C by GroupKFold(5) CV on train, scoring=AUC-PR",
    "best_C": gs.best_params_["lr__C"],
    "test_auc_pr": {r["Method"]: r["AUC-PR (test)"] for r in comparison.to_dict("records")},
    "coefficients_standardised": coef.round(4).to_dict(),
    "ablation": ablation.to_dict("records"),
}

json_out = os.path.join(OUT_DIR, "task3_results.json")
with open(json_out, "w") as f:
    json.dump(final, f, indent=2)

print(f"\n✓ Saved model:           {model_out}")
print(f"✓ Saved comparison CSV:  {csv_out}")
print(f"✓ Saved results JSON:    {json_out}")

total_elapsed = time.time() - TOTAL_START_TIME
hours = int(total_elapsed // 3600)
minutes = int((total_elapsed % 3600) // 60)
seconds = int(total_elapsed % 60)

print("\n" + "=" * 60)
print(f"⏱ TOTAL EXECUTION TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.2f} seconds)")
print("=" * 60)
print("\n✓ Task 3 completed successfully!")

