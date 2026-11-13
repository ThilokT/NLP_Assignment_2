#!/usr/bin/env python3
"""
Task 3.2: Stacked Ensemble Meta-Classifier Suite WITHOUT Length-Confounded Features
Excludes: 'answer_token_count' and 'sum_log_prob' (both kc_answer_token_count and kc_sum_log_prob)
NLP Assignment 2 - Project 7 (HaluEval QA)

Features Evaluated (15 features total):
- Task 1 (7 features): kc_mean_log_prob, kc_mean_entropy, kc_log_prob_variance,
                       kc_entropy_variance, kc_min_log_prob, kc_max_entropy, kc_perplexity
- Task 2 (8 features): v2_bertscore_inconsistency, v2_bertscore_inconsistency_ctx,
                       v2_bertscore_inconsistency_max, v2_nli_contradiction,
                       v2_nli_contradiction_ctx, v2_nli_entailment,
                       v2_nli_max_contradiction, v2_semantic_entropy_proxy

Models Evaluated:
- Logistic Regression (T1 only, T2 only, All 15 features)
- Random Forest Classifier (All 15 features)
- Gradient Boosting Classifier (All 15 features)

Cross-Validation: GroupKFold(5) grouped by sample_id on train split (640 instances).
Evaluation: AUC-PR with cluster bootstrap 95% CIs (B=2000) on held-out test split (160 instances).
"""

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TOTAL_START_TIME = time.time()

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ============================================================
# 1. DIRECTORIES & PREREQUISITE CHECK
# ============================================================
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

split_path   = os.path.join(DATA_DIR, "train_test_split.json")
sampled_path = os.path.join(DATA_DIR, "halueval_qa_sampled.json")

t1_primary   = os.path.join(CKPT_DIR, "task1_features.json")
t1_legacy    = os.path.join(CKPT_DIR, "task1_v2_features.json")
t2_primary   = os.path.join(CKPT_DIR, "task2_scores.json")
t2_legacy    = os.path.join(CKPT_DIR, "task2_v2_scores.json")

t1_path = t1_primary if os.path.exists(t1_primary) else t1_legacy
t2_path = t2_primary if os.path.exists(t2_primary) else t2_legacy

print("=" * 75)
print("   TASK 3.2: ENSEMBLE SUITE WITHOUT LENGTH-CONFOUNDED FEATURES")
print("   (EXCLUDING 'answer_token_count' & 'sum_log_prob')")
print("=" * 75)

if not os.path.exists(split_path):
    print(f"\n❌ Cannot run Task 3.2: Split file missing at {split_path}")
    sys.exit(1)

ckpt_status = {
    "Task 1 Features": (t1_path, os.path.exists(t1_path)),
    "Task 2 Scores":   (t2_path, os.path.exists(t2_path)),
}

available_count = sum(1 for _, (_, exists) in ckpt_status.items() if exists)
if available_count == 0:
    print("\n❌ Cannot run Task 3.2: No feature checkpoint files found in checkpoints/.")
    print("Run Task 1 and/or Task 2 first to generate checkpoints.")
    sys.exit(1)


# ============================================================
# 2. LOAD SPLITS & FEATURE FILES
# ============================================================
split_info = json.load(open(split_path, "r", encoding="utf-8"))
train_ids = split_info["train_sample_ids"]
test_ids  = split_info["test_sample_ids"]
SEED      = split_info.get("random_seed", 42)

print(f"Random seed:     {SEED}")
print(f"Train split:     {len(train_ids)} questions ({len(train_ids)*2} instances)")
print(f"Test split:      {len(test_ids)} questions ({len(test_ids)*2} instances)")
print("\n--- Checkpoint Files Status ---")

def load_json_graceful(path, name):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  ✓ Found & loaded: {name:<25s} ({len(data)} items from {os.path.basename(path)})")
            return data
        except Exception as e:
            print(f"  ⚠️ Error parsing {path}: {e}")
            return None
    else:
        print(f"  ⚠️ Notice: {name:<25s} NOT found ({os.path.basename(path)}). Skipping.")
        return None

t1_data = load_json_graceful(t1_path, "Task 1 Features")
t2_data = load_json_graceful(t2_path, "Task 2 Scores")


# ============================================================
# 3. BUILD UNIFIED FEATURE TABLE (EXCLUDING LENGTH FEATURES)
# ============================================================
# Notice sum_log_prob and answer_token_count are completely removed:
T1_FEATS = [
    "kc_mean_log_prob", "kc_mean_entropy",
    "kc_log_prob_variance", "kc_entropy_variance", "kc_min_log_prob",
    "kc_max_entropy", "kc_perplexity"
]
T2_FEATS = [
    "bertscore_inconsistency", "bertscore_inconsistency_ctx", "bertscore_inconsistency_max",
    "nli_contradiction", "nli_contradiction_ctx", "nli_entailment", "nli_max_contradiction",
    "semantic_entropy_proxy"
]

EXCLUDED_KEYS = {
    "sum_log_prob", "kc_sum_log_prob",
    "answer_token_count", "kc_answer_token_count"
}

def build_dataframe(ids, desc="Split"):
    rows = []
    for sid in tqdm(sorted(ids), desc=f"Building {desc} feature table", leave=False):
        sid_str = str(sid)
        for answer_type, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            r = {"sample_id": sid, "label": lab}

            if t1_data is not None and sid_str in t1_data:
                ans = t1_data[sid_str].get(answer_type, {})
                for k in T1_FEATS:
                    short_k = k[3:] if k.startswith("kc_") else k
                    if k in ans and ans[k] is not None:
                        r[k] = float(ans[k])
                    elif short_k in ans and ans[short_k] is not None:
                        r[f"kc_{short_k}"] = float(ans[short_k])
                for k, v in ans.items():
                    if k in EXCLUDED_KEYS or f"kc_{k}" in EXCLUDED_KEYS:
                        continue
                    if k not in ("sample_id", "original_id", "label") and isinstance(v, (int, float)):
                        col = k if k.startswith("kc_") else f"kc_{k}"
                        if col not in r and col not in EXCLUDED_KEYS:
                            r[col] = float(v)

            if t2_data is not None and sid_str in t2_data:
                ans = t2_data[sid_str].get(answer_type, {})
                for k in T2_FEATS:
                    col = k if k.startswith("v2_") else f"v2_{k}"
                    short_k = k[3:] if k.startswith("v2_") else k
                    if k in ans and ans[k] is not None:
                        r[col] = float(ans[k])
                    elif short_k in ans and ans[short_k] is not None:
                        r[col] = float(ans[short_k])
                for k, v in ans.items():
                    if k in EXCLUDED_KEYS or f"v2_{k}" in EXCLUDED_KEYS:
                        continue
                    if k not in ("sample_id", "original_id", "label") and isinstance(v, (int, float)):
                        col = k if k.startswith("v2_") else f"v2_{k}"
                        if col not in r and col not in EXCLUDED_KEYS:
                            r[col] = float(v)

            rows.append(r)
    return pd.DataFrame(rows)

train_df = build_dataframe(train_ids, desc="Train")
test_df  = build_dataframe(test_ids, desc="Test")

# Double check that excluded features are definitely not present
dropped_cols = [c for c in train_df.columns if any(ex in c for ex in ["sum_log_prob", "answer_token_count"])]
if dropped_cols:
    print(f"  Dropping excluded length features: {dropped_cols}")
    train_df = train_df.drop(columns=dropped_cols, errors="ignore")
    test_df = test_df.drop(columns=dropped_cols, errors="ignore")

overlap = set(train_df["sample_id"]) & set(test_df["sample_id"])
assert not overlap, f"Data leakage: Questions {overlap} appear in both train and test!"

meta_cols = {"sample_id", "label"}
all_feature_cols = [c for c in train_df.columns if c not in meta_cols]

if not all_feature_cols:
    print("\n❌ Error: No features extracted into feature table.")
    sys.exit(1)

t1_feature_cols = [c for c in all_feature_cols if c.startswith("kc_")]
t2_feature_cols = [c for c in all_feature_cols if c.startswith("v2_")]

print(f"\nUnified Feature Table (15 Features):")
print(f"  Train shape:                {train_df.shape}")
print(f"  Test shape:                 {test_df.shape}")
print(f"  Task 1 Features ({len(t1_feature_cols)}):        {t1_feature_cols}")
print(f"  Task 2 Features ({len(t2_feature_cols)}):        {t2_feature_cols}")
print(f"  Total Features ({len(all_feature_cols)}):       {all_feature_cols}")

# Impute missing values with training median
for col in all_feature_cols:
    train_nulls = train_df[col].isna().sum()
    test_nulls = test_df[col].isna().sum()
    if train_nulls > 0 or test_nulls > 0:
        fill_val = train_df[col].median()
        if pd.isna(fill_val):
            fill_val = 0.0
        train_df[col] = train_df[col].fillna(fill_val)
        test_df[col]  = test_df[col].fillna(fill_val)


# ============================================================
# 4. SINGLE-SIGNAL BASELINES ON TEST SET
# ============================================================
y_train = train_df["label"].values.astype(int)
y_test  = test_df["label"].values.astype(int)

single_feat_scores = {}
single_feat_ap = {}
single_feat_train_ap = {}
single_feat_dir = {}

print("\n" + "=" * 75)
print("   SECTION 4: SINGLE-SIGNAL BASELINES (NO LENGTH FEATURES)")
print("=" * 75)

for feat in all_feature_cols:
    tr_vals = train_df[feat].values
    std_tr = np.std(tr_vals)
    corr = np.corrcoef(tr_vals, y_train)[0, 1] if std_tr > 1e-12 else 0.0
    direction = -1.0 if corr < 0 else 1.0
    score = direction * test_df[feat].values
    ap = average_precision_score(y_test, score)
    train_ap = average_precision_score(y_train, direction * tr_vals)

    single_feat_scores[feat] = score
    single_feat_ap[feat] = ap
    single_feat_train_ap[feat] = train_ap
    single_feat_dir[feat] = "-" if direction < 0 else "+"

best_single_feat = max(single_feat_train_ap, key=single_feat_train_ap.get)
best_single_ap = single_feat_ap[best_single_feat]
sorted_single = sorted(single_feat_train_ap.items(), key=lambda x: x[1], reverse=True)

for feat, tr_ap in sorted_single:
    print(f"  {feat:<38} {single_feat_dir[feat]:<3} Train AUC-PR = {tr_ap:.4f} | Test AUC-PR = {single_feat_ap[feat]:.4f}")

print(f"\n★ Best Single Non-Length Feature (selected on train): '{best_single_feat}' (Test AUC-PR = {best_single_ap:.4f})")


# ============================================================
# 5. FIT LOGISTIC REGRESSION, RANDOM FOREST & GRADIENT BOOSTING
# ============================================================
groups = train_df["sample_id"].values
cv_splitter = GroupKFold(n_splits=5)

models = {}
cv_scores = {}
test_scores = {}
test_aps = {}

print("\n" + "=" * 75)
print("   SECTION 5: META-CLASSIFIER TRAINING & CROSS-VALIDATION")
print("=" * 75)

# 1. LR with Task 1 features only (7 features)
if t1_feature_cols:
    print(f"\n[1/5] Training LR (Task 1 non-length features: {len(t1_feature_cols)} feats)...")
    pipe_lr_t1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
    ])
    grid_lr_t1 = GridSearchCV(pipe_lr_t1, {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]}, scoring="average_precision", cv=cv_splitter, n_jobs=-1, refit=True)
    grid_lr_t1.fit(train_df[t1_feature_cols].values, y_train, groups=groups)
    models["LR (T1 features)"] = grid_lr_t1
    cv_scores["LR (T1 features)"] = grid_lr_t1.best_score_
    pred = grid_lr_t1.predict_proba(test_df[t1_feature_cols].values)[:, 1]
    test_scores["LR (T1 features)"] = pred
    test_aps["LR (T1 features)"] = average_precision_score(y_test, pred)
    print(f"      CV AUC-PR={cv_scores['LR (T1 features)']:.4f} | Test AUC-PR={test_aps['LR (T1 features)']:.4f}")

# 2. LR with Task 2 features only (8 features)
if t2_feature_cols:
    print(f"\n[2/5] Training LR (Task 2 features: {len(t2_feature_cols)} feats)...")
    pipe_lr_t2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
    ])
    grid_lr_t2 = GridSearchCV(pipe_lr_t2, {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]}, scoring="average_precision", cv=cv_splitter, n_jobs=-1, refit=True)
    grid_lr_t2.fit(train_df[t2_feature_cols].values, y_train, groups=groups)
    models["LR (T2 features)"] = grid_lr_t2
    cv_scores["LR (T2 features)"] = grid_lr_t2.best_score_
    pred = grid_lr_t2.predict_proba(test_df[t2_feature_cols].values)[:, 1]
    test_scores["LR (T2 features)"] = pred
    test_aps["LR (T2 features)"] = average_precision_score(y_test, pred)
    print(f"      CV AUC-PR={cv_scores['LR (T2 features)']:.4f} | Test AUC-PR={test_aps['LR (T2 features)']:.4f}")

# 3. LR with All 15 features
print(f"\n[3/5] Training LR (All 15 non-length features)...")
pipe_lr_all = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
])
grid_lr_all = GridSearchCV(pipe_lr_all, {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]}, scoring="average_precision", cv=cv_splitter, n_jobs=-1, refit=True)
grid_lr_all.fit(train_df[all_feature_cols].values, y_train, groups=groups)
models["LR (All features)"] = grid_lr_all
cv_scores["LR (All features)"] = grid_lr_all.best_score_
pred = grid_lr_all.predict_proba(test_df[all_feature_cols].values)[:, 1]
test_scores["LR (All features)"] = pred
test_aps["LR (All features)"] = average_precision_score(y_test, pred)
print(f"      CV AUC-PR={cv_scores['LR (All features)']:.4f} | Test AUC-PR={test_aps['LR (All features)']:.4f}")

# 4. Random Forest with All 15 features
print(f"\n[4/5] Training Random Forest (All 15 features)...")
pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=SEED))
])
grid_rf = GridSearchCV(pipe_rf, {"rf__n_estimators": [200, 500], "rf__max_depth": [3, 5, 7]}, scoring="average_precision", cv=cv_splitter, n_jobs=-1, refit=True)
grid_rf.fit(train_df[all_feature_cols].values, y_train, groups=groups)
models["RF (All features)"] = grid_rf
cv_scores["RF (All features)"] = grid_rf.best_score_
pred = grid_rf.predict_proba(test_df[all_feature_cols].values)[:, 1]
test_scores["RF (All features)"] = pred
test_aps["RF (All features)"] = average_precision_score(y_test, pred)
print(f"      CV AUC-PR={cv_scores['RF (All features)']:.4f} | Test AUC-PR={test_aps['RF (All features)']:.4f}")

# 5. Gradient Boosting with All 15 features
print(f"\n[5/5] Training Gradient Boosting (All 15 features)...")
pipe_gbt = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("gbt", GradientBoostingClassifier(random_state=SEED))
])
grid_gbt = GridSearchCV(pipe_gbt, {"gbt__n_estimators": [100, 200], "gbt__max_depth": [2, 3], "gbt__learning_rate": [0.05, 0.1]}, scoring="average_precision", cv=cv_splitter, n_jobs=-1, refit=True)
grid_gbt.fit(train_df[all_feature_cols].values, y_train, groups=groups)
models["GBT (All features)"] = grid_gbt
cv_scores["GBT (All features)"] = grid_gbt.best_score_
pred = grid_gbt.predict_proba(test_df[all_feature_cols].values)[:, 1]
test_scores["GBT (All features)"] = pred
test_aps["GBT (All features)"] = average_precision_score(y_test, pred)
print(f"      CV AUC-PR={cv_scores['GBT (All features)']:.4f} | Test AUC-PR={test_aps['GBT (All features)']:.4f}")

ensemble_models = ["LR (All features)", "RF (All features)", "GBT (All features)"]
best_model_name = max(ensemble_models, key=lambda m: cv_scores[m])
print(f"\n★ Best Model (selected by CV on 15 features): '{best_model_name}' "
      f"(CV AUC-PR = {cv_scores[best_model_name]:.4f} | Test AUC-PR = {test_aps[best_model_name]:.4f})")


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

table_scores = {
    f"Best Single Feature ({best_single_feat})": single_feat_scores[best_single_feat],
}
if "LR (T1 features)" in test_scores:
    table_scores["Task 1 only: LR (T1 non-length)"] = test_scores["LR (T1 features)"]
if "LR (T2 features)" in test_scores:
    table_scores["Task 2 only: LR (T2 features)"] = test_scores["LR (T2 features)"]
table_scores.update({
    "LR (All 15 features)":  test_scores["LR (All features)"],
    "RF (All 15 features)":  test_scores["RF (All features)"],
    "GBT (All 15 features)": test_scores["GBT (All features)"],
    f"Best Ensemble ({best_model_name})": test_scores[best_model_name],
})

comparison_rows = []
for name, s in table_scores.items():
    b = boot_ap(s)
    ap = average_precision_score(y_test, s)
    comparison_rows.append({
        "Method": name,
        "AUC-PR (test)": round(ap, 4),
        "95% CI low": round(float(np.percentile(b, 2.5)), 4),
        "95% CI high": round(float(np.percentile(b, 97.5)), 4),
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n" + "=" * 80)
print("   FINAL COMPARISON TABLE (TEST SET AUC-PR WITH 95% BOOTSTRAP CI)")
print("   [EXCLUDING LENGTH-CONFOUNDED FEATURES]")
print("=" * 80)
print(comparison_df.to_string(index=False))


# ============================================================
# 7. PRECISION-RECALL CURVE PLOT
# ============================================================
plt.figure(figsize=(8.5, 6.2))
for m_label, s in [
    (f"Best Single: {best_single_feat}", single_feat_scores[best_single_feat]),
    ("LR (All 15 features)", test_scores["LR (All features)"]),
    ("RF (All 15 features)", test_scores["RF (All features)"]),
    ("GBT (All 15 features)", test_scores["GBT (All features)"]),
]:
    p, r, _ = precision_recall_curve(y_test, s)
    ap = average_precision_score(y_test, s)
    plt.plot(r, p, lw=2, label=f"{m_label} (AP={ap:.4f})")

plt.axhline(y_test.mean(), ls=":", color="gray", lw=1.2, label=f"Random ({y_test.mean():.4f})")
plt.xlabel("Recall", fontsize=11, fontweight="bold")
plt.ylabel("Precision", fontsize=11, fontweight="bold")
plt.title(f"Task 3.2: PR Curves (Without Length Features - {len(y_test)} instances)", fontsize=12, fontweight="bold")
plt.legend(loc="lower left", fontsize=8.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()

pr_plot_path = os.path.join(OUT_DIR, "task3_2_pr_curves.png")
plt.savefig(pr_plot_path, dpi=200)
plt.close()
print(f"\n✓ PR curves plot saved to: {pr_plot_path}")


# ============================================================
# 8. SAVE DELIVERABLES
# ============================================================
model_out = os.path.join(OUT_DIR, "task3_2_meta_classifier.joblib")
joblib.dump(models[best_model_name].best_estimator_, model_out)

csv_out = os.path.join(OUT_DIR, "task3_2_comparison.csv")
comparison_df.to_csv(csv_out, index=False)

results_json_out = os.path.join(OUT_DIR, "task3_2_results.json")
final_results = {
    "split_seed": SEED,
    "train_instances": len(train_df),
    "test_instances": len(test_df),
    "excluded_features": list(EXCLUDED_KEYS),
    "features_total_count": len(all_feature_cols),
    "all_features": all_feature_cols,
    "best_single_feature": {
        "feature": best_single_feat,
        "test_auc_pr": round(best_single_ap, 4),
    },
    "model_cv_scores": {k: round(v, 4) for k, v in cv_scores.items()},
    "model_test_scores": {k: round(v, 4) for k, v in test_aps.items()},
    "best_overall_model": best_model_name,
    "final_comparison_table": comparison_df.to_dict("records"),
}

with open(results_json_out, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

print("\n" + "=" * 75)
print("   DELIVERABLES SAVED SUCCESSFULLY")
print("=" * 75)
print(f"✓ Saved meta-classifier:  {model_out}")
print(f"✓ Saved comparison CSV:   {csv_out}")
print(f"✓ Saved results JSON:     {results_json_out}")
print(f"✓ Saved PR curves plot:   {pr_plot_path}")
print(f"\n⏱ TOTAL TIME: {int((time.time()-TOTAL_START_TIME)//60)}m {int((time.time()-TOTAL_START_TIME)%60)}s")
print("✓ Task 3.2 complete!")

