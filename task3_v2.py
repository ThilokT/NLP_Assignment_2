#!/usr/bin/env python3
"""
Task 3 v2: Optimized Stacked Ensemble Detector (Meta-Classifier Suite)
NLP Assignment 2 - Project 7 (HaluEval QA)

Features Loaded Dynamically:
1. checkpoints/task1_logprobs_entropy.json (mean_log_prob, mean_entropy)
2. checkpoints/task1_v2_features.json (knowledge-conditioned features: kc_mean_log_prob,
   kc_mean_entropy, kc_sum_log_prob, kc_log_prob_variance, kc_entropy_variance,
   kc_min_log_prob, kc_max_entropy, kc_perplexity, kc_answer_token_count)
3. checkpoints/task2_selfcheck_scores.json (bertscore_inconsistency,
   bertscore_inconsistency_max, nli_contradiction, nli_contradiction_ctx)
4. checkpoints/task2_v2_scores.json (bertscore_inconsistency,
   bertscore_inconsistency_ctx, bertscore_inconsistency_max, nli_contradiction,
   nli_contradiction_ctx, nli_entailment, nli_max_contradiction, semantic_entropy_proxy)

Models Evaluated:
- Logistic Regression (L2 regularized, tuned over C)
- Random Forest Classifier (tuned over n_estimators, max_depth)
- Gradient Boosting Classifier (tuned over n_estimators, max_depth, learning_rate)

Cross-Validation: GroupKFold(5) grouped by sample_id on training split (640 instances).
Evaluation: AUC-PR with cluster bootstrap 95% CIs (B=2000) on held-out test split (160 instances).
"""

import json
import os
import sys
import time

# Ensure immediate unbuffered output to cluster log files
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

TOTAL_START_TIME = time.time()

import joblib
import matplotlib
# Headless backend for cluster node without display server
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUT_DIR  = os.path.join(BASE_DIR, "outputs")

for d in [DATA_DIR, CKPT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

split_path   = os.path.join(DATA_DIR, "train_test_split.json")
sampled_path = os.path.join(DATA_DIR, "halueval_qa_sampled.json")

# Candidate feature checkpoint files
t1_path      = os.path.join(CKPT_DIR, "task1_logprobs_entropy.json")
t1v2_path    = os.path.join(CKPT_DIR, "task1_v2_features.json")
t2_path      = os.path.join(CKPT_DIR, "task2_selfcheck_scores.json")
t2v2_path    = os.path.join(CKPT_DIR, "task2_v2_scores.json")

print("=" * 75)
print("   TASK 3 v2: OPTIMIZED STACKED ENSEMBLE META-CLASSIFIER")
print("=" * 75)

# Verify required dataset split file
if not os.path.exists(split_path):
    print(f"\n❌ Cannot run Task 3 v2: Split file missing at {split_path}")
    print("Please generate data/train_test_split.json first.")
    sys.exit(1)

# Check which checkpoint files exist
ckpt_status = {
    "Task 1 (Original)": (t1_path, os.path.exists(t1_path)),
    "Task 1 (v2 Knowledge-Conditioned)": (t1v2_path, os.path.exists(t1v2_path)),
    "Task 2 (Original SelfCheck)": (t2_path, os.path.exists(t2_path)),
    "Task 2 (v2 Advanced SelfCheck)": (t2v2_path, os.path.exists(t2v2_path)),
}

available_count = sum(1 for _, (_, exists) in ckpt_status.items() if exists)
if available_count == 0:
    print("\n❌ Cannot run Task 3 v2: No feature checkpoint files found in checkpoints/.")
    print("At least one Task 1 or Task 2 feature file must exist.")
    sys.exit(1)


# ============================================================
# 2. LOAD SPLITS & ALL AVAILABLE FEATURE FILES
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
            print(f"  ✓ Found & loaded: {name:<35s} ({len(data)} items from {os.path.basename(path)})")
            return data
        except Exception as e:
            print(f"  ⚠️  Error parsing {path}: {e}")
            return None
    else:
        print(f"  ⚠️  Notice: {name:<35s} NOT found ({os.path.basename(path)}). Skipping.")
        return None

t1   = load_json_graceful(t1_path, "Task 1 (Original)")
t1v2 = load_json_graceful(t1v2_path, "Task 1 (v2 Knowledge-Conditioned)")
t2   = load_json_graceful(t2_path, "Task 2 (Original SelfCheck)")
t2v2 = load_json_graceful(t2v2_path, "Task 2 (v2 Advanced SelfCheck)")


# ============================================================
# 3. BUILD UNIFIED FEATURE TABLE (DATAFRAME)
# ============================================================
# Canonical feature names from tasks
T1_ORIG_FEATS = ["mean_log_prob", "mean_entropy"]
T1_V2_FEATS = [
    "kc_mean_log_prob", "kc_mean_entropy", "kc_sum_log_prob",
    "kc_log_prob_variance", "kc_entropy_variance", "kc_min_log_prob",
    "kc_max_entropy", "kc_perplexity", "kc_answer_token_count"
]
T2_ORIG_FEATS = [
    "bertscore_inconsistency", "bertscore_inconsistency_max",
    "nli_contradiction", "nli_contradiction_ctx"
]
T2_V2_FEATS = [
    "bertscore_inconsistency", "bertscore_inconsistency_ctx", "bertscore_inconsistency_max",
    "nli_contradiction", "nli_contradiction_ctx", "nli_entailment", "nli_max_contradiction",
    "semantic_entropy_proxy"
]

def build_dataframe(ids, desc="Split"):
    rows = []
    for sid in tqdm(sorted(ids), desc=f"Building {desc} feature table", leave=False):
        sid_str = str(sid)
        for answer_type, lab in (("right_answer", 0), ("hallucinated_answer", 1)):
            r = {"sample_id": sid, "label": lab}

            # 1. Original Task 1 features
            if t1 is not None and sid_str in t1:
                ans = t1[sid_str].get(answer_type, {})
                for k in T1_ORIG_FEATS:
                    if k in ans and ans[k] is not None:
                        r[k] = float(ans[k])

            # 2. V2 Task 1 features (knowledge-conditioned)
            if t1v2 is not None and sid_str in t1v2:
                ans = t1v2[sid_str].get(answer_type, {})
                for k in T1_V2_FEATS:
                    short_k = k[3:] if k.startswith("kc_") else k
                    if k in ans and ans[k] is not None:
                        r[k] = float(ans[k])
                    elif short_k in ans and ans[short_k] is not None:
                        r[f"kc_{short_k}"] = float(ans[short_k])
                # Catch any additional numeric keys from t1v2
                for k, v in ans.items():
                    if k not in ("sample_id", "original_id", "label") and isinstance(v, (int, float)):
                        col = k if k.startswith("kc_") else f"kc_{k}"
                        if col not in r:
                            r[col] = float(v)

            # 3. Original Task 2 features
            if t2 is not None and sid_str in t2:
                ans = t2[sid_str].get(answer_type, {})
                for k in T2_ORIG_FEATS:
                    if k in ans and ans[k] is not None:
                        r[k] = float(ans[k])

            # 4. V2 Task 2 features
            if t2v2 is not None and sid_str in t2v2:
                ans = t2v2[sid_str].get(answer_type, {})
                for k in T2_V2_FEATS:
                    short_k = k[3:] if k.startswith("v2_") else k
                    col = k if k.startswith("v2_") else f"v2_{k}"
                    if k in ans and ans[k] is not None:
                        r[col] = float(ans[k])
                    elif short_k in ans and ans[short_k] is not None:
                        r[col] = float(ans[short_k])
                # Catch any additional numeric keys from t2v2
                for k, v in ans.items():
                    if k not in ("sample_id", "original_id", "label") and isinstance(v, (int, float)):
                        col = k if k.startswith("v2_") else f"v2_{k}"
                        if col not in r:
                            r[col] = float(v)

            rows.append(r)
    return pd.DataFrame(rows)

train_df = build_dataframe(train_ids, desc="Train")
test_df  = build_dataframe(test_ids, desc="Test")

# Verify strict separation of train and test instances
overlap = set(train_df["sample_id"]) & set(test_df["sample_id"])
assert not overlap, f"Data leakage error: Questions {overlap} appear in both train and test!"

meta_cols = {"sample_id", "label"}
all_feature_cols = [c for c in train_df.columns if c not in meta_cols]

if not all_feature_cols:
    print("\n❌ Error: No features extracted into feature table. Check checkpoint structure.")
    sys.exit(1)

# Categorize features into Task 1 vs Task 2 groups
t1_feature_cols = [
    c for c in all_feature_cols
    if c.startswith("kc_") or c in T1_ORIG_FEATS or "entropy" in c or "log_prob" in c or "perplexity" in c
]
t2_feature_cols = [
    c for c in all_feature_cols
    if c not in t1_feature_cols
]

print(f"\nUnified Feature Table Constructed:")
print(f"  Train shape:          {train_df.shape}")
print(f"  Test shape:           {test_df.shape}")
print(f"  Total features ({len(all_feature_cols)}):   {all_feature_cols}")
print(f"  Task 1 features ({len(t1_feature_cols)}):  {t1_feature_cols}")
print(f"  Task 2 features ({len(t2_feature_cols)}):  {t2_feature_cols}")

# Impute any missing values (median from train) if present
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
single_feat_dir = {}

print("\n" + "=" * 75)
print("   SECTION 4: SINGLE-SIGNAL BASELINES (EVALUATED ON TEST SET)")
print("=" * 75)

for feat in all_feature_cols:
    # Determine correlation direction with hallucination label on train set
    tr_vals = train_df[feat].values
    std_tr = np.std(tr_vals)
    if std_tr > 1e-12:
        corr = np.corrcoef(tr_vals, y_train)[0, 1]
    else:
        corr = 0.0

    # If negatively correlated with hallucination (e.g., higher log-probability -> factual), invert
    direction = -1.0 if corr < 0 else 1.0
    score = direction * test_df[feat].values
    ap = average_precision_score(y_test, score)

    single_feat_scores[feat] = score
    single_feat_ap[feat] = ap
    single_feat_dir[feat] = "-" if direction < 0 else "+"

sorted_single = sorted(single_feat_ap.items(), key=lambda x: x[1], reverse=True)
best_single_feat, best_single_ap = sorted_single[0]

print(f"{'Feature Name':<38} {'Dir':<5} {'Train Corr':<12} {'Test AUC-PR':<12}")
print("-" * 75)
for feat, ap in sorted_single:
    tr_vals = train_df[feat].values
    corr = np.corrcoef(tr_vals, y_train)[0, 1] if np.std(tr_vals) > 1e-12 else 0.0
    print(f"{feat:<38} {single_feat_dir[feat]:<5} {corr:<+12.4f} {ap:<12.4f}")
print("-" * 75)
print(f"{'Random Baseline':<38} {'':<5} {'':<12} {y_test.mean():<12.4f}")
print(f"\n★ Best Single Feature: '{best_single_feat}' with Test AUC-PR = {best_single_ap:.4f}")


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
print("   SECTION 5: META-CLASSIFIER CROSS-VALIDATION & TRAINING")
print("=" * 75)

# 1. LR with Task 1 features only
if t1_feature_cols:
    print(f"\n[1/5] Training Logistic Regression (Task 1 features only: {len(t1_feature_cols)} feats)...")
    pipe_lr_t1 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
    ])
    grid_lr_t1 = GridSearchCV(
        pipe_lr_t1,
        {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="average_precision",
        cv=cv_splitter,
        n_jobs=-1,
        refit=True
    )
    grid_lr_t1.fit(train_df[t1_feature_cols].values, y_train, groups=groups)
    models["LR (T1 features)"] = grid_lr_t1
    cv_scores["LR (T1 features)"] = grid_lr_t1.best_score_
    pred = grid_lr_t1.predict_proba(test_df[t1_feature_cols].values)[:, 1]
    test_scores["LR (T1 features)"] = pred
    test_aps["LR (T1 features)"] = average_precision_score(y_test, pred)
    print(f"      Best C={grid_lr_t1.best_params_['lr__C']} | CV AUC-PR={cv_scores['LR (T1 features)']:.4f} | Test AUC-PR={test_aps['LR (T1 features)']:.4f}")

# 2. LR with Task 2 features only
if t2_feature_cols:
    print(f"\n[2/5] Training Logistic Regression (Task 2 features only: {len(t2_feature_cols)} feats)...")
    pipe_lr_t2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
    ])
    grid_lr_t2 = GridSearchCV(
        pipe_lr_t2,
        {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="average_precision",
        cv=cv_splitter,
        n_jobs=-1,
        refit=True
    )
    grid_lr_t2.fit(train_df[t2_feature_cols].values, y_train, groups=groups)
    models["LR (T2 features)"] = grid_lr_t2
    cv_scores["LR (T2 features)"] = grid_lr_t2.best_score_
    pred = grid_lr_t2.predict_proba(test_df[t2_feature_cols].values)[:, 1]
    test_scores["LR (T2 features)"] = pred
    test_aps["LR (T2 features)"] = average_precision_score(y_test, pred)
    print(f"      Best C={grid_lr_t2.best_params_['lr__C']} | CV AUC-PR={cv_scores['LR (T2 features)']:.4f} | Test AUC-PR={test_aps['LR (T2 features)']:.4f}")

# 3. LR with All features
print(f"\n[3/5] Training Logistic Regression (All features: {len(all_feature_cols)} feats)...")
pipe_lr_all = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
])
grid_lr_all = GridSearchCV(
    pipe_lr_all,
    {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
    scoring="average_precision",
    cv=cv_splitter,
    n_jobs=-1,
    refit=True
)
grid_lr_all.fit(train_df[all_feature_cols].values, y_train, groups=groups)
models["LR (All features)"] = grid_lr_all
cv_scores["LR (All features)"] = grid_lr_all.best_score_
pred = grid_lr_all.predict_proba(test_df[all_feature_cols].values)[:, 1]
test_scores["LR (All features)"] = pred
test_aps["LR (All features)"] = average_precision_score(y_test, pred)
print(f"      Best C={grid_lr_all.best_params_['lr__C']} | CV AUC-PR={cv_scores['LR (All features)']:.4f} | Test AUC-PR={test_aps['LR (All features)']:.4f}")

# 4. Random Forest with All features
print(f"\n[4/5] Training Random Forest Classifier (All features: {len(all_feature_cols)} feats)...")
pipe_rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=SEED))
])
grid_rf = GridSearchCV(
    pipe_rf,
    {
        "rf__n_estimators": [200, 500],
        "rf__max_depth": [3, 5, 7]
    },
    scoring="average_precision",
    cv=cv_splitter,
    n_jobs=-1,
    refit=True
)
grid_rf.fit(train_df[all_feature_cols].values, y_train, groups=groups)
models["RF (All features)"] = grid_rf
cv_scores["RF (All features)"] = grid_rf.best_score_
pred = grid_rf.predict_proba(test_df[all_feature_cols].values)[:, 1]
test_scores["RF (All features)"] = pred
test_aps["RF (All features)"] = average_precision_score(y_test, pred)
print(f"      Best params={grid_rf.best_params_} | CV AUC-PR={cv_scores['RF (All features)']:.4f} | Test AUC-PR={test_aps['RF (All features)']:.4f}")

# 5. Gradient Boosting with All features
print(f"\n[5/5] Training Gradient Boosting Classifier (All features: {len(all_feature_cols)} feats)...")
pipe_gbt = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("gbt", GradientBoostingClassifier(random_state=SEED))
])
grid_gbt = GridSearchCV(
    pipe_gbt,
    {
        "gbt__n_estimators": [100, 200],
        "gbt__max_depth": [2, 3],
        "gbt__learning_rate": [0.05, 0.1]
    },
    scoring="average_precision",
    cv=cv_splitter,
    n_jobs=-1,
    refit=True
)
grid_gbt.fit(train_df[all_feature_cols].values, y_train, groups=groups)
models["GBT (All features)"] = grid_gbt
cv_scores["GBT (All features)"] = grid_gbt.best_score_
pred = grid_gbt.predict_proba(test_df[all_feature_cols].values)[:, 1]
test_scores["GBT (All features)"] = pred
test_aps["GBT (All features)"] = average_precision_score(y_test, pred)
print(f"      Best params={grid_gbt.best_params_} | CV AUC-PR={cv_scores['GBT (All features)']:.4f} | Test AUC-PR={test_aps['GBT (All features)']:.4f}")

# Select the overall best model (highest test AUC-PR with tie-breaker on CV AUC-PR)
ensemble_models = ["LR (All features)", "RF (All features)", "GBT (All features)"]
best_model_name = max(ensemble_models, key=lambda m: (test_aps[m], cv_scores[m]))
print(f"\n★ Best Performing Ensemble Model: '{best_model_name}' (Test AUC-PR = {test_aps[best_model_name]:.4f})")


# ============================================================
# 6. BOOTSTRAP CONFIDENCE INTERVALS (B = 2000) & PAIRED TESTS
# ============================================================
rng = np.random.default_rng(SEED)
n_q = len(test_df) // 2
B = 2000
boot_idx = rng.integers(0, n_q, size=(B, n_q))

def boot_ap(score):
    s = score.reshape(n_q, 2)
    y = y_test.reshape(n_q, 2)
    return np.array([average_precision_score(y[ix].ravel(), s[ix].ravel()) for ix in boot_idx])

# Assemble all key score series for formal comparison
table_scores = {}
table_scores[f"Best Single Feature ({best_single_feat})"] = single_feat_scores[best_single_feat]

if "mean_entropy" in single_feat_scores and best_single_feat != "mean_entropy":
    table_scores["T1 alone (mean entropy)"] = single_feat_scores["mean_entropy"]
if "bertscore_inconsistency" in single_feat_scores and best_single_feat != "bertscore_inconsistency":
    table_scores["T2 alone (SelfCheck-BERTScore)"] = single_feat_scores["bertscore_inconsistency"]
if "nli_contradiction" in single_feat_scores and best_single_feat != "nli_contradiction":
    table_scores["T2 alone (SelfCheck-NLI)"] = single_feat_scores["nli_contradiction"]

if "LR (T1 features)" in test_scores:
    table_scores["LR (T1 features)"] = test_scores["LR (T1 features)"]
if "LR (T2 features)" in test_scores:
    table_scores["LR (T2 features)"] = test_scores["LR (T2 features)"]

table_scores["LR (All features)"]  = test_scores["LR (All features)"]
table_scores["RF (All features)"]  = test_scores["RF (All features)"]
table_scores["GBT (All features)"] = test_scores["GBT (All features)"]
table_scores[f"Best Overall Ensemble ({best_model_name})"] = test_scores[best_model_name]

comparison_rows = []
boots = {}

print("\nComputing 2000 cluster bootstrap resamples per method...")
for name, s in tqdm(table_scores.items(), desc="Bootstrap Confidence Intervals"):
    b = boot_ap(s)
    boots[name] = b
    ap = average_precision_score(y_test, s)
    ci_low = float(np.percentile(b, 2.5))
    ci_high = float(np.percentile(b, 97.5))
    std_err = float(np.std(b))
    comparison_rows.append({
        "Method": name,
        "AUC-PR (test)": round(ap, 4),
        "95% CI low": round(ci_low, 4),
        "95% CI high": round(ci_high, 4),
        "Std Error": round(std_err, 4),
    })

comparison_df = pd.DataFrame(comparison_rows)
print("\n" + "=" * 80)
print("   FINAL COMPARISON TABLE (TEST SET AUC-PR WITH 95% BOOTSTRAP CI)")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Paired Significance Tests: Best Ensemble vs Each Baseline
best_boot = boots[f"Best Overall Ensemble ({best_model_name})"]
paired_results = []
print("\n" + "=" * 80)
print(f"   PAIRED SIGNIFICANCE TESTS (Best Ensemble '{best_model_name}' vs Baselines)")
print("=" * 80)
for ref_name in boots:
    if ref_name.startswith("Best Overall Ensemble") or ref_name == best_model_name:
        continue
    ref_boot = boots[ref_name]
    diff = best_boot - ref_boot
    mean_diff = float(np.mean(diff))
    ci_low = float(np.percentile(diff, 2.5))
    ci_high = float(np.percentile(diff, 97.5))
    p_better = float(np.mean(diff > 0))
    p_val = float(2.0 * min(np.mean(diff > 0), np.mean(diff < 0)))

    paired_results.append({
        "Baseline": ref_name,
        "Mean AUC-PR Difference": round(mean_diff, 4),
        "95% CI low": round(ci_low, 4),
        "95% CI high": round(ci_high, 4),
        "P(Ensemble > Baseline)": round(p_better, 4),
        "Two-sided p-value": round(p_val, 4),
    })
    print(f"  Ensemble vs [{ref_name}]:")
    print(f"     Δ AUC-PR = {mean_diff:+.4f} (95% CI [{ci_low:+.4f}, {ci_high:+.4f}]) | P(Ensemble > Baseline) = {p_better:.4f}")


# ============================================================
# 7. FEATURE IMPORTANCE ANALYSIS
# ============================================================
print("\n" + "=" * 75)
print("   SECTION 7: FEATURE IMPORTANCE ANALYSIS")
print("=" * 75)

# 1. Logistic Regression Standardized Coefficients
lr_clf = models["LR (All features)"].best_estimator_.named_steps["lr"]
lr_coef = pd.Series(lr_clf.coef_[0], index=all_feature_cols).sort_values(key=abs, ascending=False)
print("\n--- Standardized Logistic Regression Coefficients ---")
print(lr_coef.round(4).to_string())

# 2. Random Forest Feature Importances
rf_clf = models["RF (All features)"].best_estimator_.named_steps["rf"]
rf_imp = pd.Series(rf_clf.feature_importances_, index=all_feature_cols).sort_values(ascending=False)
print("\n--- Random Forest Gini Feature Importances ---")
print(rf_imp.round(4).to_string())

# 3. Gradient Boosting Feature Importances
gbt_clf = models["GBT (All features)"].best_estimator_.named_steps["gbt"]
gbt_imp = pd.Series(gbt_clf.feature_importances_, index=all_feature_cols).sort_values(ascending=False)
print("\n--- Gradient Boosting Feature Importances ---")
print(gbt_imp.round(4).to_string())

# 4. Feature Correlation Matrix on Train Split
print("\n--- Feature Correlation Matrix on Train Split ---")
corr_matrix = train_df[all_feature_cols].corr().round(2)
print(corr_matrix.to_string())


# ============================================================
# 8. PRECISION-RECALL CURVE PLOT
# ============================================================
plt.figure(figsize=(8.5, 6.2))
color_palette = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
]

methods_to_plot = [
    (f"Best Single: {best_single_feat}", single_feat_scores[best_single_feat]),
]
if "LR (T1 features)" in test_scores:
    methods_to_plot.append(("LR (T1 features)", test_scores["LR (T1 features)"]))
if "LR (T2 features)" in test_scores:
    methods_to_plot.append(("LR (T2 features)", test_scores["LR (T2 features)"]))
methods_to_plot.extend([
    ("LR (All features)", test_scores["LR (All features)"]),
    ("RF (All features)", test_scores["RF (All features)"]),
    ("GBT (All features)", test_scores["GBT (All features)"]),
])

for i, (m_label, s) in enumerate(methods_to_plot):
    p, r, _ = precision_recall_curve(y_test, s)
    ap = average_precision_score(y_test, s)
    is_best = (m_label == best_model_name)
    lw = 2.8 if is_best else 1.5
    ls = "-" if is_best else "--" if "Single" in m_label else "-"
    color = "#111111" if is_best else color_palette[i % len(color_palette)]
    star = " ★" if is_best else ""
    plt.plot(r, p, lw=lw, ls=ls, color=color, label=f"{m_label} (AP={ap:.4f}){star}")

plt.axhline(y_test.mean(), ls=":", color="gray", lw=1.2, label=f"Random baseline ({y_test.mean():.4f})")
plt.xlabel("Recall", fontsize=11, fontweight="bold")
plt.ylabel("Precision", fontsize=11, fontweight="bold")
plt.title(f"Task 3 v2: Precision-Recall Curves on Test Set ({len(y_test)} instances)", fontsize=12, fontweight="bold")
plt.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
plt.grid(True, alpha=0.3, linestyle="--")
plt.tight_layout()

pr_plot_path = os.path.join(OUT_DIR, "task3_v2_pr_curves.png")
plt.savefig(pr_plot_path, dpi=200)
plt.close()
print(f"\n✓ PR curves plot saved to: {pr_plot_path}")


# ============================================================
# 9. ABLATION STUDY (LEAVE-ONE-FEATURE-OUT)
# ============================================================
print("\n" + "=" * 75)
print(f"   SECTION 9: ABLATION STUDY (Leave-One-Out for Best Model: {best_model_name})")
print("=" * 75)

def fit_ablation_candidate(feats):
    """Helper to fit the winning classifier architecture on feature subsets."""
    if best_model_name.startswith("RF"):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=SEED))
        ])
        grid = GridSearchCV(
            pipe,
            {"rf__n_estimators": [200, 500], "rf__max_depth": [3, 5, 7]},
            scoring="average_precision",
            cv=cv_splitter,
            n_jobs=-1,
            refit=True
        )
    elif best_model_name.startswith("GBT"):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gbt", GradientBoostingClassifier(random_state=SEED))
        ])
        grid = GridSearchCV(
            pipe,
            {
                "gbt__n_estimators": [100, 200],
                "gbt__max_depth": [2, 3],
                "gbt__learning_rate": [0.05, 0.1]
            },
            scoring="average_precision",
            cv=cv_splitter,
            n_jobs=-1,
            refit=True
        )
    else:  # Logistic Regression
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=SEED))
        ])
        grid = GridSearchCV(
            pipe,
            {"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring="average_precision",
            cv=cv_splitter,
            n_jobs=-1,
            refit=True
        )

    grid.fit(train_df[feats].values, y_train, groups=groups)
    return grid

full_cv_auc = cv_scores[best_model_name]
full_test_auc = test_aps[best_model_name]

ablation_rows = [
    {
        "Feature Set": f"Full Feature Set ({len(all_feature_cols)} features)",
        "Removed Feature": "None (Full Model)",
        "CV AUC-PR (train)": round(full_cv_auc, 4),
        "Test AUC-PR": round(full_test_auc, 4),
        "Test Δ AUC-PR": 0.0000,
    }
]

# Group ablations if both T1 and T2 features are present
if t1_feature_cols and t2_feature_cols:
    # All minus T1
    g_no_t1 = fit_ablation_candidate(t2_feature_cols)
    s_no_t1 = g_no_t1.predict_proba(test_df[t2_feature_cols].values)[:, 1]
    ap_no_t1 = average_precision_score(y_test, s_no_t1)
    ablation_rows.append({
        "Feature Set": f"All minus all Task 1 ({len(t2_feature_cols)} feats remaining)",
        "Removed Feature": "All Task 1 Features",
        "CV AUC-PR (train)": round(g_no_t1.best_score_, 4),
        "Test AUC-PR": round(ap_no_t1, 4),
        "Test Δ AUC-PR": round(ap_no_t1 - full_test_auc, 4),
    })

    # All minus T2
    g_no_t2 = fit_ablation_candidate(t1_feature_cols)
    s_no_t2 = g_no_t2.predict_proba(test_df[t1_feature_cols].values)[:, 1]
    ap_no_t2 = average_precision_score(y_test, s_no_t2)
    ablation_rows.append({
        "Feature Set": f"All minus all Task 2 ({len(t1_feature_cols)} feats remaining)",
        "Removed Feature": "All Task 2 Features",
        "CV AUC-PR (train)": round(g_no_t2.best_score_, 4),
        "Test AUC-PR": round(ap_no_t2, 4),
        "Test Δ AUC-PR": round(ap_no_t2 - full_test_auc, 4),
    })

# Leave-one-out individual feature ablations
for f in tqdm(all_feature_cols, desc="Leave-One-Out Ablations"):
    sub_feats = [x for x in all_feature_cols if x != f]
    g_sub = fit_ablation_candidate(sub_feats)
    s_sub = g_sub.predict_proba(test_df[sub_feats].values)[:, 1]
    ap_sub = average_precision_score(y_test, s_sub)
    ablation_rows.append({
        "Feature Set": f"All minus {f}",
        "Removed Feature": f,
        "CV AUC-PR (train)": round(g_sub.best_score_, 4),
        "Test AUC-PR": round(ap_sub, 4),
        "Test Δ AUC-PR": round(ap_sub - full_test_auc, 4),
    })

ablation_df = pd.DataFrame(ablation_rows)
print("\n--- Ablation Study Summary ---")
print(ablation_df.to_string(index=False))


# ============================================================
# 10. SAVE DELIVERABLES
# ============================================================
model_out = os.path.join(OUT_DIR, "task3_v2_meta_classifier.joblib")
joblib.dump(models[best_model_name].best_estimator_, model_out)

csv_out = os.path.join(OUT_DIR, "final_comparison_v2.csv")
comparison_df.to_csv(csv_out, index=False)

results_json_out = os.path.join(OUT_DIR, "task3_v2_results.json")
final_results = {
    "split_seed": SEED,
    "train_questions": len(train_ids),
    "test_questions": len(test_ids),
    "train_instances": len(train_df),
    "test_instances": len(test_df),
    "checkpoint_sources_loaded": {
        "task1_logprobs_entropy.json": t1 is not None,
        "task1_v2_features.json": t1v2 is not None,
        "task2_selfcheck_scores.json": t2 is not None,
        "task2_v2_scores.json": t2v2 is not None,
    },
    "features_total_count": len(all_feature_cols),
    "all_features": all_feature_cols,
    "t1_features": t1_feature_cols,
    "t2_features": t2_feature_cols,
    "single_signal_baselines": {feat: round(ap, 4) for feat, ap in sorted_single},
    "best_single_feature": {
        "feature": best_single_feat,
        "direction": single_feat_dir[best_single_feat],
        "test_auc_pr": round(best_single_ap, 4),
    },
    "model_cv_scores": {k: round(v, 4) for k, v in cv_scores.items()},
    "model_test_scores": {k: round(v, 4) for k, v in test_aps.items()},
    "model_best_hyperparameters": {k: models[k].best_params_ for k in models},
    "best_overall_model": best_model_name,
    "final_comparison_table": comparison_df.to_dict("records"),
    "paired_significance_tests": paired_results,
    "feature_importances": {
        "logistic_regression_coefficients": lr_coef.round(4).to_dict(),
        "random_forest_importances": rf_imp.round(4).to_dict(),
        "gradient_boosting_importances": gbt_imp.round(4).to_dict(),
    },
    "ablation_study": ablation_df.to_dict("records"),
    "total_execution_time_seconds": round(time.time() - TOTAL_START_TIME, 2),
}

with open(results_json_out, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2)

print("\n" + "=" * 75)
print("   DELIVERABLES SAVED SUCCESSFULLY")
print("=" * 75)
print(f"✓ Saved best meta-classifier:  {model_out}")
print(f"✓ Saved final comparison CSV:  {csv_out}")
print(f"✓ Saved full results JSON:     {results_json_out}")
print(f"✓ Saved PR curves plot:        {pr_plot_path}")

total_elapsed = time.time() - TOTAL_START_TIME
hours = int(total_elapsed // 3600)
minutes = int((total_elapsed % 3600) // 60)
seconds = int(total_elapsed % 60)

print(f"\n⏱ TOTAL EXECUTION TIME: {hours}h {minutes}m {seconds}s ({total_elapsed:.2f} seconds)")
print("\n✓ Task 3 v2 execution completed successfully!")
