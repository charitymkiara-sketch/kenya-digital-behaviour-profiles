#!/usr/bin/env python3
"""Reproduce the digital-behaviour clustering and prediction analyses.

Run from a folder containing the Afrobarometer Kenya Round 10 .sav file:
    python reproduce_digital_profiles.py --data KEN_R10.Data_28June24.wtd_.final_.release_updated.13Feb25.sav

Outputs are written to ./analysis_output. The 80:20 split, nested CV and
k-prototypes settings below implement the settings agreed for the revision.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import pyreadstat
from kmodes.kprototypes import KPrototypes
from scipy.stats import randint, uniform
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

SEED = 42
WEIGHT = "withinwt_hh"
CLUSTER_NUMERIC = ["Q90I", "Q90J", "Q65D", "Q65E", "Q90H_clean"]
CLUSTER_CATEGORICAL = ["Q90H1_clean"]
PROFILE_INPUTS = CLUSTER_NUMERIC + CLUSTER_CATEGORICAL
ORDINAL_CODES = ["Q90I", "Q90J", "Q65D", "Q65E"]
PREDICTOR_NUMERIC = ["Q1", "Q96", "LPI"]
PREDICTOR_CATEGORICAL = [
    "Q90D", "Q90E", "Q94A", "URBRUR", "Q101", "EA_SVC_A", "EA_FAC_F"
]
PREDICTORS = PREDICTOR_NUMERIC + PREDICTOR_CATEGORICAL


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented missing-value and no-phone rules."""
    d = data.copy()
    # These survey items use 8/9 (or 98/99) for refusal/don't know.
    for col in ["Q90I", "Q90J", "Q65D", "Q65E"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").replace({8: np.nan, 9: np.nan})

    # 7 means no personal phone, 8 refusal, 9 don't know.
    d["Q90H_clean"] = pd.to_numeric(d["Q90H"], errors="coerce").replace(
        {7: 0, 8: np.nan, 9: np.nan}
    )
    d["Q90H1_clean"] = pd.to_numeric(d["Q90H1_KEN"], errors="coerce").replace(
        {-1: np.nan, 8: np.nan, 9: np.nan}
    )
    no_phone = pd.to_numeric(d["Q90H"], errors="coerce").eq(7)
    d.loc[no_phone, "Q90H1_clean"] = 6  # distinct nominal category: no personal phone

    # Construct the mean Lived Poverty Index only when all five items are valid.
    lpi_items = ["Q7A", "Q7B", "Q7C", "Q7D", "Q7E"]
    lpi = d[lpi_items].apply(pd.to_numeric, errors="coerce").replace({8: np.nan, 9: np.nan})
    d["LPI"] = lpi.mean(axis=1, skipna=False)

    # Survey non-response codes in model predictors, based on the codebook.
    d["Q96"] = pd.to_numeric(d["Q96"], errors="coerce").replace({98: np.nan, 99: np.nan, -1: np.nan})
    for col in ["Q90D", "Q90E", "Q94A"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").replace({8: np.nan, 9: np.nan, -1: np.nan})
    d["EA_FAC_F"] = pd.to_numeric(d["EA_FAC_F"], errors="coerce").replace({9: np.nan, -1: np.nan})
    d["EA_SVC_A"] = pd.to_numeric(d["EA_SVC_A"], errors="coerce").replace({9: np.nan, -1: np.nan})
    d["Q1"] = pd.to_numeric(d["Q1"], errors="coerce")
    d["URBRUR"] = d["URBRUR"].astype("object")
    d["Q101"] = d["Q101"].astype("object")
    return d


def make_cluster_matrix(d: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Median-impute/standardise numeric inputs; mode-impute and one-hot nominal input."""
    num = d[CLUSTER_NUMERIC].copy()
    num = num.fillna(num.median())
    scaler = StandardScaler()
    num_scaled = pd.DataFrame(scaler.fit_transform(num), columns=CLUSTER_NUMERIC, index=d.index)
    cat = d[CLUSTER_CATEGORICAL].copy()
    cat = cat.fillna(cat.mode(dropna=True).iloc[0])
    cat = cat.astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = ohe.fit_transform(cat)
    matrix = np.column_stack([num_scaled.to_numpy(), cat_encoded])
    return matrix, num_scaled, cat


def weighted_profile_share(labels: np.ndarray, weights: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"profile": labels, "weight": weights.to_numpy()})
    share = frame.groupby("profile", dropna=False)["weight"].sum()
    share = share / share.sum()
    return share.rename("weighted_share").reset_index()


def cluster_analysis(d: pd.DataFrame, out: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    matrix, num_scaled, cat = make_cluster_matrix(d)
    diagnostics = []
    models = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=50, random_state=SEED)
        labels = model.fit_predict(matrix)
        models[k] = (model, labels)
        diagnostics.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(matrix, labels),
            "calinski_harabasz": calinski_harabasz_score(matrix, labels),
            "davies_bouldin": davies_bouldin_score(matrix, labels),
        })
    pd.DataFrame(diagnostics).to_csv(out / "cluster_diagnostics_k2_k6.csv", index=False)

    km, labels = models[3]
    d["profile_id"] = labels
    # 50 subsamples of 80%; compare each subset's partition with the full-sample partition.
    rng = np.random.default_rng(SEED)
    stability = []
    n = len(matrix)
    for i in range(50):
        idx = np.sort(rng.choice(n, size=int(0.8 * n), replace=False))
        sub_labels = KMeans(n_clusters=3, n_init=50, random_state=SEED + i + 1).fit_predict(matrix[idx])
        stability.append({"replicate": i + 1, "ari_vs_full": adjusted_rand_score(labels[idx], sub_labels)})
    pd.DataFrame(stability).to_csv(out / "kmeans_subsample_stability.csv", index=False)

    # Complete-case K-means sensitivity analysis.
    complete = d[PROFILE_INPUTS].notna().all(axis=1).to_numpy()
    cc_matrix, _, _ = make_cluster_matrix(d.loc[complete].copy())
    cc_labels = KMeans(n_clusters=3, n_init=50, random_state=SEED).fit_predict(cc_matrix)
    pd.DataFrame([{
        "complete_case_n": int(complete.sum()),
        "ari_vs_primary_on_complete_cases": adjusted_rand_score(labels[complete], cc_labels),
    }]).to_csv(out / "complete_case_sensitivity.csv", index=False)

    # K-prototypes: numeric inputs remain numeric; main phone use remains categorical.
    # Missing numeric values use medians; missing category uses the mode. No-phone is its own category.
    kp_numeric = d[CLUSTER_NUMERIC].copy()
    kp_numeric = kp_numeric.fillna(kp_numeric.median())
    kp_numeric = StandardScaler().fit_transform(kp_numeric)
    kp_cat = cat[CLUSTER_CATEGORICAL].to_numpy(dtype=object)
    kp_data = np.column_stack([kp_numeric, kp_cat])
    kp = KPrototypes(
        n_clusters=3, init="Huang", n_init=30, max_iter=100,
        gamma=0.5, random_state=SEED, verbose=0,
    )
    kp_labels = kp.fit_predict(kp_data, categorical=[len(CLUSTER_NUMERIC)])
    kp_means = d.assign(kprototypes_profile=kp_labels).groupby("kprototypes_profile")["Q90J"].mean()
    kp_high_id = int(kp_means.idxmax())
    kp_y = (kp_labels == kp_high_id).astype(int)
    kp_share = weighted_profile_share(kp_labels, d[WEIGHT])
    kp_share["unweighted_n"] = pd.Series(kp_labels).value_counts().reindex(kp_share["profile"]).to_numpy()
    kp_share = kp_share.merge(kp_means.rename("mean_Q90J").reset_index(), left_on="profile", right_on="kprototypes_profile")
    kp_share.drop(columns="kprototypes_profile").to_csv(out / "kprototypes_profile_shares.csv", index=False)
    pd.DataFrame([{
        "k": 3,
        "initialization": "Huang",
        "n_init": 30,
        "max_iter": 100,
        "gamma": float(kp.gamma),
        "seed": SEED,
        "ari_vs_kmeans": adjusted_rand_score(labels, kp_labels),
    }]).to_csv(out / "kprototypes_sensitivity.csv", index=False)

    # Define high engagement as the profile with the highest mean Internet-use frequency.
    profile_means = d.assign(profile_id=labels).groupby("profile_id")[ORDINAL_CODES].mean()
    high_id = int(profile_means["Q90J"].idxmax())
    y = (labels == high_id).astype(int)
    d["profile_id"] = labels
    d["high_engagement"] = y
    profile_means.to_csv(out / "profile_centroids_original_scale.csv")
    weighted_profile_share(labels, d[WEIGHT]).to_csv(out / "weighted_profile_shares.csv", index=False)
    pd.DataFrame({"profile_id": labels, "high_engagement": y}).to_csv(out / "profile_assignments.csv", index=False)
    return labels, y, d


def make_pipeline(model) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric, PREDICTOR_NUMERIC),
        ("categorical", categorical, PREDICTOR_CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def model_specs():
    return {
        "logistic_regression": (
            make_pipeline(LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
            {"model__C": [0.01, 0.1, 1, 10, 100]}, 5,
        ),
        "random_forest": (
            make_pipeline(RandomForestClassifier(class_weight=None, random_state=SEED, n_jobs=1)),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [2, 3, 5],
                "model__max_features": ["sqrt", 0.5],
            }, 12,
        ),
        "xgboost": (
            make_pipeline(XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                random_state=SEED, n_jobs=1,
            )),
            {
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            }, 12,
        ),
    }


def score_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def prediction_analysis(d: pd.DataFrame, y: np.ndarray, out: Path) -> None:
    X = d[PREDICTORS].copy()
    y = np.asarray(y)
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    pd.DataFrame([{"n_development": len(X_dev), "n_holdout": len(X_hold), "seed": SEED}]).to_csv(
        out / "holdout_split.csv", index=False
    )

    specs = model_specs()
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows, importance_rows, holdout_rows = [], [], []
    for name, (pipe, params, n_iter) in specs.items():
        for fold, (tr, va) in enumerate(outer.split(X_dev, y_dev), start=1):
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            search = RandomizedSearchCV(
                estimator=clone(pipe), param_distributions=params,
                n_iter=n_iter, scoring="roc_auc", cv=inner,
                random_state=SEED, n_jobs=2, refit=True,
            )
            fit_args = {}
            if name == "xgboost":
                from sklearn.utils.class_weight import compute_sample_weight
                fit_args["model__sample_weight"] = compute_sample_weight("balanced", y_dev[tr])
            search.fit(X_dev.iloc[tr], y_dev[tr], **fit_args)
            best = search.best_estimator_
            prob = best.predict_proba(X_dev.iloc[va])[:, 1]
            pred = (prob >= 0.5).astype(int)
            row = score_row(name, y_dev[va], pred, prob)
            row.update({"outer_fold": fold, "best_params": json.dumps(search.best_params_, sort_keys=True)})
            cv_rows.append(row)
            if name == "logistic_regression":
                pi = permutation_importance(
                    best, X_dev.iloc[va], y_dev[va], scoring="roc_auc",
                    n_repeats=30, random_state=SEED + fold, n_jobs=1,
                )
                for feature, mean, sd in zip(PREDICTORS, pi.importances_mean, pi.importances_std):
                    importance_rows.append({"outer_fold": fold, "predictor": feature, "mean_auc_decrease": mean, "within_fold_sd": sd})

        # Tune only on development data, refit there, and evaluate once on holdout.
        final_search = RandomizedSearchCV(
            estimator=clone(pipe), param_distributions=params,
            n_iter=n_iter, scoring="roc_auc",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
            random_state=SEED, n_jobs=2, refit=True,
        )
        fit_args = {}
        if name == "xgboost":
            from sklearn.utils.class_weight import compute_sample_weight
            fit_args["model__sample_weight"] = compute_sample_weight("balanced", y_dev)
        final_search.fit(X_dev, y_dev, **fit_args)
        hold_prob = final_search.best_estimator_.predict_proba(X_hold)[:, 1]
        hold_pred = (hold_prob >= 0.5).astype(int)
        holdout_rows.append(score_row(name, y_hold, hold_pred, hold_prob))
        pd.DataFrame(confusion_matrix(y_hold, hold_pred, labels=[0, 1])).to_csv(
            out / f"holdout_confusion_{name}.csv", index=False, header=False
        )
        (100 * pd.DataFrame(confusion_matrix(y_hold, hold_pred, labels=[0, 1], normalize="true"))).to_csv(
            out / f"holdout_confusion_row_percent_{name}.csv", index=False, header=False
        )

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "nested_cv_fold_results.csv", index=False)
    cv.groupby("model")[["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"]).to_csv(
        out / "nested_cv_summary.csv"
    )
    pd.DataFrame(holdout_rows).to_csv(out / "holdout_model_results.csv", index=False)
    imp = pd.DataFrame(importance_rows)
    imp.groupby("predictor")["mean_auc_decrease"].agg(["mean", "std"]).sort_values("mean", ascending=False).to_csv(
        out / "permutation_importance_logistic.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Afrobarometer Kenya Round 10 .sav file")
    parser.add_argument("--output", default="analysis_output", help="Output directory")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw, meta = pyreadstat.read_sav(args.data, apply_value_formats=False)
    required = set(
        ["Q90I", "Q90J", "Q65D", "Q65E", "Q90H", "Q90H1_KEN", WEIGHT,
         "Q7A", "Q7B", "Q7C", "Q7D", "Q7E", "Q1", "Q96", "Q90D", "Q90E",
         "Q94A", "URBRUR", "Q101", "EA_SVC_A", "EA_FAC_F"]
    )
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing required variables: {missing}")
    d = clean_data(raw)
    missingness = d[list(dict.fromkeys(PROFILE_INPUTS + PREDICTORS))].isna().sum().rename("missing_n").to_frame()
    missingness["missing_percent"] = 100 * missingness["missing_n"] / len(d)
    missingness.to_csv(out / "variable_missingness.csv")
    labels, y, d = cluster_analysis(d, out)
    prediction_analysis(d, y, out)
    versions = {pkg: version(pkg) for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "xgboost", "kmodes", "pyreadstat"]}
    (out / "software_versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    print(f"Completed. N={len(d)}; high-engagement n={int(y.sum())}; outputs: {out.resolve()}")
    print("Important: compare generated outputs with manuscript tables before submission.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
