#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 4-CLASS HIERARCHICAL RF PIPELINE


Stage-1: immotile vs motile
Stage-2: normal / stiff / circular (motile subset only)

- Patient-based StratifiedGroupKFold
- SMOTE + class_weight
- Compact wave features extracted from motion CSVs (normalized2)

"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

from glob import glob
from itertools import cycle
from datetime import datetime

import numpy as np
import pandas as pd

from scipy import stats
from scipy.fft import fft

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)

from imblearn.over_sampling import SMOTE

import matplotlib
matplotlib.use("Agg")  # GitHub/CI/headless safe
import matplotlib.pyplot as plt

# seaborn optional
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# shap optional
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


# =============================================================
# 0) CLI
# =============================================================
import argparse

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_bank_csv", required=True, help="Engineered feature bank CSV (must include filename + func3_label)")
    ap.add_argument("--normalized_dir", required=True, help="Directory containing motion CSVs (normalized2).")
    ap.add_argument("--output_dir", required=True, help="Where to save figures + logs.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[13, 27, 42], help="Seeds to evaluate; best by mean BACC will be selected.")
    ap.add_argument("--n_splits", type=int, default=5, help="StratifiedGroupKFold splits.")
    ap.add_argument("--figure_dpi", type=int, default=300)
    ap.add_argument("--figure_format", type=str, default="png", choices=["png", "pdf"])
    ap.add_argument("--no_shap", action="store_true", help="Disable SHAP even if installed.")
    ap.add_argument("--top_n_importance", type=int, default=20, help="Top-N features for importance plots.")
    # RF params
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--min_samples_leaf", type=int, default=2)
    ap.add_argument("--max_depth", type=int, default=None)
    return ap.parse_args()


# =============================================================
# 1) PATIENT MAPPING
# =============================================================

GEN_TOKENS = r"(?:dnah\d*|dnai\d+|rsph\d*[a-z]*|hydin|odaida|oda|ccdc\d+|lrrc\d+|armc4|dyx1c1|zmynd10|ktu|ndr?c|rsph)"
STRIP_SUFFIXES = [
    "_enhanced_analysis", "_motion_stats", "_bbox_motion_stats",
    "_interpolated", "_analysis", "_stats", "_mp4", "_csv"
]

def _normalize_tr(s: str) -> str:
    """Türkçe karakterleri düzelt, küçük harfe çevir, sadeleştir."""
    tbl = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    s = str(s).translate(tbl).lower()
    s = s.replace(".csv", "")
    s = re.sub(r"^frames_+", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    for suf in STRIP_SUFFIXES:
        s = re.sub(f"{suf}$", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _split_tokens(s: str):
    toks = s.split("_")
    while toks and re.fullmatch(r"(csv|mp4|v\d+)", toks[-1]):
        toks.pop()
    return toks

def extract_video_index(name_norm: str):
    m = re.search(r"(?:[_-]v(\d+)|[_-](\d+))$", name_norm)
    if m:
        for g in m.groups():
            if g:
                return int(g)
    return None

def extract_patient_base(filename: str) -> str:
    n = _normalize_tr(filename)
    toks = _split_tokens(n)

    gen_pos = None
    for i, t in enumerate(toks):
        if re.fullmatch(GEN_TOKENS, t):
            gen_pos = i
            break

    # dnah5_pat1_v1 ...
    if gen_pos == 0:
        pat = None
        for t in toks[1:4]:
            m = re.fullmatch(r"pat\d+", t)
            if m:
                pat = m.group(0)
                break
        if pat:
            base = f"{toks[0]}_{pat}"
        else:
            tail = [t for t in toks[1:4] if not re.fullmatch(r"(v\d+|\d+)", t)]
            base = "_".join([toks[0]] + (tail[:1] if tail else []))
        return base

    # gen daha sonra: soldakileri hasta baz al
    if gen_pos is not None and gen_pos > 0:
        left = [t for t in toks[:gen_pos] if not re.fullmatch(r"(v\d+|\d+)", t)]
        if left:
            return "_".join(left)

    # normal videolar
    if toks and toks[0] == "normal":
        tail = []
        for t in toks[1:4]:
            if re.search(r"[a-z]", t):
                tail.append(t)
        return "_".join(["normal"] + tail) if tail else "normal"

    # fallback
    if toks:
        if re.fullmatch(r"(v\d+|\d+)", toks[-1]):
            toks = toks[:-1]
        return "_".join(toks)
    return "unknown"


# =============================================================
# 2) COMPACT WAVE FEATURES from motion CSV
# =============================================================

WAVE_COLS = [
    "wave_mag_mean","wave_mag_std","wave_fft_peak",
    "wave_entropy","wave_repeatability",
    "wave_angle_change_mean","wave_score"
]

def build_motion_file_index(normalized_dir: str):
    """
    normalized_dir içinde CSV tarar (recursive).
    normalize edilmiş isim → path dict.
    Priorities: motion_stats > bbox_motion_stats > analysis > other
    """
    all_files = []
    all_files += glob(os.path.join(normalized_dir, "*.csv"))
    all_files += glob(os.path.join(normalized_dir, "**/*.csv"), recursive=True)
    all_files = sorted(set(all_files))

    def score_name(name: str) -> int:
        n = name.lower()
        if "motion_stats" in n and "bbox" not in n:
            return 4
        if "bbox_motion_stats" in n:
            return 3
        if "analysis" in n:
            return 2
        return 1

    idx = {}
    for p in all_files:
        base = os.path.basename(p)
        key  = _normalize_tr(base)
        sc   = score_name(base)
        if not key:
            continue
        if key not in idx or sc > idx[key][1]:
            idx[key] = (p, sc)
    return {k: v[0] for k, v in idx.items()}

def _get_series_by_candidates(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return df[c].to_numpy(dtype=float)
    return None

def extract_compact_wave_features_for_one(sequence_df: pd.DataFrame, max_len=30):
    if sequence_df is None or len(sequence_df) < 3:
        return {k: 0.0 for k in WAVE_COLS}

    seq = sequence_df.iloc[:max_len].copy()

    mag = _get_series_by_candidates(seq, ["mean_magnitude", "motion", "mag", "magnitude"])
    if mag is None:
        return {k: 0.0 for k in WAVE_COLS}

    mag = np.asarray(mag, dtype=float)
    wave_mag_mean = float(np.mean(mag))
    wave_mag_std  = float(np.std(mag))

    if len(mag) >= 8:
        mag_fft = np.abs(fft(mag))[: len(mag)//2]
        wave_fft_peak = float(np.argmax(mag_fft[1:]) + 1) if len(mag_fft) > 1 else 0.0
        wave_fft_energy = float(np.sum(mag_fft**2))
        wave_entropy    = float(stats.entropy(mag_fft + 1e-9))
    else:
        wave_fft_peak   = 0.0
        wave_fft_energy = 0.0
        wave_entropy    = 0.0

    if len(mag) >= 10:
        half = len(mag) // 2
        if half > 0:
            corr = np.corrcoef(mag[:half], mag[half:2*half])[0, 1]
            wave_repeatability = float(max(0.0, corr)) if np.isfinite(corr) else 0.0
        else:
            wave_repeatability = 0.0
    else:
        wave_repeatability = 0.0

    ang = _get_series_by_candidates(seq, ["mean_angle_deg", "angle_deg", "angle"])
    if ang is not None and len(ang) > 1:
        ang = np.asarray(ang, dtype=float)
        wave_angle_change_mean = float(np.mean(np.abs(np.diff(ang))))
    else:
        wave_angle_change_mean = 0.0

    def _safe_norm_scalar(x: float) -> float:
        if not np.isfinite(x) or abs(x) < 1e-12:
            return 0.0
        return float(np.clip(np.log1p(abs(x)) / 10.0, 0, 1))

    score_parts = [
        _safe_norm_scalar(wave_mag_mean),
        _safe_norm_scalar(wave_mag_std),
        _safe_norm_scalar(wave_fft_energy),
        float(np.clip(wave_repeatability, 0, 1)),
        _safe_norm_scalar(wave_angle_change_mean),
    ]
    wave_score = float(np.mean(score_parts))

    return {
        "wave_mag_mean": wave_mag_mean,
        "wave_mag_std": wave_mag_std,
        "wave_fft_peak": wave_fft_peak,
        "wave_entropy": wave_entropy,
        "wave_repeatability": wave_repeatability,
        "wave_angle_change_mean": wave_angle_change_mean,
        "wave_score": wave_score,
    }


# =============================================================
# 3) LOAD + MERGE engineered + wave
# =============================================================

def load_engineered_plus_wave(feature_bank_csv: str, normalized_dir: str):
    if not os.path.exists(feature_bank_csv):
        raise FileNotFoundError(feature_bank_csv)
    if not os.path.isdir(normalized_dir):
        raise NotADirectoryError(normalized_dir)

    df = pd.read_csv(feature_bank_csv)
    if "filename" not in df.columns:
        raise ValueError("feature_bank_csv must contain 'filename' column.")
    if "func3_label" not in df.columns:
        raise ValueError("feature_bank_csv must contain 'func3_label' column (immotile/normal/stiff/circular).")

    df = df[~df["func3_label"].isna()].copy()

    # patient_base fallback
    if "patient_base" not in df.columns or df["patient_base"].isna().all():
        df["filename_norm"]    = df["filename"].apply(_normalize_tr)
        df["video_index_hint"] = df["filename_norm"].apply(extract_video_index)
        df["patient_base"]     = df["filename"].apply(extract_patient_base)

    df = df[~df["patient_base"].isna()].copy()

    motion_index = build_motion_file_index(normalized_dir)

    wave_feats_list = []
    found_count, not_found = 0, 0

    for _, row in df.iterrows():
        key = _normalize_tr(row["filename"])
        if key in motion_index:
            found_count += 1
            try:
                sdf = pd.read_csv(motion_index[key])
                wf  = extract_compact_wave_features_for_one(sdf)
            except Exception:
                not_found += 1
                wf = {k: 0.0 for k in WAVE_COLS}
        else:
            not_found += 1
            wf = {k: 0.0 for k in WAVE_COLS}
        wave_feats_list.append(wf)

    df_wave = pd.DataFrame(wave_feats_list)
    df_comb = pd.concat([df.reset_index(drop=True), df_wave.reset_index(drop=True)], axis=1)

    return df_comb, {"motion_found": found_count, "motion_not_found": not_found, "motion_index_size": len(motion_index)}


# =============================================================
# 4) PREPARE X/y
# =============================================================

def prepare_xy(df_comb: pd.DataFrame):
    classes4 = ["immotile", "normal", "stiff", "circular"]
    label_map = {c: i for i, c in enumerate(classes4)}
    y4 = df_comb["func3_label"].map(label_map).astype(int).values
    groups = df_comb["patient_base"].astype(str).values

    y_stage1 = (df_comb["func3_label"] != "immotile").astype(int).values

    numeric_cols = df_comb.select_dtypes(include=[np.number]).columns.tolist()
    leakage_keywords = ["pred", "prob", "label", "fold", "stage"]
    numeric_cols = [c for c in numeric_cols if not any(k in c.lower() for k in leakage_keywords)]

    X = df_comb[numeric_cols].fillna(0).values
    return X, y4, y_stage1, groups, numeric_cols, classes4

def _smote_safe(X, y, seed):
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        return X, y
    k_neighbors = max(1, min(5, min_count - 1))
    sm = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    return sm.fit_resample(X, y)


# =============================================================
# 5) PLOTS
# =============================================================

def _init_plot_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
    })
    if _HAS_SEABORN:
        sns.set_context("paper")

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", save_path=None, dpi=300, fmt="png"):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    if normalize:
        cm_show = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        title = title + " (Normalized)"
    else:
        cm_show = cm

    im = ax.imshow(cm_show, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_title(title, weight="bold", pad=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm_show.max() / 2.0 if cm_show.size else 0.5
    for i in range(cm_show.shape[0]):
        for j in range(cm_show.shape[1]):
            if normalize:
                text = f"{cm_show[i,j]:.2f}\n({cm[i,j]})"
            else:
                text = f"{cm[i,j]}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if cm_show[i,j] > thresh else "black",
                    fontsize=11, weight="bold")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return save_path

def plot_roc_curves_multiclass(y_true, y_probs, classes, save_path=None, dpi=300, fmt="png"):
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(8.5, 7))
    colors = cycle(["C0", "C1", "C2", "C3", "C4"])

    for i, color in zip(range(n_classes), colors):
        ax.plot(fpr[i], tpr[i], lw=2.2, color=color, label=f"{classes[i]} (AUC={roc_auc[i]:.3f})")

    ax.plot(fpr["micro"], tpr["micro"], linestyle=":", linewidth=2.6, label=f"micro (AUC={roc_auc['micro']:.3f})")
    ax.plot(fpr["macro"], tpr["macro"], linestyle=":", linewidth=2.6, label=f"macro (AUC={roc_auc['macro']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.8, label="random")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (OvR)", weight="bold", pad=12)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return save_path, roc_auc

def plot_per_class_metrics(cm, classes, save_path=None, dpi=300, fmt="png"):
    n = len(classes)
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i]    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i]        = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, precision, w, label="Precision")
    ax.bar(x,     recall,    w, label="Recall")
    ax.bar(x + w, f1,        w, label="F1")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics", weight="bold", pad=10)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return save_path

def plot_feature_importance_comparison(imp1, imp2, feature_names, top_n=20, save_path=None, dpi=300, fmt="png"):
    top_n = int(top_n)
    idx1 = np.argsort(imp1)[-top_n:]
    idx2 = np.argsort(imp2)[-top_n:]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
    ax1, ax2 = axes

    ax1.barh(range(top_n), imp1[idx1])
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in idx1], fontsize=9)
    ax1.set_title("Stage-1 Top Features", weight="bold")
    ax1.set_xlabel("Importance")

    ax2.barh(range(top_n), imp2[idx2])
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([feature_names[i] for i in idx2], fontsize=9)
    ax2.set_title("Stage-2 Top Features", weight="bold")
    ax2.set_xlabel("Importance")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return save_path

def plot_cv_stability(results, save_path=None, dpi=300, fmt="png"):
    seeds = [r["seed"] for r in results]
    acc_means = [r["acc_mean"] for r in results]
    acc_stds  = [r["acc_std"] for r in results]
    bacc_means = [r["bacc_mean"] for r in results]
    bacc_stds  = [r["bacc_std"] for r in results]

    x = np.arange(len(seeds))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax1.errorbar(x, acc_means, yerr=acc_stds, fmt="o-", capsize=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylim([0, 1])
    ax1.set_title("Accuracy Across Seeds", weight="bold")
    ax1.set_ylabel("ACC")

    ax2.errorbar(x, bacc_means, yerr=bacc_stds, fmt="s-", capsize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylim([0, 1])
    ax2.set_title("Balanced Accuracy Across Seeds", weight="bold")
    ax2.set_ylabel("BACC")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    return save_path


# =============================================================
# 6) HIERARCHICAL RF (single seed CV)
# =============================================================

def run_two_stage_hierarchical_rf_single_seed(df_comb, seed, n_splits, rf_params):
    X_all, y4, y_stage1, groups, feature_names, classes4 = prepare_xy(df_comb)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    acc_list, bacc_list = [], []
    all_true, all_pred, all_probs = [], [], []
    stage1_importances, stage2_importances = [], []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_all, y4, groups), start=1):
        Xtr, Xte = X_all[tr_idx], X_all[te_idx]
        y4_tr, y4_te = y4[tr_idx], y4[te_idx]

        scaler = StandardScaler()
        Xtr_sc = scaler.fit_transform(Xtr)
        Xte_sc = scaler.transform(Xte)

        # ---- Stage-1 ----
        y1_tr = (y4_tr != 0).astype(int)
        X1_tr_sm, y1_tr_sm = _smote_safe(Xtr_sc, y1_tr, seed)

        n_total1 = len(y1_tr_sm)
        class_weights1 = {}
        for c in [0, 1]:
            cnt = (y1_tr_sm == c).sum()
            class_weights1[c] = (n_total1 / (2 * cnt)) if cnt > 0 else 0.0

        rf1 = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            class_weight=class_weights1,
            n_jobs=-1,
            random_state=seed + fold * 10 + 1,
        )
        rf1.fit(X1_tr_sm, y1_tr_sm)
        stage1_importances.append(rf1.feature_importances_)

        # ---- Stage-2 ----
        mot_tr_mask = (y4_tr != 0)
        X2_tr = Xtr_sc[mot_tr_mask]
        y2_tr_full = y4_tr[mot_tr_mask]  # 1,2,3
        y2_tr = y2_tr_full - 1

        if len(np.unique(y2_tr)) > 1:
            X2_tr_sm, y2_tr_sm = _smote_safe(X2_tr, y2_tr, seed)
        else:
            X2_tr_sm, y2_tr_sm = X2_tr, y2_tr

        n_total2 = len(y2_tr_sm)
        uniq2 = np.unique(y2_tr_sm)
        n_classes2 = len(uniq2)
        class_weights2 = {}
        for c in uniq2:
            cnt = (y2_tr_sm == c).sum()
            class_weights2[int(c)] = (n_total2 / (n_classes2 * cnt)) if cnt > 0 else 0.0

        rf2 = RandomForestClassifier(
            n_estimators=rf_params["n_estimators"],
            max_depth=rf_params["max_depth"],
            min_samples_leaf=rf_params["min_samples_leaf"],
            class_weight=class_weights2,
            n_jobs=-1,
            random_state=seed + fold * 10 + 2,
        )
        rf2.fit(X2_tr_sm, y2_tr_sm)
        stage2_importances.append(rf2.feature_importances_)

        # ---- Hierarchical prediction ----
        y1_pred = rf1.predict(Xte_sc)
        y1_proba = rf1.predict_proba(Xte_sc)

        y4_pred = np.zeros_like(y4_te)
        probs = np.zeros((len(y4_te), 4), dtype=float)

        imm_mask = (y1_pred == 0)
        mot_mask = (y1_pred == 1)

        y4_pred[imm_mask] = 0
        probs[imm_mask, 0] = y1_proba[imm_mask, 0]

        if mot_mask.any():
            X2_te = Xte_sc[mot_mask]
            y2_pred = rf2.predict(X2_te)    # 0,1,2
            y4_pred[mot_mask] = y2_pred + 1

            p2 = rf2.predict_proba(X2_te)   # (n_mot,3)
            p1_mot = y1_proba[mot_mask, 1]  # (n_mot,)
            probs[mot_mask, 0] = y1_proba[mot_mask, 0]
            probs[mot_mask, 1:] = p1_mot[:, None] * p2

        rs = probs.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        probs = probs / rs

        acc  = accuracy_score(y4_te, y4_pred)
        bacc = balanced_accuracy_score(y4_te, y4_pred)

        acc_list.append(acc)
        bacc_list.append(bacc)
        all_true.append(y4_te)
        all_pred.append(y4_pred)
        all_probs.append(probs)

    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_probs = np.vstack(all_probs)

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(classes4))))
    roc_macro = roc_auc_score(all_true, all_probs, multi_class="ovr", average="macro")
    roc_micro = roc_auc_score(all_true, all_probs, multi_class="ovr", average="micro")

    return {
        "seed": seed,
        "acc_mean": float(np.mean(acc_list)),
        "acc_std": float(np.std(acc_list)),
        "bacc_mean": float(np.mean(bacc_list)),
        "bacc_std": float(np.std(bacc_list)),
        "cm": cm,
        "y_true": all_true,
        "y_pred": all_pred,
        "y_probs": all_probs,
        "classes": classes4,
        "feature_names": feature_names,
        "stage1_importances": np.mean(stage1_importances, axis=0),
        "stage2_importances": np.mean(stage2_importances, axis=0),
        "roc_macro": float(roc_macro),
        "roc_micro": float(roc_micro),
        "report": classification_report(all_true, all_pred, target_names=classes4, zero_division=0),
    }


# =============================================================
# 7) SHAP Stage-2 (optional)
# =============================================================

def run_shap_stage2(df_comb, best_seed, output_dir, fig_dpi, fig_fmt):
    if not _HAS_SHAP:
        print("⚠️ SHAP not installed. Skipping SHAP.")
        return None

    X, y4, y_stage1, groups, feature_names, classes4 = prepare_xy(df_comb)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    mot_idx = np.where(y_stage1 == 1)[0]
    X_mot = X_sc[mot_idx]
    y2_full = y4[mot_idx]        # 1,2,3
    y2 = y2_full - 1             # 0,1,2

    if len(np.unique(y2)) > 1:
        X_res, y_res = _smote_safe(X_mot, y2, best_seed)
    else:
        X_res, y_res = X_mot, y2

    uniq = np.unique(y_res)
    n_total = len(y_res)
    n_classes = len(uniq)
    class_weights = {}
    for c in uniq:
        cnt = (y_res == c).sum()
        class_weights[int(c)] = (n_total / (n_classes * cnt)) if cnt > 0 else 0.0

    rf2 = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        class_weight=class_weights,
        n_jobs=-1,
        random_state=best_seed,
    )
    rf2.fit(X_res, y_res)

    explainer = shap.TreeExplainer(rf2)
    X_sample = X_res[: min(1000, X_res.shape[0])]
    shap_values = explainer.shap_values(X_sample)

    # SHAP summary per class
    class_names = ["normal", "stiff", "circular"]
    fig = plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=15, show=False)
    out = os.path.join(output_dir, f"shap_summary_stage2.{fig_fmt}")
    plt.tight_layout()
    plt.savefig(out, dpi=fig_dpi, bbox_inches="tight", format=fig_fmt)
    plt.close()
    print(f"✅ Saved SHAP: {out}")
    return out


# =============================================================
# 8) MAIN
# =============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    _init_plot_style()

    # log file
    log_path = os.path.join(args.output_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    def _log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    _log("=" * 80)
    _log("HIERARCHICAL RF PUBLICATION PIPELINE")
    _log("=" * 80)
    _log(f"feature_bank_csv: {args.feature_bank_csv}")
    _log(f"normalized_dir  : {args.normalized_dir}")
    _log(f"output_dir      : {args.output_dir}")
    _log(f"seeds           : {args.seeds}")
    _log(f"n_splits        : {args.n_splits}")
    _log(f"fig             : {args.figure_format} @ {args.figure_dpi} dpi")
    _log(f"shap_enabled    : {(_HAS_SHAP and (not args.no_shap))}")
    _log("-" * 80)

    df_comb, motion_stats = load_engineered_plus_wave(args.feature_bank_csv, args.normalized_dir)

    _log(f"Loaded df_comb shape: {df_comb.shape}")
    _log(f"Unique patients: {df_comb['patient_base'].nunique()}")
    _log("Class distribution:")
    _log(str(df_comb["func3_label"].value_counts()))
    _log(f"Motion index size: {motion_stats['motion_index_size']}")
    _log(f"Motion found/notfound: {motion_stats['motion_found']}/{motion_stats['motion_not_found']}")
    _log("-" * 80)

    rf_params = {
        "n_estimators": args.n_estimators,
        "min_samples_leaf": args.min_samples_leaf,
        "max_depth": args.max_depth,
    }

    results = []
    for s in args.seeds:
        _log(f"\n>>> Running seed={s}")
        res = run_two_stage_hierarchical_rf_single_seed(df_comb, seed=s, n_splits=args.n_splits, rf_params=rf_params)
        results.append(res)
        _log(f"Seed {s} | ACC={res['acc_mean']:.3f}±{res['acc_std']:.3f} | BACC={res['bacc_mean']:.3f}±{res['bacc_std']:.3f} "
             f"| AUCmacro={res['roc_macro']:.3f} | AUCmicro={res['roc_micro']:.3f}")

    best = max(results, key=lambda x: x["bacc_mean"])
    _log("\n" + "=" * 80)
    _log(f"BEST SEED = {best['seed']}  (BACC={best['bacc_mean']:.3f}, ACC={best['acc_mean']:.3f})")
    _log("=" * 80)
    _log("\nClassification report (best seed, pooled CV preds):")
    _log(best["report"])

    # Save figures for best seed
    cm_raw_path = os.path.join(args.output_dir, f"confusion_matrix_best.{args.figure_format}")
    cm_norm_path = os.path.join(args.output_dir, f"confusion_matrix_best_normalized.{args.figure_format}")
    roc_path = os.path.join(args.output_dir, f"roc_curves_best.{args.figure_format}")
    metrics_path = os.path.join(args.output_dir, f"per_class_metrics_best.{args.figure_format}")
    imp_path = os.path.join(args.output_dir, f"feature_importance_best.{args.figure_format}")
    stab_path = os.path.join(args.output_dir, f"cv_stability.{args.figure_format}")

    plot_confusion_matrix(best["cm"], best["classes"], normalize=False,
                          title="Confusion Matrix (Best Seed)",
                          save_path=cm_raw_path, dpi=args.figure_dpi, fmt=args.figure_format)
    plot_confusion_matrix(best["cm"], best["classes"], normalize=True,
                          title="Confusion Matrix (Best Seed)",
                          save_path=cm_norm_path, dpi=args.figure_dpi, fmt=args.figure_format)
    plot_roc_curves_multiclass(best["y_true"], best["y_probs"], best["classes"],
                               save_path=roc_path, dpi=args.figure_dpi, fmt=args.figure_format)
    plot_per_class_metrics(best["cm"], best["classes"],
                           save_path=metrics_path, dpi=args.figure_dpi, fmt=args.figure_format)
    plot_feature_importance_comparison(best["stage1_importances"], best["stage2_importances"],
                                       best["feature_names"],
                                       top_n=args.top_n_importance,
                                       save_path=imp_path, dpi=args.figure_dpi, fmt=args.figure_format)
    plot_cv_stability(results, save_path=stab_path, dpi=args.figure_dpi, fmt=args.figure_format)

    _log("\nSaved figures:")
    _log(f" - {cm_raw_path}")
    _log(f" - {cm_norm_path}")
    _log(f" - {roc_path}")
    _log(f" - {metrics_path}")
    _log(f" - {imp_path}")
    _log(f" - {stab_path}")

    # SHAP optional
    if _HAS_SHAP and (not args.no_shap):
        run_shap_stage2(df_comb, best["seed"], args.output_dir, args.figure_dpi, args.figure_format)

    _log("\n✅ DONE. All outputs in: " + args.output_dir)
    _log("Log saved to: " + log_path)


if __name__ == "__main__":
    main()
