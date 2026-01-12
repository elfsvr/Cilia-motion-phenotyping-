#!/usr/bin/env python3



from __future__ import annotations
import os, re, glob, json, argparse, unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks, savgol_filter
from scipy import stats
from scipy.fft import fft


# =========================
# Key normalization
# =========================
STRIP_SUFFIXES = [
    "_enhanced_analysis", "_motion_stats", "_bbox_motion_stats",
    "_interpolated", "_analysis", "_stats", "_mp4", "_csv"
]

def norm_key(s: str) -> str:
    """Normalize identifiers / filenames for reliable matching."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = s.replace("\\", "/")
    s = os.path.basename(s)
    s = re.sub(r"\.csv$", "", s)

    # remove known suffixes (end only)
    for suf in STRIP_SUFFIXES:
        s = re.sub(f"{re.escape(suf)}$", "", s)

    # some files include prefixes like frames_
    s = re.sub(r"^frames_+", "", s)

    s = s.replace("-", "_")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# =========================
# Robust IO
# =========================
def robust_read_csv(fp: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(fp)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def clean_numeric(x: np.ndarray | None) -> np.ndarray | None:
    """float + interpolate NaNs (gap filling for time-series)."""
    if x is None:
        return None
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    if x.size == 0:
        return None
    return pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=float)


def pad_or_trim(x: np.ndarray | None, target_len: int) -> np.ndarray | None:
    if x is None:
        return None
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    if x.size >= target_len:
        return x[:target_len]
    return np.pad(x, (0, target_len - x.size), mode="edge")


# =========================
# Column getters (MMO-aware)
# =========================
def get_series(df: pd.DataFrame, names: list[str]) -> np.ndarray | None:
    # direct
    for n in names:
        if n in df.columns:
            return df[n].to_numpy()

    alt_map = {
        "mmo": ["mmo", "MMO", "mean_mmo", "mask_mmo", "motion_mmo", "mean_motion_overlap", "motion_overlap"],
        "iou": ["iou", "IoU", "mask_iou", "Mask_IoU", "iou_score", "mean_iou", "avg_iou"],
        "dominant_angle_deg": ["dominant_angle_deg", "dom_angle_deg", "dominant_angle", "mode_angle_deg"],
        "mean_angle_deg": ["mean_angle_deg", "angle_deg", "mean_angle", "angle"],
        "mean_magnitude": ["mean_magnitude", "magnitude", "mag", "meanMag", "mean_mag", "motion"],
        "fps": ["fps", "base_fps", "effective_fps", "FPS", "frame_rate"],
    }
    for n in names:
        if n in alt_map:
            for a in alt_map[n]:
                if a in df.columns:
                    return df[a].to_numpy()
    return None


# =========================
# FPS selection (NO fallback)
# =========================
def pick_fps(df: pd.DataFrame, basename: str, fps_map: dict | None) -> float | None:
    # 1) from CSV column
    fps_series = get_series(df, ["fps"])
    if fps_series is not None:
        fpv = float(np.nanmedian(pd.to_numeric(fps_series, errors="coerce")))
        if fpv > 0:
            return fpv

    # 2) from mapping JSON (BASENAME key)
    if fps_map:
        key = os.path.basename(basename)
        if key in fps_map:
            v = fps_map[key]
            if isinstance(v, dict):
                fps_val = v.get("fps", None)
            else:
                fps_val = v
            if fps_val is not None and float(fps_val) > 0:
                return float(fps_val)

    # 3) missing => None (important!)
    return None


# =========================
# CBF estimation (leakage-free)
# =========================
def _acf_method(x: np.ndarray, fps: float, fmin: float, fmax: float):
    x = x - np.nanmean(x)
    if len(x) >= 9:
        x = savgol_filter(x, window_length=9, polyorder=2, mode="interp")

    acf = np.correlate(x, x, mode="full")[len(x) - 1 :]
    if not np.isfinite(acf[0]) or acf[0] == 0:
        return (np.nan, 0.0)
    acf = acf / (acf[0] + 1e-12)

    min_lag = max(int(np.floor(fps / fmax)), 2)
    max_lag = int(np.ceil(fps / fmin))
    if max_lag <= min_lag + 2 or min_lag >= len(acf) - 1:
        return (np.nan, 0.0)

    seg = acf[min_lag : min(max_lag, len(acf))]
    if len(seg) < 3:
        return (np.nan, 0.0)

    peaks, props = find_peaks(seg, prominence=0.05)
    if len(peaks) == 0:
        return (np.nan, 0.0)

    idx = int(np.argmax(props["prominences"]))
    p = int(peaks[idx])
    lag = min_lag + p
    period_s = lag / float(fps)
    if period_s <= 0:
        return (np.nan, 0.0)

    cbf = 1.0 / period_s
    prom = float(props["prominences"][idx])
    height = float(seg[p])
    qual = float(np.clip(0.5 * prom + 0.5 * height, 0, 1))
    return (float(cbf), float(qual))


def _welch_method(x: np.ndarray, fps: float, fmin: float, fmax: float):
    x = x - np.nanmean(x)
    if len(x) >= 9:
        x = savgol_filter(x, window_length=9, polyorder=2, mode="interp")

    nper = min(len(x), 64)
    if nper < 16:
        return (np.nan, 0.0)

    f, pxx = welch(x, fs=fps, nperseg=nper, noverlap=int(0.5 * nper))
    band = (f >= fmin) & (f <= fmax)
    if not np.any(band):
        return (np.nan, 0.0)

    fb, pb = f[band], pxx[band]
    if np.allclose(pb.max(), 0):
        return (np.nan, 0.0)

    fi = float(fb[np.argmax(pb)])
    snr_like = float((pb.max() + 1e-12) / (pb.mean() + 1e-12))
    qual = float(np.clip((snr_like - 1) / 4, 0, 1))
    return (fi, qual)


def estimate_cbf_from_signal(x: np.ndarray | None, fps: float | None, fmin: float, fmax: float, min_len: int = 24):
    if x is None or fps is None or not (fps > 0) or len(x) < min_len:
        return (np.nan, 0.0, "insufficient")

    x = clean_numeric(x)
    if x is None or len(x) < min_len:
        return (np.nan, 0.0, "insufficient")

    cbf_acf, q_acf = _acf_method(x.copy(), float(fps), fmin, fmax)
    cbf_wel, q_wel = _welch_method(x.copy(), float(fps), fmin, fmax)

    if np.isnan(cbf_acf) and np.isnan(cbf_wel):
        return (np.nan, 0.0, "no_peak")

    if (q_acf >= q_wel) and not np.isnan(cbf_acf):
        return (float(cbf_acf), float(q_acf), "acf")
    if not np.isnan(cbf_wel):
        return (float(cbf_wel), float(q_wel), "welch")
    return (np.nan, 0.0, "no_peak")


def choose_best_cbf(df: pd.DataFrame, fps: float | None, fmin: float, fmax: float):
    # magnitude
    mag = get_series(df, ["mean_magnitude"])
    cbf_mag, q_mag, m_mag = estimate_cbf_from_signal(mag, fps, fmin, fmax)

    # angle (unwrap)
    ang = get_series(df, ["dominant_angle_deg", "mean_angle_deg"])
    if ang is not None:
        ang = clean_numeric(ang)
        if ang is not None:
            a = ((ang + 180) % 360) - 180
            ang = np.rad2deg(np.unwrap(np.deg2rad(a)))
    cbf_ang, q_ang, m_ang = estimate_cbf_from_signal(ang, fps, fmin, fmax)

    # MMO preferred, IOU fallback
    mmo = get_series(df, ["mmo"])
    if mmo is None:
        mmo = get_series(df, ["iou"])
    if mmo is not None:
        mmo = clean_numeric(mmo)
        if mmo is not None:
            mmo = np.clip(mmo, 0, 1)
    cbf_mmo, q_mmo, m_mmo = estimate_cbf_from_signal(mmo, fps, fmin, fmax)

    cands = [
        ("magnitude", cbf_mag, q_mag, m_mag),
        ("angle", cbf_ang, q_ang, m_ang),
        ("mmo", cbf_mmo, q_mmo, m_mmo),
    ]

    def score(t):
        _, cbf, q, _ = t
        return q if np.isfinite(cbf) else -1.0

    source, cbf, qual, method = max(cands, key=score)
    best = {
        "cbf_hz": float(cbf) if np.isfinite(cbf) else np.nan,
        "cbf_quality": float(qual) if np.isfinite(cbf) else 0.0,
        "cbf_method": f"{source}:{method}" if np.isfinite(cbf) else f"{source}:no_peak",
        "cbf_source": source if np.isfinite(cbf) else "none",
    }
    parts = {
        "magnitude": (cbf_mag, q_mag, m_mag),
        "angle": (cbf_ang, q_ang, m_ang),
        "mmo": (cbf_mmo, q_mmo, m_mmo),
    }
    return best, parts


# =========================
# Wave feature extraction (compact)
# =========================
WAVE_COLS = [
    "wave_mag_mean", "wave_mag_std", "wave_fft_peak",
    "wave_entropy", "wave_repeatability",
    "wave_angle_change_mean", "wave_score"
]

def _get_series_by_candidates(df: pd.DataFrame, candidates: list[str]) -> np.ndarray | None:
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    return None


def extract_compact_wave_features_for_one(sequence_df: pd.DataFrame, max_len: int = 30) -> dict[str, float]:
    if sequence_df is None or len(sequence_df) < 3:
        return {k: 0.0 for k in WAVE_COLS}

    seq = sequence_df.iloc[:max_len].copy()
    mag = _get_series_by_candidates(seq, ["mean_magnitude", "motion", "mag", "magnitude"])
    if mag is None:
        return {k: 0.0 for k in WAVE_COLS}

    mag = pd.Series(mag).interpolate(limit_direction="both").to_numpy(dtype=float)
    wave_mag_mean = float(np.mean(mag))
    wave_mag_std  = float(np.std(mag))

    if len(mag) >= 8:
        mag_fft = np.abs(fft(mag))[: len(mag)//2]
        wave_fft_peak   = float(np.argmax(mag_fft[1:]) + 1) if len(mag_fft) > 1 else 0.0
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
        ang = pd.Series(ang).interpolate(limit_direction="both").to_numpy(dtype=float)
        wave_angle_change_mean = float(np.mean(np.abs(np.diff(ang))))
    else:
        wave_angle_change_mean = 0.0

    def _safe_norm(x: float) -> float:
        if not np.isfinite(x) or abs(x) < 1e-12:
            return 0.0
        return float(np.clip(np.log1p(abs(x)) / 10.0, 0, 1))

    score_parts = [
        _safe_norm(wave_mag_mean),
        _safe_norm(wave_mag_std),
        _safe_norm(wave_fft_energy),
        float(np.clip(wave_repeatability, 0, 1)),
        _safe_norm(wave_angle_change_mean),
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


def build_motion_file_index(normalized_dir: str) -> dict[str, str]:
    """Index normalized2 motion CSVs: key -> best file path."""
    all_files = []
    all_files += glob.glob(os.path.join(normalized_dir, "*.csv"))
    all_files += glob.glob(os.path.join(normalized_dir, "**/*.csv"), recursive=True)
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

    idx: dict[str, tuple[str, int]] = {}
    for p in all_files:
        base = os.path.basename(p)
        key = norm_key(base)
        sc = score_name(base)
        if not key:
            continue
        if key not in idx or sc > idx[key][1]:
            idx[key] = (p, sc)

    return {k: v[0] for k, v in idx.items()}


# =========================
# Velocity fill (from velocity CSV)
# =========================
def fill_velocity_features(
    base: pd.DataFrame,
    base_key_col: str,
    velocity_csv: str,
    velocity_key_col: str,
    velocity_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge velocity table by key and fill ONLY where base has 0 or NaN.
    Returns (updated_base, qc_df).
    """
    vel = pd.read_csv(velocity_csv)
    qc_rows = []

    if velocity_key_col not in vel.columns:
        raise ValueError(f"velocity_key_col '{velocity_key_col}' not found in velocity CSV.")

    # Ensure cols exist
    missing_cols = [c for c in velocity_cols if c not in vel.columns]
    if missing_cols:
        raise ValueError(f"Missing velocity columns in velocity CSV: {missing_cols}")

    base = base.copy()
    base["__key__"] = base[base_key_col].astype(str).apply(norm_key)
    vel["__key__"] = vel[velocity_key_col].astype(str).apply(norm_key)

    vel_sub = vel[["__key__"] + velocity_cols].copy()
    merged = base.merge(vel_sub, on="__key__", how="left", suffixes=("", "_new"))

    filled_cells = 0
    for col in velocity_cols:
        new_col = f"{col}_new"
        if col not in merged.columns:
            merged[col] = np.nan
        mask = merged[col].isna() | (pd.to_numeric(merged[col], errors="coerce").fillna(0) == 0)
        before = mask.sum()
        merged.loc[mask, col] = merged.loc[mask, new_col]
        after = (merged[col].isna() | (pd.to_numeric(merged[col], errors="coerce").fillna(0) == 0)).sum()
        filled_cells += int(before - after)

    # QC
    qc_rows.append({
        "step": "velocity_fill",
        "rows_base": len(base),
        "rows_velocity": len(vel),
        "filled_cells": filled_cells
    })

    # cleanup
    drop_new = [c for c in merged.columns if c.endswith("_new")]
    merged.drop(columns=drop_new, inplace=True, errors="ignore")
    merged.drop(columns=["__key__"], inplace=True, errors="ignore")

    return merged, pd.DataFrame(qc_rows)


# =========================
# Wave fill (from normalized2)
# =========================
def fill_wave_features_from_normalized2(
    base: pd.DataFrame,
    base_key_col: str,
    normalized2_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = base.copy()
    for c in WAVE_COLS:
        if c not in base.columns:
            base[c] = 0.0

    motion_index = build_motion_file_index(normalized2_dir)

    base["__key__"] = base[base_key_col].astype(str).apply(norm_key)

    qc_rows = []
    filled_cells = 0
    matched = 0
    not_found = 0
    read_errors = 0

    for i in range(len(base)):
        key = base.at[i, "__key__"]
        if key in motion_index:
            matched += 1
            p = motion_index[key]
            try:
                mdf = pd.read_csv(p)
                wf = extract_compact_wave_features_for_one(mdf)
                status = "matched"
                err = ""
            except Exception as e:
                wf = {k: 0.0 for k in WAVE_COLS}
                status = "read_error"
                err = str(e)[:200]
                read_errors += 1

            # fill only where 0/NaN
            for c in WAVE_COLS:
                cur = base.at[i, c]
                cur_val = pd.to_numeric(pd.Series([cur]), errors="coerce").iloc[0]
                if pd.isna(cur_val) or float(cur_val) == 0.0:
                    if float(wf.get(c, 0.0)) != 0.0:
                        base.at[i, c] = float(wf[c])
                        filled_cells += 1

            qc_rows.append({
                "filename": base.at[i, base_key_col],
                "key": key,
                "matched_path": p,
                "status": status,
                "error": err
            })
        else:
            not_found += 1
            qc_rows.append({
                "filename": base.at[i, base_key_col],
                "key": key,
                "matched_path": "",
                "status": "not_found",
                "error": ""
            })

    # summary qc row
    qc_rows.append({
        "filename": "",
        "key": "",
        "matched_path": "",
        "status": "SUMMARY",
        "error": f"matched={matched} not_found={not_found} read_errors={read_errors} filled_cells={filled_cells}"
    })

    base.drop(columns=["__key__"], inplace=True, errors="ignore")
    return base, pd.DataFrame(qc_rows)


# =========================
# CBF compute + merge (from newnormalized)
# =========================
def compute_cbf_table_from_newnormalized(
    newnormalized_dir: str,
    fps_map: dict | None,
    cbf_min: float,
    cbf_max: float,
    quality_thr: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_csvs = sorted(set(glob.glob(os.path.join(newnormalized_dir, "*.csv"))))
    if not all_csvs:
        raise ValueError(f"No CSVs found in newnormalized_dir: {newnormalized_dir}")

    rows = []
    qc_rows = []

    for fp in all_csvs:
        base_name = os.path.basename(fp)
        df = robust_read_csv(fp)
        if df is None:
            continue

        fps = pick_fps(df, base_name, fps_map)

        if fps is None:
            best = {"cbf_hz": np.nan, "cbf_quality": 0.0, "cbf_method": "fps_missing", "cbf_source": "none"}
            parts = {
                "magnitude": (np.nan, 0.0, "fps_missing"),
                "angle": (np.nan, 0.0, "fps_missing"),
                "mmo": (np.nan, 0.0, "fps_missing"),
            }
        else:
            best, parts = choose_best_cbf(df, fps, cbf_min, cbf_max)

        merge_key = norm_key(base_name)

        rows.append({
            "__key__": merge_key,
            "cbf_motion_csv_basename": base_name,
            "cbf_fps": float(fps) if fps is not None else np.nan,
            "cbf_hz": best["cbf_hz"],
            "cbf_quality": best["cbf_quality"],
            "cbf_method": best["cbf_method"],
            "cbf_source": best.get("cbf_source", "none"),
            "cbf_hz_filtered": best["cbf_hz"] if (np.isfinite(best["cbf_hz"]) and best["cbf_quality"] >= quality_thr) else np.nan,
        })

        (cbf_mag, q_mag, m_mag) = parts["magnitude"]
        (cbf_ang, q_ang, m_ang) = parts["angle"]
        (cbf_mmo, q_mmo, m_mmo) = parts["mmo"]
        qc_rows.append({
            "__key__": merge_key,
            "motion_csv_basename": base_name,
            "fps": float(fps) if fps is not None else np.nan,
            "best_cbf_hz": best["cbf_hz"],
            "best_quality": best["cbf_quality"],
            "best_method": best["cbf_method"],
            "mag_cbf": cbf_mag, "mag_q": q_mag, "mag_method": m_mag,
            "ang_cbf": cbf_ang, "ang_q": q_ang, "ang_method": m_ang,
            "mmo_cbf": cbf_mmo, "mmo_q": q_mmo, "mmo_method": m_mmo,
        })

    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def merge_cbf_into_base(base: pd.DataFrame, base_key_col: str, cbf_df: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["__key__"] = base[base_key_col].astype(str).apply(norm_key)
    out = base.merge(cbf_df, on="__key__", how="left")
    out.drop(columns=["__key__"], inplace=True, errors="ignore")
    return out


# =========================
# CLI / main
# =========================
def infer_key_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "filename", "motion_csv", "motion_file", "analysis_file",
        "video", "video_name", "path", "file"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_feature_bank", required=True, help="Base feature bank CSV path")
    ap.add_argument("--base_key_col", default=None, help="Column used as filename key in base feature bank (default: infer)")
    ap.add_argument("--out_dir", default=".", help="Output directory")
    ap.add_argument("--out_prefix", default="FEATURE_BANK_MASTER", help="Output prefix")

    # velocity fill (optional)
    ap.add_argument("--velocity_csv", default=None, help="Optional velocity CSV (to fill velocity features)")
    ap.add_argument("--velocity_key_col", default=None, help="Key column in velocity CSV (default: infer)")
    ap.add_argument("--velocity_cols", nargs="*", default=None, help="Velocity columns to fill (space-separated)")

    # wave from normalized2
    ap.add_argument("--normalized2_dir", default=None, help="normalized2 dir (for wave features). If None, wave step skipped.")

    # CBF from newnormalized
    ap.add_argument("--newnormalized_dir", default=None, help="newnormalized dir (for CBF). If None, CBF step skipped.")
    ap.add_argument("--fps_mapping", default=None, help="FPS mapping JSON (basename keys)")
    ap.add_argument("--cbf_min", type=float, default=2.0)
    ap.add_argument("--cbf_max", type=float, default=15.0)
    ap.add_argument("--cbf_quality_thr", type=float, default=0.30)

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base = pd.read_csv(args.base_feature_bank)
    if base.empty:
        raise SystemExit("Base feature bank is empty.")

    base_key_col = args.base_key_col or infer_key_column(base)
    if base_key_col is None:
        raise SystemExit("Could not infer base_key_col. Please pass --base_key_col.")
    if base_key_col not in base.columns:
        raise SystemExit(f"base_key_col '{base_key_col}' not found in base feature bank.")

    qc_velocity = pd.DataFrame()
    qc_wave = pd.DataFrame()
    qc_cbf = pd.DataFrame()

    # (B) velocity fill
    if args.velocity_csv:
        if not os.path.exists(args.velocity_csv):
            raise SystemExit(f"velocity_csv not found: {args.velocity_csv}")
        vel_df = pd.read_csv(args.velocity_csv)
        velocity_key_col = args.velocity_key_col or infer_key_column(vel_df)
        if velocity_key_col is None:
            raise SystemExit("Could not infer velocity_key_col. Please pass --velocity_key_col.")
        if args.velocity_cols is None or len(args.velocity_cols) == 0:
            raise SystemExit("You provided --velocity_csv but no --velocity_cols. Provide the list of velocity columns to fill.")
        base, qc_velocity = fill_velocity_features(
            base=base,
            base_key_col=base_key_col,
            velocity_csv=args.velocity_csv,
            velocity_key_col=velocity_key_col,
            velocity_cols=args.velocity_cols,
        )

    # (C) wave fill from normalized2
    if args.normalized2_dir:
        if not os.path.isdir(args.normalized2_dir):
            raise SystemExit(f"normalized2_dir not found: {args.normalized2_dir}")
        base, qc_wave = fill_wave_features_from_normalized2(
            base=base,
            base_key_col=base_key_col,
            normalized2_dir=args.normalized2_dir,
        )

    # (D) CBF compute + merge
    if args.newnormalized_dir:
        if not os.path.isdir(args.newnormalized_dir):
            raise SystemExit(f"newnormalized_dir not found: {args.newnormalized_dir}")

        fps_map = None
        if args.fps_mapping:
            if not os.path.exists(args.fps_mapping):
                raise SystemExit(f"fps_mapping not found: {args.fps_mapping}")
            with open(args.fps_mapping, "r", encoding="utf-8") as f:
                fps_map = json.load(f)

        cbf_df, qc_cbf = compute_cbf_table_from_newnormalized(
            newnormalized_dir=args.newnormalized_dir,
            fps_map=fps_map,
            cbf_min=args.cbf_min,
            cbf_max=args.cbf_max,
            quality_thr=args.cbf_quality_thr,
        )
        base = merge_cbf_into_base(base, base_key_col=base_key_col, cbf_df=cbf_df)

    # Save final
    out_final = os.path.join(args.out_dir, f"{args.out_prefix}_{stamp}.csv")
    base.to_csv(out_final, index=False)

    # Save QC
    if not qc_velocity.empty:
        qc_velocity.to_csv(os.path.join(args.out_dir, f"{args.out_prefix}_{stamp}_QC_velocity.csv"), index=False)
    if not qc_wave.empty:
        qc_wave.to_csv(os.path.join(args.out_dir, f"{args.out_prefix}_{stamp}_QC_wave.csv"), index=False)
    if not qc_cbf.empty:
        qc_cbf.to_csv(os.path.join(args.out_dir, f"{args.out_prefix}_{stamp}_QC_cbf.csv"), index=False)

    print("\n✅ SAVED FINAL:", out_final)
    print("base_key_col:", base_key_col)
    print("rows:", len(base))

    # quick stats
    if "cbf_hz_filtered" in base.columns:
        ok = base["cbf_hz_filtered"].notna()
        print(f"CBF usable (quality≥{args.cbf_quality_thr}): {int(ok.sum())} ({100*ok.mean():.1f}%)")
    if all(c in base.columns for c in WAVE_COLS):
        all_zero = (base[WAVE_COLS].fillna(0).sum(axis=1) == 0).sum()
        print(f"Rows with ALL wave=0: {int(all_zero)} ({100*all_zero/len(base):.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
