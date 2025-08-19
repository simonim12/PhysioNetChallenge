#!/usr/bin/env python3
"""
ECG -> features -> Gaussian Process Regression (GPR) detector for Chagas disease.
- Supports .dat int16 little-endian (no header) or ASCII numeric files.
- Extracts HRV stats, simple morphology, and bandpowers.
- Trains GPR (RBF kernel) on features with optional Platt calibration + F1 threshold.
"""
from __future__ import annotations

import argparse, os, json, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, average_precision_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import joblib
import glob


@dataclass
class Config:
    data_dir: Optional[str]
    labels: Optional[str]
    file: Optional[str]
    fmt: str
    fs: float
    pca: Optional[int]
    test_size: float
    random_state: int
    calibrate: bool
    outdir: str
    demo: bool
    load_model: Optional[str]
    scaler: Optional[str]
    calibrator: Optional[str]
    threshold_file: Optional[str]
    max_samples: Optional[int]  


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="ECG -> GPR (Chagas)")
    p.add_argument("--data-dir", type=str, default=None, help="Folder of .dat files.")
    p.add_argument("--labels", type=str, default=None, help="CSV with columns: file,label (label in {0,1}).")
    p.add_argument("--file", type=str, default=None, help="Single ECG file to featurize/evaluate.")
    p.add_argument("--format", type=str, default="int16le", choices=["int16le","text"], help="Input file format.")
    p.add_argument("--fs", type=float, default=360.0, help="Sampling rate (Hz).")
    p.add_argument("--pca", type=int, default=None, help="PCA components (optional).")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--calibrate", action="store_true", help="Enable Platt calibration + F1 threshold.")
    p.add_argument("--outdir", type=str, default="runs/ecg_gpr")
    p.add_argument("--demo", action="store_true", help="Run synthetic demo.")
    p.add_argument("--load-model", type=str, default=None, help="Path to a saved model_gpr.joblib.")
    p.add_argument("--scaler", type=str, default=None, help="Path to saved scaler.joblib.")
    p.add_argument("--calibrator", type=str, default=None, help="Path to saved calibrator.joblib.")
    p.add_argument("--threshold-file", type=str, default=None, help="Path to saved threshold.txt.")
    p.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to use (for testing).")
    a = p.parse_args()
    return Config(
        data_dir=a.data_dir, labels=a.labels, file=a.file, fmt=a.format, fs=a.fs, pca=a.pca,
        test_size=a.test_size, random_state=a.random_state, calibrate=bool(a.calibrate),
        outdir=a.outdir, demo=bool(a.demo), load_model=a.load_model, scaler=a.scaler,
        calibrator=a.calibrator, threshold_file=a.threshold_file, max_samples=a.max_samples
    )


def load_ecg(path: str, fmt: str) -> np.ndarray:
    if fmt == "int16le":
        with open(path, "rb") as f:
            raw = f.read()
        sig = np.frombuffer(raw, dtype="<i2").astype(np.float64)
        return sig
    else:  # text
        sig = np.loadtxt(path).astype(np.float64)
        return sig


def bandpass(x: np.ndarray, fs: float, lo=0.5, hi=45.0, order=4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, x)


def detect_r_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    # Simple energy-based approach on bandpassed signal
    xb = bandpass(x, fs)
    # Square to emphasize peaks
    s = xb**2
    # Min peak distance 0.25s; prominence heuristic
    distance = int(0.25 * fs)
    prom = max(0.05 * np.std(s), 1e-6)
    peaks, _ = find_peaks(s, distance=distance, prominence=prom)
    return peaks


def hrv_features(r_peaks: np.ndarray, fs: float) -> dict:
    if len(r_peaks) < 3:
        return {"hr_mean": np.nan, "hr_std": np.nan, "rr_mean": np.nan, "rr_std": np.nan, "rmssd": np.nan, "pnn50": np.nan}
    rr = np.diff(r_peaks) / fs  # seconds
    hr = 60.0 / rr
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr)))) if len(rr) > 1 else np.nan
    pnn50 = 100.0 * np.mean((np.abs(np.diff(rr)) > 0.05)) if len(rr) > 1 else np.nan
    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "rr_mean": float(np.mean(rr)),
        "rr_std": float(np.std(rr)),
        "rmssd": float(rmssd),
        "pnn50": float(pnn50),
    }


def bandpowers(x: np.ndarray, fs: float) -> dict:
    f, Pxx = welch(x, fs=fs, nperseg=min(len(x), 4*int(fs)))
    total = np.trapezoid(Pxx, f) + 1e-12
    def bp(lo, hi):
        mask = (f >= lo) & (f < hi)
        return float(np.trapezoid(Pxx[mask], f[mask]) / total)
    return {
        "bp_0_4": bp(0.0, 4.0),
        "bp_4_15": bp(4.0, 15.0),
        "bp_15_40": bp(15.0, 40.0),
    }


def morphology_features(x: np.ndarray, r_peaks: np.ndarray, fs: float) -> dict:
    if len(r_peaks) == 0:
        return {"r_amp_mean": np.nan, "r_amp_std": np.nan, "qrs_width_mean": np.nan}
    amps = x[r_peaks]
    # crude QRS width proxy: width at half max in a 0.2s window
    half_widths = []
    win = int(0.2 * fs)
    for p in r_peaks:
        lo = max(0, p - win//2); hi = min(len(x), p + win//2)
        seg = x[lo:hi]
        if len(seg) < 3: 
            continue
        peak = x[p]
        thr = peak * 0.5
        left = p
        while left > lo and x[left] > thr:
            left -= 1
        right = p
        while right < hi and x[right] > thr:
            right += 1
        half_widths.append((right - left) / fs)
    qrs_w = np.mean(half_widths) if half_widths else np.nan
    return {"r_amp_mean": float(np.mean(amps)), "r_amp_std": float(np.std(amps)), "qrs_width_mean": float(qrs_w)}


def extract_features(x: np.ndarray, fs: float) -> dict:
    # Normalize to zero-mean unit-std for morphology features robustness
    x = (x - np.median(x)) / (np.std(x) + 1e-9)
    peaks = detect_r_peaks(x, fs)
    feats = {}
    feats.update(hrv_features(peaks, fs))
    feats.update(bandpowers(x, fs))
    feats.update(morphology_features(x, peaks, fs))
    feats["n_peaks"] = int(len(peaks))
    feats["duration_s"] = float(len(x)/fs)
    return feats


def choose_f1_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    ts = np.linspace(0.05, 0.95, 19)
    scores = [f1_score(y_val, (p_val >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(scores))])


def load_label_map(labels_csv: str) -> dict:
    df = pd.read_csv(labels_csv)
    if not {"file","label"}.issubset(df.columns):
        raise ValueError("labels.csv must have columns: file,label")
    mp = {}
    for _, r in df.iterrows():
        fname = str(r["file"])
        mp[fname] = int(r["label"])
    return mp



def load_all_one_beat_csv(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, '**', 'one_beat.csv'), recursive=True)
    data = []
    filenames = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'signal' in df.columns:
                signal = df['signal'].values.astype(np.float64)
            else:
                signal = df.values.flatten().astype(np.float64)
            data.append(signal)
            # Use folder name + _one_beat.hea for label lookup
            folder_name = os.path.basename(os.path.dirname(csv_file))
            filenames.append(f"{folder_name}_one_beat.hea")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    return data, filenames

def main():
    cfg = parse_args()
    os.makedirs(cfg.outdir, exist_ok=True)


    # DEMO mode: make simple synthetic ECGs
    if cfg.demo:
        def synth(n=3600, fs=360, hr=70, noise=0.05, seed=0):
            rng = np.random.default_rng(seed)
            t = np.arange(n)/fs
            # simple stylized heartbeat impulses + noise
            rr = int(fs * 60/hr)
            x = np.zeros(n)
            for k in range(0, n, rr):
                if k < n: x[k] = 1.0
            # convolve with a QRS-like kernel
            kernel = np.exp(-np.linspace(-1,1,21)**2*10.0)
            x = np.convolve(x, kernel, mode="same")
            x += noise*rng.normal(size=n)
            return x.astype(np.float64)
        X = []; y = []
        for i in range(200):
            x = synth(seed=i, hr=70 + (i%2)*10)  # alternate classes
            X.append(extract_features(x, cfg.fs))
            y.append(i%2)
        df = pd.DataFrame(X); y = np.array(y, int)
    else:
        # Single file predict/inspect path
        if cfg.file and not cfg.labels:
            x = load_ecg(cfg.file, cfg.fmt)
            feats = extract_features(x, cfg.fs)
            df = pd.DataFrame([feats])
            print("Extracted features for single file:", json.dumps(feats, indent=2))
            if cfg.load_model and cfg.scaler and cfg.calibrator and cfg.threshold_file:
                gpr = joblib.load(cfg.load_model)
                scaler = joblib.load(cfg.scaler)
                with open(cfg.threshold_file, "r") as f:
                    thr = float(f.read().strip())
                cal = joblib.load(cfg.calibrator)
                Xs = scaler.transform(df.values)
                mu = gpr.predict(Xs).reshape(-1,1)
                p = cal.predict_proba(mu)[:,1]
                y_pred = (p >= thr).astype(int)
                print(f"Predicted probability: {float(p[0]):.3f}  class: {int(y_pred[0])}")
            df.to_csv(os.path.join(cfg.outdir, "features_single.csv"), index=False)
            return

        # Training path: data-dir + labels
        if not (cfg.data_dir and cfg.labels):
            raise SystemExit("Provide --data-dir and --labels, or use --file for single ECG.")

        label_map = load_label_map(cfg.labels)
        rows = []
        y = []
        signals, filenames = load_all_one_beat_csv(cfg.data_dir)

        # Create a balanced dataset for max_samples parameter
        if cfg.max_samples is not None:
            # Match filenames to labels
            label_map = load_label_map(cfg.labels)
            # Get indices for each class
            idx_0 = [i for i, f in enumerate(filenames) if label_map.get(f, None) == 0]
            idx_1 = [i for i, f in enumerate(filenames) if label_map.get(f, None) == 1]
            # Take up to half from each class
            n0 = min(len(idx_0), cfg.max_samples // 2)
            n1 = min(len(idx_1), cfg.max_samples - n0)
            selected_idx = idx_0[:n0] + idx_1[:n1]
            # If not enough for both, fill up with remaining
            if len(selected_idx) < cfg.max_samples:
                rest = [i for i in range(len(filenames)) if i not in selected_idx]
                selected_idx += rest[:cfg.max_samples - len(selected_idx)]
            signals = [signals[i] for i in selected_idx]
            filenames = [filenames[i] for i in selected_idx]

        for signal, fname in zip(signals, filenames):
            feats = extract_features(signal, cfg.fs)
            rows.append(feats)
            key = fname
            if key not in label_map:
                key2 = os.path.splitext(fname)[0]
                if key2 in label_map:
                    key = key2
                else:
                    raise KeyError(f"No label for {fname} in labels CSV.")
            y.append(label_map[key])
        df = pd.DataFrame(rows)
        y = np.array(y, int)

    # Handle NaNs by imputation (median)
    df = df.replace([np.inf, -np.inf], np.nan)
    med = df.median(numeric_only=True)
    df = df.fillna(med)

    # Split for training/eval if we have labels
    if not cfg.demo and cfg.file and not cfg.labels:
        # single-file path already returned
        return

    if not cfg.demo:
        # in non-demo training path, df and y already filled above
        pass

    X = df.values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    pca = None
    if cfg.pca:
        pca = PCA(n_components=cfg.pca, random_state=cfg.random_state)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # GPR with (Constant * RBF) + White noise
    nfeat = X_train.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(nfeat), length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=2, random_state=cfg.random_state
    )
    gpr.fit(X_train, y_train)

    # Regression metrics
    mu_test = gpr.predict(X_test)
    mse = mean_squared_error(y_test, mu_test)
    r2  = r2_score(y_test, mu_test)

    # Calibration
    if cfg.calibrate:
        # take 20% from train as val
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=cfg.random_state
        )
        gpr.fit(X_tr, y_tr)
        mu_val = gpr.predict(X_val).reshape(-1,1)
        cal = LogisticRegression(max_iter=300).fit(mu_val, y_val)
        p_val = cal.predict_proba(mu_val)[:,1]
        thr = float(np.linspace(0.05, 0.95, 19)[np.argmax([
            __import__("sklearn.metrics").metrics.f1_score(y_val, (p_val >= t).astype(int)) for t in np.linspace(0.05, 0.95, 19)
        ])])
        p_test = cal.predict_proba(mu_test.reshape(-1,1))[:,1]
    else:
        cal = None; thr = 0.5; p_test = np.clip(mu_test, 0.0, 1.0)

    y_pred = (p_test >= thr).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        roc = roc_auc_score(y_test, p_test)
    except ValueError:
        roc = float("nan")
    pr  = average_precision_score(y_test, p_test)

    print("== Regression (test) ==")
    print(f"MSE: {mse:.4f}  R^2: {r2:.4f}")
    print("== Detection (test) ==")
    print(f"Acc: {acc:.4f}  AUROC: {roc:.4f}  AUPRC: {pr:.4f}")
    print(classification_report(y_test, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5,4.0))
    ConfusionMatrixDisplay(cm, display_labels=["Negative","Positive"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (ECG features + GPR)")
    fig.tight_layout()
    cm_path = os.path.join(cfg.outdir, "confusion_matrix_ecg_gpr.png")
    fig.savefig(cm_path, dpi=150); plt.close(fig)

    # Save artifacts
    os.makedirs(cfg.outdir, exist_ok=True)
    df.to_csv(os.path.join(cfg.outdir, "features.csv"), index=False)
    joblib.dump(gpr, os.path.join(cfg.outdir, "model_gpr.joblib"))
    joblib.dump(scaler, os.path.join(cfg.outdir, "scaler.joblib"))
    if pca is not None:
        joblib.dump(pca, os.path.join(cfg.outdir, "pca.joblib"))
    if cfg.calibrate and cal is not None:
        joblib.dump(cal, os.path.join(cfg.outdir, "calibrator.joblib"))
    with open(os.path.join(cfg.outdir, "threshold.txt"), "w") as f:
        f.write(str(thr))

    report = {
        "regression": {"mse": float(mse), "r2": float(r2)},
        "detection": {"accuracy": float(acc), "roc_auc": float(roc), "pr_auc": float(pr), "threshold": float(thr)},
        "n_features": int(X_train.shape[1]),
        "notes": "Features: HRV (hr_mean/std, rr_mean/std, rmssd, pnn50), bandpowers, simple morphology (R amplitude, QRS width)."
    }
    with open(os.path.join(cfg.outdir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved to {cfg.outdir}")
    print(f"- {cm_path}")
    print("- model_gpr.joblib, scaler.joblib, [pca.joblib], [calibrator.joblib], threshold.txt, features.csv, report.json")


if __name__ == "__main__":
    main()
