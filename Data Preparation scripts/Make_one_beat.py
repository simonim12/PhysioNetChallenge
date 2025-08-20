#!/usr/bin/env python3
"""
Bulk one-beat extractor: recursively scan a root folder for *.dat and
create ONE beat per file (plus per-file plot and metadata). Also writes
an aggregate dataset (beats_dataset.npz) and beats_manifest.csv in each beat folder.

Usage:
  python 'Data PreparationScripts'\make_one_beat.py --root C:\path\to\root --outdir C:\path\to\out

  python Make_one_beat.py --root C:\Physionet\PhysioNetChallange\samitrop_output --outdir 'C:\Physionet\PhysioNetChallange\data Processing' --raw-if-missing --fs 360 --nch 1 --lead 0 --dtype int16 --endian little --pre 0.25 --post 0.45 --target-len 300
"""

import os, sys, json, argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import wfdb
    HAVE_WFDB = True
except Exception:
    HAVE_WFDB = False

try:
    import neurokit2 as nk
    HAVE_NK = True
except Exception:
    HAVE_NK = False

try:
    from scipy.signal import butter, filtfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def zscore(x, eps=1e-9):
    m, s = float(np.mean(x)), float(np.std(x))
    return (x - m) / (s + eps)

def resample_to_len(x, target_len):
    if x.shape[0] == target_len:
        return x.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    xq = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
    return np.interp(xq, xp, x).astype(np.float32)

def bandpass_qrs(x, fs, lo=5.0, hi=35.0):
    if HAVE_SCIPY:
        nyq = 0.5 * fs
        b, a = butter(3, [lo/nyq, hi/nyq], btype="band")
        return filtfilt(b, a, x)
    win = max(3, int(round(fs * 0.6)))
    hp = x - np.convolve(x, np.ones(win)/win, mode="same")
    lpw = max(3, int(round(fs * 0.04)))
    return np.convolve(hp, np.ones(lpw)/lpw, mode="same")

def detect_r_peaks_simple(x, fs):
    y = bandpass_qrs(x, fs)
    y = y**2
    y = y / (y.max() + 1e-9)
    thr = 0.3 * y.mean() + 0.2 * y.max()
    refr = int(0.2 * fs)
    peaks = []
    i = 0; N = len(y)
    while i < N:
        if y[i] >= thr:
            j = i + int(np.argmax(y[i:min(N, i+refr)]))
            peaks.append(j); i += refr
        else:
            i += 1
    return np.asarray(peaks, dtype=int)

def pick_one_r(valid_peaks, which="first"):
    if valid_peaks.size == 0:
        raise RuntimeError("No valid R-peaks within margins.")
    return int(valid_peaks[len(valid_peaks)//2] if which == "middle" else valid_peaks[0])

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_with_wfdb(dat_path, lead=None, lead_name=None):
    if not HAVE_WFDB:
        raise FileNotFoundError("WFDB not installed.")
    base = os.path.splitext(dat_path)[0]
    hea = base + ".hea"
    if not os.path.exists(hea):
        raise FileNotFoundError(f"WFDB header not found: {hea}")
    rec = wfdb.rdrecord(base)
    fs = float(rec.fs)
    sig_names = getattr(rec, "sig_name", None)
    X = rec.p_signal

    if lead_name and sig_names:
        try:
            idx = [s.strip().lower() for s in sig_names].index(lead_name.strip().lower())
            lead = idx
        except ValueError:
            raise SystemExit(f"Lead name '{lead_name}' not found. Available: {sig_names}")
    if lead is None:
        lead = 0
    if lead < 0 or lead >= X.shape[1]:
        raise SystemExit(f"Lead index {lead} out of range. Leads: {sig_names or X.shape[1]}")
    lead_nm = sig_names[lead] if sig_names else None
    return X[:, lead].astype(float), fs, lead, lead_nm

def load_raw_dat(dat_path, fs, nch=1, lead=0, dtype="int16", endian="little"):
    if fs is None:
        raise SystemExit("--fs is required for raw mode (no .hea present).")
    dt = np.dtype({"int16":"i2","int32":"i4","float32":"f4","float64":"f8"}[dtype])
    if endian == "little":
        dt = dt.newbyteorder("<")
    elif endian == "big":
        dt = dt.newbyteorder(">")
    data = np.fromfile(dat_path, dtype=dt)
    if nch > 1:
        if data.size % nch != 0:
            raise RuntimeError(f"Raw size {data.size} not divisible by nch={nch}")
        data = data.reshape(-1, nch)
        if lead < 0 or lead >= nch:
            raise SystemExit(f"--lead {lead} out of range for nch={nch}")
        sig = data[:, lead].astype(float)
    else:
        sig = data.astype(float)
    return sig, float(fs), int(lead), None

def process_one_folder(folder_path, out_root, root, pre_sec, post_sec, target_len,
                      which_r, prefer_lead, prefer_lead_name, raw_if_missing,
                      raw_fs, raw_nch, raw_dtype, raw_endian):
    beats = []
    rows = []
    failed = []
    # Find all .dat files in this folder
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".dat"):
            dat_path = os.path.join(folder_path, fn)
            rel = os.path.relpath(dat_path, root)
            outdir = ensure_dir(os.path.join(out_root, os.path.splitext(rel)[0]))
            try:
                try:
                    sig, fs, lead_idx, lead_nm = load_with_wfdb(dat_path, prefer_lead, prefer_lead_name)
                    mode = "wfdb"
                except Exception as wferr:
                    if not raw_if_missing:
                        raise
                    sig, fs, lead_idx, lead_nm = load_raw_dat(dat_path, fs=raw_fs, nch=raw_nch,
                                                              lead=(prefer_lead or 0), dtype=raw_dtype, endian=raw_endian)
                    mode = "raw"
                if HAVE_NK:
                    ecg = nk.ecg_clean(sig, sampling_rate=fs)
                    _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
                    rpeaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
                else:
                    ecg = bandpass_qrs(sig, fs)
                    rpeaks = detect_r_peaks_simple(ecg, fs)

                pre = int(round(pre_sec * fs))
                post = int(round(post_sec * fs))
                valid = rpeaks[(rpeaks >= pre) & (rpeaks + post < len(ecg))]
                r = pick_one_r(valid, which=which_r)

                seg = ecg[r-pre : r+post]
                seg = zscore(seg)
                beat = resample_to_len(seg, target_len)
                t_rel = np.linspace(-pre_sec, post_sec, target_len)

                # Save per-file artifacts
                np.save(os.path.join(outdir, "one_beat.npy"), beat.astype(np.float32))
                np.savetxt(os.path.join(outdir, "one_beat.csv"), beat, delimiter=",", fmt="%.7f")

                plt.figure(figsize=(7.2, 3.0))
                plt.plot(t_rel, beat)
                plt.axvline(0.0)
                ttl = f"{os.path.basename(dat_path)} (lead={lead_nm or lead_idx}, fs={fs:.1f}Hz)"
                plt.title(ttl); plt.xlabel("Time around R (s)"); plt.ylabel("Amplitude (z)")
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, "one_beat.png"), dpi=150, bbox_inches="tight")
                plt.close()

                meta = {
                    "input_dat": os.path.abspath(dat_path),
                    "rel_path": rel.replace("\\","/"),
                    "mode": mode,
                    "fs": fs,
                    "lead_index": int(lead_idx),
                    "lead_name": lead_nm,
                    "pre_sec": pre_sec,
                    "post_sec": post_sec,
                    "target_len": target_len,
                    "r_index": int(r),
                    "r_time_s": float(r / fs),
                    "neurokit_used": bool(HAVE_NK),
                }
                with open(os.path.join(outdir, "one_beat_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                # Create .hea file with label
                hea_out = os.path.join(outdir, "one_beat.hea")
                with open(hea_out, "w") as fout:
                    fout.write(f"one_beat 1 {fs} {len(beat)} 16\n")
                    label = 1 if "samitrop" in dat_path.lower() else 0
                    fout.write(f"# Chagas label: {label}\n")

                beats.append(beat)
                rows.append(meta)
            except Exception as e:
                failed.append((dat_path, str(e)))
    # Save aggregate files in this folder
    if beats:
        np.savez_compressed(os.path.join(folder_path, "beats_dataset.npz"), beats=np.array(beats))
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(folder_path, "beats_manifest.csv"), index=False)
    if failed:
        with open(os.path.join(folder_path, "failed.json"), "w", encoding="utf-8") as f:
            json.dump([{"file": f, "error": e} for f, e in failed], f, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Recursively create ONE beat per .dat file and aggregate per folder.")
    ap.add_argument("--root", required=True, help="Root folder to scan for *.dat (recursively).")
    ap.add_argument("--outdir", required=True, help="Where to write outputs (mirrors folder structure).")
    ap.add_argument("--pre", type=float, default=0.25, help="Seconds before R.")
    ap.add_argument("--post", type=float, default=0.45, help="Seconds after R.")
    ap.add_argument("--target-len", type=int, default=300, help="Resampled beat length.")
    ap.add_argument("--which-r", choices=["first","middle"], default="first", help="Which valid R-peak to use.")
    ap.add_argument("--lead", type=int, default=None, help="Lead index (default 0).")
    ap.add_argument("--lead-name", type=str, default=None, help="Lead name (e.g., I, II, V1). Overrides --lead if found.")

    ap.add_argument("--raw-if-missing", action="store_true", help="If .hea missing, load .dat as raw.")
    ap.add_argument("--fs", type=float, default=None, help="(RAW) sampling rate (Hz).")
    ap.add_argument("--nch", type=int, default=1, help="(RAW) number of interleaved channels.")
    ap.add_argument("--dtype", type=str, default="int16", choices=["int16","int32","float32","float64"], help="(RAW) data type.")
    ap.add_argument("--endian", type=str, default="little", choices=["little","big"], help="(RAW) endianness.")

    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.outdir)
    os.makedirs(out_root, exist_ok=True)

    # Find all subfolders under root
    for dirpath, dirnames, filenames in os.walk(root):
        # Only process folders that contain .dat files
        if any(fn.lower().endswith(".dat") for fn in filenames):
            process_one_folder(
                folder_path=dirpath,
                out_root=out_root,
                root=root,
                pre_sec=args.pre,
                post_sec=args.post,
                target_len=args.target_len,
                which_r=args.which_r,
                prefer_lead=args.lead,
                prefer_lead_name=args.lead_name,
                raw_if_missing=args.raw_if_missing,
                raw_fs=args.fs,
                raw_nch=args.nch,
                raw_dtype=args.dtype,
                raw_endian=args.endian,
            )
    print("\nDone.")

if __name__ == "__main__":
    main()