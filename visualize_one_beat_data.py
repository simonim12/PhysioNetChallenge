#!/usr/bin/env python3
"""
Visualize one-beat ECG data from two folders (A vs B).

Assumes each beat is saved as:
  <folder>/.../<file_base>/one_beat.npy
and (optionally) <file_base>/one_beat_meta.json with:
  pre_sec, post_sec, target_len

Outputs (written to --outdir):
  overlay_random.png                 # random overlay of beats (A & B)
  mean_ci.png                        # mean ± 95% CI for A and B
  energy_hist.png                    # histogram of per-beat energy
  pca_scatter.png (if sklearn found) # PCA(2D) of beats
  grid_A.png, grid_B.png             # grids of sample beats from each folder

How to run
  python viz_two_folders_beats.py `
  --folder-a C:\Physionet\PhysioNetChallange\one_beat_out\GROUP_A `
  --folder-b C:\Physionet\PhysioNetChallange\one_beat_out\GROUP_B `
  --outdir   C:\Physionet\PhysioNetChallange\viz_beats `
  --overlay-k 150
"""

import os, json, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

# Optional: scikit-learn for PCA
try:
    from sklearn.decomposition import PCA
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def resample_to_len(x, n):
    if len(x) == n:
        return x.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    xq = np.linspace(0.0, 1.0, num=n, endpoint=True)
    return np.interp(xq, xp, x).astype(np.float32)

def load_folder(folder):
    """
    Returns:
      beats: np.array (N, L_i)  (raw lengths, may differ)
      metas: list of dicts (may be empty dicts if no meta)
    """
    paths = sorted(glob.glob(os.path.join(folder, "**", "one_beat.npy"), recursive=True))
    beats, metas = [], []
    for p in paths:
        try:
            b = np.load(p).astype(np.float32).ravel()
            # meta
            meta_path = os.path.join(os.path.dirname(p), "one_beat_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
            else:
                m = {}
            beats.append(b)
            metas.append(m)
        except Exception as e:
            print(f"[skip] {p}: {e}")
    if not beats:
        print(f"[warn] no beats found under: {folder}")
        return np.zeros((0,)), []
    # Return list so we can resample later
    return beats, metas

def infer_time_axis(metas, default_pre=0.25, default_post=0.45, target_len=None):
    """
    Try to infer a consistent time axis from metas; if mixed, fallback to defaults.
    """
    pre_vals = {round(float(m.get("pre_sec", default_pre)), 6) for m in metas if isinstance(m, dict)}
    post_vals = {round(float(m.get("post_sec", default_post)), 6) for m in metas if isinstance(m, dict)}
    len_vals = {int(m.get("target_len", -1)) for m in metas if isinstance(m, dict) and "target_len" in m}
    # choose
    pre = list(pre_vals)[0] if len(pre_vals) == 1 else default_pre
    post = list(post_vals)[0] if len(post_vals) == 1 else default_post
    if target_len is None:
        if len(len_vals) == 1 and list(len_vals)[0] > 0:
            L = list(len_vals)[0]
        else:
            L = None  # let caller decide (e.g., from other folder or flag)
    else:
        L = int(target_len)
    return pre, post, L

def make_time_axis(pre, post, L):
    if L is None:
        return None
    return np.linspace(-pre, post, L)

def resample_list(beats, L_target):
    return [resample_to_len(b, L_target) for b in beats]

def mean_ci(x, axis=0, ci=0.95):
    # mean ± z * std/sqrt(n); use 1.96 for ~95% if n is decent
    n = x.shape[0]
    if n < 2:
        return x.mean(axis=axis), np.zeros_like(x.mean(axis=axis))
    z = 1.96
    mu = x.mean(axis=axis)
    se = x.std(axis=axis, ddof=1) / np.sqrt(n)
    return mu, z * se

def plot_overlay_random(t, A, B, k, out_png, title="Overlay (random subset)"):
    np.random.seed(42)
    idxA = np.random.choice(len(A), size=min(k, len(A)), replace=False) if len(A) else []
    idxB = np.random.choice(len(B), size=min(k, len(B)), replace=False) if len(B) else []
    plt.figure(figsize=(9,4))
    for i in idxA:
        plt.plot(t, A[i], alpha=0.3, label="A" if i==idxA[0] else None)
    for i in idxB:
        plt.plot(t, B[i], alpha=0.3, label="B" if i==idxB[0] else None)
    plt.axvline(0.0)
    plt.title(title)
    plt.xlabel("Time around R (s)")
    plt.ylabel("Amplitude (z)")
    if len(idxA) and len(idxB): plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

def plot_mean_ci(t, A, B, out_png):
    plt.figure(figsize=(9,4))
    if len(A):
        muA, ciA = mean_ci(np.asarray(A))
        plt.plot(t, muA, label="A mean")
        plt.fill_between(t, muA-ciA, muA+ciA, alpha=0.3, label="A ±95% CI")
    if len(B):
        muB, ciB = mean_ci(np.asarray(B))
        plt.plot(t, muB, label="B mean")
        plt.fill_between(t, muB-ciB, muB+ciB, alpha=0.3, label="B ±95% CI")
    plt.axvline(0.0)
    plt.title("Mean ± 95% CI (A vs B)")
    plt.xlabel("Time around R (s)")
    plt.ylabel("Amplitude (z)")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

def plot_energy_hist(A, B, out_png):
    # simple per-beat energy
    eA = [float(np.sum(a*a)) for a in A] if len(A) else []
    eB = [float(np.sum(b*b)) for b in B] if len(B) else []
    plt.figure(figsize=(8,4))
    if len(eA): plt.hist(eA, bins=30, alpha=0.5, label="A")
    if len(eB): plt.hist(eB, bins=30, alpha=0.5, label="B")
    plt.title("Per-beat energy histogram")
    plt.xlabel("Energy"); plt.ylabel("Count")
    if len(eA) and len(eB): plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

def plot_pca(A, B, out_png):
    if not HAVE_SK or (len(A)==0 and len(B)==0):
        return
    X = []
    lab = []
    if len(A):
        X.append(np.asarray(A)); lab += ["A"]*len(A)
    if len(B):
        X.append(np.asarray(B)); lab += ["B"]*len(B)
    X = np.vstack(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(7,5))
    off = 0
    if len(A):
        plt.scatter(Z[:len(A),0], Z[:len(A),1], alpha=0.6, label="A")
        off = len(A)
    if len(B):
        plt.scatter(Z[off:,0], Z[off:,1], alpha=0.6, label="B")
    plt.title("PCA of beats (2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()

def plot_grid(t, arr, out_png, rows=3, cols=5, title="Samples"):
    n = min(rows*cols, len(arr))
    if n == 0: return
    idx = np.linspace(0, len(arr)-1, n, dtype=int)
    plt.figure(figsize=(cols*2.4, rows*1.8))
    for k, i in enumerate(idx):
        ax = plt.subplot(rows, cols, k+1)
        ax.plot(t, arr[i])
        ax.axvline(0.0)
        ax.set_xticks([]); ax.set_yticks([])
        if k==0: ax.set_title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser(description="Visualize one-beat data from two folders.")
    ap.add_argument("--folder-a", required=True, help="Folder A (root, searched recursively).")
    ap.add_argument("--folder-b", required=True, help="Folder B (root, searched recursively).")
    ap.add_argument("--outdir", default="viz_out", help="Where to save plots.")
    ap.add_argument("--pre", type=float, default=0.25, help="Fallback pre (s) if meta missing.")
    ap.add_argument("--post", type=float, default=0.45, help="Fallback post (s) if meta missing.")
    ap.add_argument("--target-len", type=int, default=None, help="Force resample length for both groups.")
    ap.add_argument("--overlay-k", type=int, default=100, help="How many random beats per group to overlay.")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    # Load A and B
    A_list, A_meta = load_folder(args.folder_a)
    B_list, B_meta = load_folder(args.folder_b)

    if (not A_list) and (not B_list):
        print("No beats found in either folder.")
        return

    # Infer time axis and resample length
    preA, postA, LA = infer_time_axis(A_meta, args.pre, args.post, args.target_len)
    preB, postB, LB = infer_time_axis(B_meta, args.pre, args.post, args.target_len)

    # Choose a common pre/post and length
    pre = preA if preA == preB else args.pre
    post = postA if postA == postB else args.post
    if args.target_len is not None:
        L = int(args.target_len)
    else:
        # prefer a known length; otherwise pick the mode of lengths
        cand = []
        if LA is not None: cand.append(LA)
        if LB is not None: cand.append(LB)
        if cand:
            L = int(cand[0])
        else:
            # fallback to the most frequent observed length among beats
            lengths = [len(b) for b in A_list+B_list]
            (vals, counts) = np.unique(lengths, return_counts=True)
            L = int(vals[np.argmax(counts)])
            print(f"[info] Using inferred target length: {L}")

    t = np.linspace(-pre, post, L)

    # Resample to L
    A = np.asarray(resample_list(A_list, L), dtype=np.float32) if len(A_list) else np.zeros((0,L), np.float32)
    B = np.asarray(resample_list(B_list, L), dtype=np.float32) if len(B_list) else np.zeros((0,L), np.float32)

    print(f"A: {A.shape} beats   B: {B.shape} beats   (L={L}, pre={pre}s, post={post}s)")

    # Plots
    plot_overlay_random(t, A, B, args.overlay_k, os.path.join(outdir, "overlay_random.png"))
    plot_mean_ci(t, A, B, os.path.join(outdir, "mean_ci.png"))
    plot_energy_hist(A, B, os.path.join(outdir, "energy_hist.png"))
    if HAVE_SK:
        plot_pca(A, B, os.path.join(outdir, "pca_scatter.png"))
    plot_grid(t, A, os.path.join(outdir, "grid_A.png"), title="A samples")
    plot_grid(t, B, os.path.join(outdir, "grid_B.png"), title="B samples")

    # Save stacked arrays (optional, for downstream work)
    np.savez_compressed(os.path.join(outdir, "beats_AB.npz"), A=A, B=B, t=t, pre=pre, post=post)
    print(f"Saved plots and data to: {outdir}")


if __name__ == "__main__":
    main()
