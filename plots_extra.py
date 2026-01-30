# plots_extra.py
# -----------------------------------------
# Extra visualizations for DHT experiments
# Reads results/results.csv and produces:
# 1) Heatmaps per protocol (mean hops by operation x N)
# 2) Boxplots per operation (distribution of hops per protocol-N)
# 3) ECDF curves per operation and N (Chord vs Pastry)
# 4) Churn scatter: moved_keys vs move_hops for dynamic join/leave
#
# Usage:
#   python plots_extra.py --results results/results.csv --outdir results/extra
# -----------------------------------------

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OPS_ORDER = ["build", "insert", "update", "lookup", "delete", "join", "leave"]


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.array([]), np.array([])
    x = np.sort(v)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot_heatmaps(df: pd.DataFrame, outdir: Path):
    """
    Heatmap of mean hops by (operation x N) for each protocol.
    Uses df['hops'] (join/leave already store locate-hops in your pipeline).
    """
    _ensure_dir(outdir)

    # keep only known operations (in case extras exist)
    df2 = df[df["operation"].isin(OPS_ORDER)].copy()

    # ordered operations
    df2["operation"] = pd.Categorical(df2["operation"], categories=OPS_ORDER, ordered=True)

    for protocol in sorted(df2["protocol"].dropna().unique()):
        d = df2[df2["protocol"] == protocol]
        if d.empty:
            continue

        pivot = (d.groupby(["operation", "N"])["hops"]
                   .mean()
                   .reset_index()
                   .pivot(index="operation", columns="N", values="hops"))

        # ensure order
        pivot = pivot.reindex(index=OPS_ORDER)

        plt.figure()
        # imshow expects numeric matrix
        mat = pivot.values.astype(float)
        # mask NaNs so they appear blank
        mat_masked = np.ma.masked_invalid(mat)

        im = plt.imshow(mat_masked, aspect="auto")
        plt.colorbar(im, label="Mean hops")

        plt.xticks(ticks=np.arange(len(pivot.columns)), labels=[str(c) for c in pivot.columns])
        plt.yticks(ticks=np.arange(len(pivot.index)), labels=[str(r) for r in pivot.index])

        plt.xlabel("N (number of nodes)")
        plt.ylabel("operation")
        plt.title(f"Heatmap — mean hops (protocol={protocol})")

        # annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center")

        plt.tight_layout()
        plt.savefig(outdir / f"heatmap_{protocol.lower()}.png", dpi=170)
        plt.close()


def plot_boxplots(df: pd.DataFrame, outdir: Path, ops: List[str] = None):
    """
    Boxplot per operation: distribution of hops across protocol-N groups.
    x-axis = categories like 'Chord-16', 'Pastry-16', ...
    """
    _ensure_dir(outdir)

    if ops is None:
        ops = ["insert", "lookup", "update", "delete", "join", "leave"]

    df2 = df.copy()
    df2 = df2[df2["operation"].isin(ops)]

    # We'll plot one figure per operation to keep it clean
    for op in ops:
        d = df2[df2["operation"] == op].copy()
        if d.empty:
            continue

        # build stable ordering by N then protocol
        Ns = sorted(d["N"].dropna().unique())
        protocols = sorted(d["protocol"].dropna().unique())

        labels = []
        data = []

        for N in Ns:
            for pr in protocols:
                v = d[(d["N"] == N) & (d["protocol"] == pr)]["hops"].dropna().values
                if v.size == 0:
                    continue
                labels.append(f"{pr}-{int(N)}")
                data.append(v)

        if not data:
            continue

        plt.figure()
        plt.boxplot(data, showfliers=True)
        plt.xticks(ticks=np.arange(1, len(labels) + 1), labels=labels, rotation=45, ha="right")
        plt.ylabel("hops")
        plt.title(f"Boxplot — hops distribution ({op})")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(outdir / f"boxplot_{op}.png", dpi=170)
        plt.close()


def plot_ecdf_curves(df: pd.DataFrame, outdir: Path, ops: List[str] = None):
    """
    ECDF curves per operation and per N:
      - one plot per (op, N), with two lines: Chord vs Pastry
    """
    _ensure_dir(outdir)

    if ops is None:
        ops = ["lookup", "insert", "update", "delete"]

    df2 = df[df["operation"].isin(ops)].copy()
    if df2.empty:
        return

    Ns = sorted(df2["N"].dropna().unique())

    for op in ops:
        for N in Ns:
            d = df2[(df2["operation"] == op) & (df2["N"] == N)]
            if d.empty:
                continue

            plt.figure()
            for pr in sorted(d["protocol"].dropna().unique()):
                v = d[d["protocol"] == pr]["hops"].dropna().values
                x, y = _ecdf(v)
                if x.size == 0:
                    continue
                plt.plot(x, y, label=pr)

            plt.xlabel("hops")
            plt.ylabel("ECDF  P(Hops ≤ x)")
            plt.title(f"ECDF — {op} (N={int(N)})")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"ecdf_{op}_N{int(N)}.png", dpi=170)
            plt.close()


def plot_churn_scatter(df: pd.DataFrame, outdir: Path):
    """
    Scatter plot: moved_keys vs move_hops for dynamic join/leave.
    One plot per N to reduce clutter.
    """
    _ensure_dir(outdir)

    needed = {"operation", "phase", "protocol", "N", "move_hops", "moved_keys"}
    if not needed.issubset(df.columns):
        return

    d = df[(df["phase"] == "dynamic") & (df["operation"].isin(["join", "leave"]))].copy()
    d = d.dropna(subset=["move_hops", "moved_keys"])
    if d.empty:
        return

    Ns = sorted(d["N"].dropna().unique())
    for N in Ns:
        dn = d[d["N"] == N]
        if dn.empty:
            continue

        plt.figure()
        # distinguish (operation, protocol) via label; markers by operation
        for op in ["join", "leave"]:
            for pr in sorted(dn["protocol"].dropna().unique()):
                part = dn[(dn["operation"] == op) & (dn["protocol"] == pr)]
                if part.empty:
                    continue
                marker = "o" if op == "join" else "s"
                plt.scatter(part["moved_keys"].values, part["move_hops"].values, marker=marker, label=f"{pr}-{op}")

        plt.xlabel("moved_keys")
        plt.ylabel("move_hops (overhead)")
        plt.title(f"Churn overhead scatter — moved_keys vs move_hops (N={int(N)})")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"churn_scatter_N{int(N)}.png", dpi=170)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=str, required=True, help="Path to results.csv (e.g., results/results.csv)")
    ap.add_argument("--outdir", type=str, default="results/extra", help="Output directory for extra plots.")
    args = ap.parse_args()

    results_path = Path(args.results)
    outdir = Path(args.outdir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    # basic sanity
    for col in ["protocol", "operation", "N", "hops"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column in results.csv: {col}")

    _ensure_dir(outdir)

    plot_heatmaps(df, outdir)
    plot_boxplots(df, outdir)
    plot_ecdf_curves(df, outdir)
    plot_churn_scatter(df, outdir)

    print(f"[OK] Extra plots saved under: {outdir}")


if __name__ == "__main__":
    main()