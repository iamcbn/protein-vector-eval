"""
05_statistical_analysis.py — Statistical Analysis and Visualization
====================================================================
Reads the raw per-query benchmark data produced by 04_online_inference.py
and 04b_online_inference_avq.py, and performs all statistical tests and
visualizations promised in the methodology (Chapter 3, Section 3.7).

Deliverables
------------
  results/wilcoxon_results.csv        — Wilcoxon p-values (latency + bio acc)
  results/full_results.csv            — Merged summary with compression ratio
  results/figures/
      pareto_speed_vs_accuracy.png    — Speed vs. Bio Accuracy scatter
                                         (AVQ plotted as hollow diamond on
                                          accuracy axis only — latency excluded)
      tradeoff_ann_vs_compression.png — ANN Recall@10 vs. Compression Ratio
      tradeoff_bio_vs_compression.png — Bio Accuracy@1 vs. Compression Ratio
      scatter_ann_vs_bio.png          — Scatter plot of ANN Recall vs Bio Accuracy
      latency_boxplot.png             — Box plot of per-query latencies
                                         (FAISS methods only; AVQ excluded)

Statistical method: Wilcoxon Signed-Rank Test (scipy.stats.wilcoxon).
Significance threshold: p < 0.05 (per §3.7 of the methodology).

AVQ LATENCY POLICY:
    Query latency for AVQ (ScaNN) is recorded in raw_latencies_avq.csv but
    is EXCLUDED from Wilcoxon latency comparisons and the Pareto plot speed
    axis.  ScaNN throughput reflects engine-level engineering (SIMD,
    batching, memory layout), not the AVQ quantisation algorithm.  All
    accuracy claims ARE cross-comparable.  See Chapter 3 §3.7.
"""

import os
import csv
from sys import path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Inputs (FAISS strategies — Windows pipeline)
RAW_LATENCIES_CSV   = os.path.join(RESULTS_DIR, "raw_latencies.csv")
RAW_ANN_CSV         = os.path.join(RESULTS_DIR, "raw_ann_hits.csv")
RAW_BIO_CSV         = os.path.join(RESULTS_DIR, "raw_bio_hits.csv")
BENCHMARK_CSV       = os.path.join(RESULTS_DIR, "benchmark_results.csv")
INDEXING_CSV        = os.path.join(RESULTS_DIR, "indexing_results.csv")
# Inputs (AVQ — WSL2 pipeline)
AVQ_BENCHMARK_CSV   = os.path.join(RESULTS_DIR, "benchmark_avq.csv")
AVQ_ANN_CSV         = os.path.join(RESULTS_DIR, "raw_ann_hits_avq.csv")
AVQ_BIO_CSV         = os.path.join(RESULTS_DIR, "raw_bio_hits_avq.csv")
AVQ_INDEXING_CSV    = os.path.join(RESULTS_DIR, "indexing_avq.csv")

# Outputs
WILCOXON_CSV        = os.path.join(RESULTS_DIR, "wilcoxon_results.csv")
FULL_RESULTS_CSV    = os.path.join(RESULTS_DIR, "full_results.csv")

# ---------------------------------------------------------------------------
# Design system — consistent colours and style across all figures
# ---------------------------------------------------------------------------
METHOD_ORDER = [
    "Mean Pooling (FlatL2)",
    "PCA + FlatL2",
    "Product Quantization (PQ)",
    "DCT Fingerprinting",
    "TEA Alphabet",
    "Knowledge Distillation (BiLSTM)",
    "AVQ (ScaNN)",      # accuracy metrics only — latency excluded (see §3.7)
]

# Methods for which latency Wilcoxon tests are skipped (per §3.7)
LATENCY_EXCLUDED = {"AVQ (ScaNN)"}

SHORT_LABELS = {
    "Mean Pooling (FlatL2)":            "Mean\nPooling",
    "PCA + FlatL2":                     "PCA\n+FlatL2",
    "Product Quantization (PQ)":        "PQ",
    "DCT Fingerprinting":               "DCT",
    "TEA Alphabet":                     "TEA",
    "Knowledge Distillation (BiLSTM)":  "KD\n(BiLSTM)",
    "AVQ (ScaNN)":                      "AVQ\n(ScaNN)",
}

# Curated palette — distinct but harmonious
PALETTE = {
    "Mean Pooling (FlatL2)":            "#4E79A7",
    "PCA + FlatL2":                     "#B07AA1",   # muted purple
    "Product Quantization (PQ)":        "#F28E2B",
    "DCT Fingerprinting":               "#E15759",
    "TEA Alphabet":                     "#76B7B2",
    "Knowledge Distillation (BiLSTM)":  "#59A14F",
    "AVQ (ScaNN)":                      "#FF9DA7",   # soft coral
}

def apply_style():
    """Light theme for journal/conference paper submission."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#111111",
        "axes.titlecolor":   "#111111",
        "axes.grid":         True,
        "grid.color":        "#DDDDDD",
        "grid.linestyle":    "--",
        "grid.alpha":        0.7,
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        "text.color":        "#111111",
        "legend.facecolor":  "white",
        "legend.edgecolor":  "#AAAAAA",
        "font.family":       "DejaVu Sans",
        "font.size":         12,
        "axes.titlesize":    15,
        "axes.titleweight":  "bold",
        "axes.labelsize":    13,
    })

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
def load_raw_data():
    """
    Load per-query data from all six FAISS methods (Windows pipeline) and
    merge accuracy data for AVQ (WSL2 pipeline).  AVQ latency is NOT merged
    into the main latency DataFrame because it must not enter Wilcoxon tests
    or the Pareto speed axis.
    """
    print("\n[1/5] Loading raw per-query data...")
    lat_df = pd.read_csv(RAW_LATENCIES_CSV)
    ann_df = pd.read_csv(RAW_ANN_CSV)
    bio_df = pd.read_csv(RAW_BIO_CSV)

    print(f"  ✓ raw_latencies.csv  : {lat_df.shape[0]} queries × {lat_df.shape[1]} FAISS methods")
    print(f"  ✓ raw_ann_hits.csv   : {ann_df.shape[0]} queries × {ann_df.shape[1]} FAISS methods")
    print(f"  ✓ raw_bio_hits.csv   : {bio_df.shape[0]} queries × {bio_df.shape[1]} FAISS methods")

    # Merge AVQ accuracy data if available
    if os.path.exists(AVQ_ANN_CSV) and os.path.exists(AVQ_BIO_CSV):
        avq_ann = pd.read_csv(AVQ_ANN_CSV)   # single column: 'AVQ (ScaNN)'
        avq_bio = pd.read_csv(AVQ_BIO_CSV)
        ann_df["AVQ (ScaNN)"] = avq_ann.iloc[:, 0].values
        bio_df["AVQ (ScaNN)"] = avq_bio.iloc[:, 0].values
        print(f"  ✓ raw_ann_hits_avq.csv / raw_bio_hits_avq.csv merged into accuracy DataFrames")
        print(f"  ⚠  AVQ latency NOT merged — excluded from latency comparisons (§3.7)")
    else:
        print(f"  [WARNING] AVQ accuracy CSVs not found — AVQ will be omitted from accuracy analysis.")
        print(f"            Run 04b_online_inference_avq.py (WSL2) to generate them.")

    # Verify row count
    for df, name in [(lat_df, "latencies"), (ann_df, "ann_hits"), (bio_df, "bio_hits")]:
        if df.shape[0] != 1000:
            print(f"  [WARNING] Expected 1000 rows in {name}, got {df.shape[0]}")

    return lat_df, ann_df, bio_df


# ---------------------------------------------------------------------------
# 2. Wilcoxon Signed-Rank Tests
# ---------------------------------------------------------------------------
def run_wilcoxon(lat_df, ann_df, bio_df):
    """
    Compare each non-baseline method against Mean Pooling (FlatL2) using
    the Wilcoxon Signed-Rank Test (two-sided, p < 0.05).

    LATENCY_EXCLUDED methods (AVQ / ScaNN) are skipped for latency tests
    and reported as N/A with an explanatory note, per Chapter 3 §3.7.

    Returns a list of result dicts for CSV output.
    """
    print("\n[2/5] Running Wilcoxon Signed-Rank Tests...")
    baseline    = "Mean Pooling (FlatL2)"
    comparisons = [m for m in METHOD_ORDER if m != baseline]

    wilcoxon_rows = []

    for method in comparisons:
        # Skip methods not present in the latency DataFrame (e.g. AVQ not merged)
        method_in_lat = method in lat_df.columns

        row = {"Method": method, "vs Baseline": baseline}

        # --- Latency ---
        if method in LATENCY_EXCLUDED or not method_in_lat:
            # Report N/A with explanatory note (per §3.7 methodology)
            row["Latency W-stat"]      = "N/A"
            row["Latency p-value"]     = "N/A"
            row["Latency Direction"]   = "excluded"
            row["Latency Sig (p<.05)"] = "N/A - latency not cross-comparable (ScaNN vs FAISS)"
            print(f"\n  -> {method}")
            print(f"     Latency  : EXCLUDED (engine-level confound — see §3.7)")
        else:
            base_lat = lat_df[baseline].values
            cand_lat = lat_df[method].values
            try:
                stat_lat, p_lat = wilcoxon(base_lat, cand_lat, alternative="two-sided")
                # Guard against NaN p-value (all differences == 0)
                if np.isnan(p_lat):
                    stat_lat, p_lat, direction_lat = float("nan"), 1.0, "identical"
                else:
                    direction_lat = "faster" if np.median(cand_lat) < np.median(base_lat) else "slower"
                sig_lat_console = "YES (p<.05)" if p_lat < 0.05 else "NO"
                sig_lat_csv     = "YES" if p_lat < 0.05 else "NO"
            except ValueError as e:
                stat_lat, p_lat, direction_lat = float("nan"), 1.0, "identical"
                sig_lat_console = sig_lat_csv = "NO"
                print(f"  [WARNING] Latency Wilcoxon for {method}: {e}")

            row["Latency W-stat"]      = round(float(stat_lat), 2) if not np.isnan(stat_lat) else "N/A"
            row["Latency p-value"]     = round(float(p_lat), 6)
            row["Latency Direction"]   = direction_lat
            row["Latency Sig (p<.05)"] = sig_lat_csv   # plain ASCII for CSV

            print(f"\n  -> {method}")
            print(f"     Latency  : W={row['Latency W-stat']}  p={row['Latency p-value']:.6f}"
                  f"  [{direction_lat}]  Significant: {sig_lat_console}")

        # --- Bio Accuracy (binary 0/1 per query — still valid for Wilcoxon) ---
        if method not in ann_df.columns or method not in bio_df.columns:
            # AVQ accuracy data was not produced by WSL2 script yet
            row["Bio Acc W-stat"]      = "N/A"
            row["Bio Acc p-value"]     = "N/A"
            row["Bio Acc Direction"]   = "data_missing"
            row["Bio Acc Sig (p<.05)"] = "N/A - run 04b_online_inference_avq.py"
            print(f"     Bio Acc  : MISSING (run 04b_online_inference_avq.py)")
        else:
            base_bio = bio_df[baseline].values.astype(float)
            cand_bio = bio_df[method].values.astype(float)
            try:
                stat_bio, p_bio = wilcoxon(base_bio, cand_bio, alternative="two-sided")
                if np.isnan(p_bio):
                    stat_bio, p_bio, direction_bio = float("nan"), 1.0, "identical"
                else:
                    direction_bio = "better" if np.mean(cand_bio) > np.mean(base_bio) else "worse/same"
                sig_bio_console = "YES (p<.05)" if p_bio < 0.05 else "NO"
                sig_bio_csv     = "YES" if p_bio < 0.05 else "NO"
            except ValueError as e:
                stat_bio, p_bio, direction_bio = float("nan"), 1.0, "identical"
                sig_bio_console = sig_bio_csv = "NO"
                print(f"  [WARNING] Bio Wilcoxon for {method}: {e}")

            row["Bio Acc W-stat"]      = round(float(stat_bio), 2) if not np.isnan(stat_bio) else "N/A"
            row["Bio Acc p-value"]     = round(float(p_bio), 6)
            row["Bio Acc Direction"]   = direction_bio
            row["Bio Acc Sig (p<.05)"] = sig_bio_csv   # plain ASCII for CSV

            print(f"     Bio Acc  : W={row['Bio Acc W-stat']}  p={row['Bio Acc p-value']:.6f}"
                  f"  [{direction_bio}]  Significant: {sig_bio_console}")

        wilcoxon_rows.append(row)

    # Save CSV with explicit UTF-8 encoding
    fieldnames = [
        "Method", "vs Baseline",
        "Latency W-stat", "Latency p-value", "Latency Direction", "Latency Sig (p<.05)",
        "Bio Acc W-stat",  "Bio Acc p-value",  "Bio Acc Direction",  "Bio Acc Sig (p<.05)",
    ]
    with open(WILCOXON_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(wilcoxon_rows)
    print(f"\n  Wilcoxon results saved -> {WILCOXON_CSV}")
    return wilcoxon_rows


# ---------------------------------------------------------------------------
# 3. Full merged table with Compression Ratio
# ---------------------------------------------------------------------------
def build_full_results():
    """
    Merge benchmark_results.csv + indexing_results.csv and compute
    Compression Ratio = baseline_size / method_size.
    """
    print("\n[3/5] Building full results table with Compression Ratio...")
    bench   = pd.read_csv(BENCHMARK_CSV)
    indexing = pd.read_csv(INDEXING_CSV)

    # Ensure Method column matches between files
    merged = bench.merge(indexing, on="Method", how="left")

    # If AVQ indexing CSV exists, append it (one extra row)
    if os.path.exists(AVQ_INDEXING_CSV):
        avq_idx = pd.read_csv(AVQ_INDEXING_CSV)[["Method", "Build Time (s)", "Index Size (GB)"]]
        # Also append AVQ accuracy from benchmark_avq.csv
        if os.path.exists(AVQ_BENCHMARK_CSV):
            avq_bench = pd.read_csv(AVQ_BENCHMARK_CSV)[["Method", "ANN Recall@10",
                                                         "Biological Accuracy@1",
                                                         "Avg Query Latency (ms)"]]
            avq_full = avq_bench.merge(avq_idx, on="Method", how="left")
            merged = pd.concat([merged, avq_full], ignore_index=True)
        print(f"  ✓ AVQ results appended from {AVQ_BENCHMARK_CSV} + {AVQ_INDEXING_CSV}")
    else:
        print(f"  [WARNING] {AVQ_INDEXING_CSV} not found — AVQ row will be omitted.")
        print(f"            Run 03b_offline_indexing_avq.py and 04b_online_inference_avq.py (WSL2).")

    baseline_size = merged.loc[
        merged["Method"] == "Mean Pooling (FlatL2)", "Index Size (GB)"
    ].values[0]

    merged["Compression Ratio"] = (baseline_size / merged["Index Size (GB)"]).round(2)

    # Reorder columns
    cols = [
        "Method",
        "ANN Recall@10",
        "Biological Accuracy@1",
        "Avg Query Latency (ms)",
        "Build Time (s)",
        "Index Size (GB)",
        "Compression Ratio",
    ]
    merged = merged[cols]

    merged.to_csv(FULL_RESULTS_CSV, index=False)
    print(f"  ✓ Full results saved → {FULL_RESULTS_CSV}")
    print()
    print(merged.to_string(index=False))
    return merged


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

def plot_pareto(df):
    """
    Pareto scatter: Speed vs. Biological Accuracy.

    FAISS methods (6): plotted normally with filled circles.
    AVQ (ScaNN): plotted as a hollow diamond on the accuracy axis only.
    A note explains that its speed coordinate is not cross-comparable.
    """
    print("  → Pareto scatter: Speed vs. Bio Accuracy")
    apply_style()
    fig, ax = plt.subplots(figsize=(13, 7))

    # Push labels far from dots; arrows connect them back
    label_offsets = {
        "Mean Pooling (FlatL2)":           ( 55,  20),   # upper-right
        "PCA + FlatL2":                    (-20, -30),   # lower-left
        "Product Quantization (PQ)":       ( 15,  18),
        "DCT Fingerprinting":              (-18,  28),   # upper-left
        "TEA Alphabet":                    ( 18, -28),   # lower-right
        "Knowledge Distillation (BiLSTM)": (-55,  20),   # upper-left
        "AVQ (ScaNN)":                     ( 40,  20),   # upper-right (accuracy-only)
    }

    arrow_style = dict(
        arrowstyle="-",
        color="#666666",
        lw=0.9,
        alpha=0.7,
    )

    # Separate FAISS methods from AVQ for different rendering
    faiss_df = df[df["Method"] != "AVQ (ScaNN)"]
    avq_rows = df[df["Method"] == "AVQ (ScaNN)"]

    # --- Plot FAISS methods (filled circles) ---
    for _, row in faiss_df.iterrows():
        m = row["Method"]
        if m not in PALETTE:
            continue
        x = row["Avg Query Latency (ms)"]
        y = row["Biological Accuracy@1"]
        c = PALETTE[m]

        ax.scatter(x, y, s=250, color=c, zorder=5,
                   edgecolors="#FFFFFF", linewidths=1.5)

        offset = label_offsets.get(m, (15, 15))
        ha = "right" if offset[0] < 0 else "left"
        va = "bottom" if offset[1] > 0 else "top"

        ax.annotate(
            SHORT_LABELS[m],
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=10,
            color=c,
            fontweight="bold",
            ha=ha, va=va,
            arrowprops=arrow_style,
        )

    # --- Plot AVQ (hollow diamond — accuracy axis only) ---
    if not avq_rows.empty:
        avq_row  = avq_rows.iloc[0]
        avq_y    = avq_row["Biological Accuracy@1"]
        avq_x_ref = faiss_df["Avg Query Latency (ms)"].mean()  # dummy x (excluded)
        avq_c    = PALETTE["AVQ (ScaNN)"]

        ax.scatter(
            avq_x_ref, avq_y,
            s=300, marker="D",
            facecolors="none", edgecolors=avq_c, linewidths=2.5,
            zorder=5, label="AVQ (ScaNN)",
        )
        ax.annotate(
            SHORT_LABELS["AVQ (ScaNN)"],
            xy=(avq_x_ref, avq_y),
            xytext=label_offsets["AVQ (ScaNN)"],
            textcoords="offset points",
            fontsize=10, color=avq_c, fontweight="bold",
            ha="left", va="bottom",
            arrowprops=arrow_style,
        )
        ax.annotate(
            "\u2191 accuracy only\n  latency not cross-comparable",
            xy=(avq_x_ref, avq_y),
            xytext=(0, -45),
            textcoords="offset points",
            fontsize=8, color="#888888", style="italic",
            ha="center",
        )

    # Pareto region lines (based on FAISS methods only)
    pareto_x = faiss_df["Avg Query Latency (ms)"].min() * 1.8
    pareto_y = faiss_df["Biological Accuracy@1"].max() * 0.90
    ax.axvline(pareto_x, color="#FFFFFF", linestyle=":", alpha=0.25, linewidth=1)
    ax.axhline(pareto_y, color="#FFFFFF", linestyle=":", alpha=0.25, linewidth=1)

    # Move Pareto annotation BELOW the h-line to avoid label collisions
    ax.annotate(
        "← Pareto-Optimal\n   Region",
        xy=(pareto_x * 0.55, pareto_y * 0.87),   # ← below the h-line now
        fontsize=9, color="#AAAAAA", style="italic",
    )

    ax.set_xlabel("Avg Query Latency (ms)  [lower is better →]  •  FAISS methods only")
    ax.set_ylabel("Biological Accuracy@1  [higher is better ↑]")
    ax.set_title("Pareto Frontier: Speed vs. Biological Accuracy\n"
                 "(AVQ/ScaNN shown on accuracy axis only — latency not cross-comparable)")

    legend_patches = [
        mpatches.Patch(color=PALETTE[m], label=m)
        for m in METHOD_ORDER if m != "AVQ (ScaNN)"
    ]
    # Add AVQ as hollow-diamond entry
    avq_patch = mpatches.Patch(
        facecolor="none", edgecolor=PALETTE["AVQ (ScaNN)"],
        label="AVQ (ScaNN)  [accuracy only]"
    )
    legend_patches.append(avq_patch)
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1),
              loc="upper left", fontsize=9)

    # --- Padding fix: add vertical headroom so the top annotation labels
    #     (KD, DCT, Mean Pooling) sit inside the axes instead of spilling
    #     over the top frame. Data points, offsets and styling are unchanged. ---
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.20 * (ymax - ymin))

    path = os.path.join(FIGURES_DIR, "pareto_speed_vs_accuracy.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")

def plot_tradeoff_ann(df):
    """Bar chart: ANN Recall@10 vs. Compression Ratio."""
    print("  → Trade-off: ANN Recall@10 vs. Compression Ratio")
    apply_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x     = np.arange(len(METHOD_ORDER))
    width = 0.4

    bars = ax1.bar(
        x, [df.loc[df["Method"] == m, "ANN Recall@10"].values[0] for m in METHOD_ORDER],
        width=width, color=[PALETTE[m] for m in METHOD_ORDER],
        edgecolor="#FFFFFF", linewidth=0.6, alpha=0.9, label="ANN Recall@10",
    )
    ax1.set_ylim(0, 1.12)
    ax1.set_ylabel("ANN Recall@10")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=10)
    ax1.set_title("ANN Recall@10 vs. Compression Ratio by Method")

    # Annotate bar values
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9, color="#FFFFFF",
        )

    # Secondary axis: Compression Ratio (line)
    ax2 = ax1.twinx()
    ratios = [df.loc[df["Method"] == m, "Compression Ratio"].values[0] for m in METHOD_ORDER]
    ax2.plot(x, ratios, color="#FFD700", marker="D", markersize=8,
             linewidth=2, label="Compression Ratio", zorder=6)
    ax2.set_ylabel("Compression Ratio  (×  baseline)", color="#FFD700")
    ax2.tick_params(axis="y", labelcolor="#FFD700")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=9)

    #fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "tradeoff_ann_vs_compression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")


def plot_tradeoff_bio(df):
    """Bar chart: Bio Accuracy@1 vs. Compression Ratio."""
    print("  → Trade-off: Bio Accuracy@1 vs. Compression Ratio")
    apply_style()
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x     = np.arange(len(METHOD_ORDER))
    width = 0.4

    bars = ax1.bar(
        x, [df.loc[df["Method"] == m, "Biological Accuracy@1"].values[0] for m in METHOD_ORDER],
        width=width, color=[PALETTE[m] for m in METHOD_ORDER],
        edgecolor="#FFFFFF", linewidth=0.6, alpha=0.9, label="Bio Accuracy@1",
    )
    ax1.set_ylabel("Biological Accuracy@1")
    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=10)
    ax1.set_title("Biological Accuracy@1 vs. Compression Ratio by Method")

    # Annotate bar values
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0003,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=9, color="#FFFFFF",
        )

    # Secondary axis: Compression Ratio (line)
    ax2 = ax1.twinx()
    ratios = [df.loc[df["Method"] == m, "Compression Ratio"].values[0] for m in METHOD_ORDER]
    ax2.plot(x, ratios, color="#FFD700", marker="D", markersize=8,
             linewidth=2, label="Compression Ratio", zorder=6)
    ax2.set_ylabel("Compression Ratio  (×  baseline)", color="#FFD700")
    ax2.tick_params(axis="y", labelcolor="#FFD700")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=9)

    #fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "tradeoff_bio_vs_compression.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")


def plot_scatter_ann_vs_bio(df):
    """Scatter plot: ANN Recall@10 vs. Biological Accuracy@1."""
    print("  → Scatter plot: ANN Recall@10 vs. Biological Accuracy@1")
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    label_offsets = {
        "Mean Pooling (FlatL2)":           ( 20,  20),
        "PCA + FlatL2":                    (-20,  20),
        "Product Quantization (PQ)":       (-45,  15),
        "DCT Fingerprinting":              ( 20, -20),
        "TEA Alphabet":                    (-35, -20),
        "Knowledge Distillation (BiLSTM)": (-20, -20),
        "AVQ (ScaNN)":                     ( 25, -15),
    }

    arrow_style = dict(arrowstyle="-", color="#666666", lw=0.9, alpha=0.7)

    for _, row in df.iterrows():
        m = row["Method"]
        if m not in PALETTE:
            continue
        x = row["ANN Recall@10"]
        y = row["Biological Accuracy@1"]
        c = PALETTE[m]

        ax.scatter(x, y, s=200, color=c, zorder=5, edgecolors="#FFFFFF", linewidths=1.5)

        offset = label_offsets.get(m, (15, 15))
        ha = "right" if offset[0] < 0 else "left"
        va = "bottom" if offset[1] > 0 else "top"

        ax.annotate(
            SHORT_LABELS[m],
            xy=(x, y), xytext=offset, textcoords="offset points",
            fontsize=10, color=c, fontweight="bold",
            ha=ha, va=va, arrowprops=arrow_style,
        )

    ax.set_xlabel("ANN Recall@10")
    ax.set_ylabel("Biological Accuracy@1")
    ax.set_title("ANN Recall@10 vs. Biological Accuracy@1")

    # Add margin
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - 0.02, xmax + 0.05)
    ax.set_ylim(ymin - 0.002, ymax + 0.002)

    path = os.path.join(FIGURES_DIR, "scatter_ann_vs_bio.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")


def plot_bubble_chart(df):
    print("  → Bubble Chart")
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    
    base_bio = df.loc[df["Method"] == "Mean Pooling (FlatL2)", "Biological Accuracy@1"].values[0]
    
    for _, row in df.iterrows():
        m = row["Method"]
        if m not in PALETTE: continue
        x = row["ANN Recall@10"] * 100
        y = (row["Biological Accuracy@1"] / base_bio) * 100
        s = max(row["Compression Ratio"] * 30, 50) # minimum size so baseline is visible
        c = PALETTE[m]
        
        ax.scatter(x, y, s=s, color=c, alpha=0.75, edgecolors="#333333", linewidths=1.5, zorder=5)
        ax.annotate(SHORT_LABELS[m].replace("\n", " "), (x, y), xytext=(0, -25), textcoords="offset points", ha='center', va='top', fontsize=9, fontweight='bold', color=c)
        
    ax.set_xlabel("Mathematical Recall (%)")
    ax.set_ylabel("Biological Accuracy (% of Uncompressed Baseline)")
    ax.set_title("Bubble Chart: Mathematical vs Biological Accuracy\n(Bubble Size = Compression Ratio)")
    
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(xmin - 5, xmax + 5)
    ax.set_ylim(ymin - 10, ymax + 20)
    
    path = os.path.join(FIGURES_DIR, "bubble_chart.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")


def plot_grouped_bar_line(df):
    print("  → Grouped Bar + Line Chart")
    apply_style()
    fig, ax1 = plt.subplots(figsize=(13, 7))
    
    x = np.arange(len(METHOD_ORDER))
    width = 0.35
    
    ann_vals = [df.loc[df["Method"] == m, "ANN Recall@10"].values[0] * 100 for m in METHOD_ORDER]
    base_bio = df.loc[df["Method"] == "Mean Pooling (FlatL2)", "Biological Accuracy@1"].values[0]
    bio_vals = [(df.loc[df["Method"] == m, "Biological Accuracy@1"].values[0] / base_bio) * 100 for m in METHOD_ORDER]
    comp_vals = [df.loc[df["Method"] == m, "Compression Ratio"].values[0] for m in METHOD_ORDER]
    
    bar1 = ax1.bar(x - width/2, ann_vals, width, label='ANN Recall@10 (%)', color='#4E79A7', edgecolor='white')
    bar2 = ax1.bar(x + width/2, bio_vals, width, label='Biological Accuracy@1 (% of baseline)', color='#59A14F', edgecolor='white')
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([SHORT_LABELS[m] for m in METHOD_ORDER], fontsize=10)
    ax1.set_ylim(0, 120)
    
    ax2 = ax1.twinx()
    line = ax2.plot(x, comp_vals, color='#F28E2B', marker='D', markersize=8, linewidth=3, label='Compression Ratio (x)')
    ax2.set_ylabel('Compression Ratio (Higher is smaller file size)', color='#F28E2B', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#F28E2B')
    
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.title('ANN Recall@10 and Biological Accuracy@1 vs. Compression Ratio')
    
    path = os.path.join(FIGURES_DIR, "grouped_bar_chart.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")


def plot_radar_chart(df):
    print("  → Radar chart: Math, Bio, Compression")
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    categories = ['Mathematical\nFidelity', 'Biological\nRelevance', 'Storage\nEfficiency']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], color="grey", size=9)
    plt.ylim(0, 1.1)
    
    max_ann = 1.0
    max_bio = df.loc[df["Method"] == "Mean Pooling (FlatL2)", "Biological Accuracy@1"].values[0]
    max_comp = df["Compression Ratio"].max()
    
    for _, row in df.iterrows():
        m = row["Method"]
        if m not in PALETTE: continue
        
        val_ann = row["ANN Recall@10"] / max_ann
        val_bio = row["Biological Accuracy@1"] / max_bio
        val_comp = row["Compression Ratio"] / max_comp
        
        values = [val_ann, val_bio, val_comp]
        values += values[:1]
        
        ax.plot(angles, values, color=PALETTE[m], linewidth=2, linestyle='solid', label=SHORT_LABELS[m].replace("\n", " "))
        ax.fill(angles, values, color=PALETTE[m], alpha=0.1)
        
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1))
    plt.title("Normalized Performance Profile (Radar Chart)", y=1.08)
    
    path = os.path.join(FIGURES_DIR, "radar_chart.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"     ✓ Saved → {path}")


def plot_latency_boxplot(lat_df):
    """
    Box plot of per-query latency distributions.
    FAISS methods only — AVQ (ScaNN) latency is excluded per §3.7.
    """
    print("  → Latency box plot (per-query distribution, FAISS methods only)")
    # Only include FAISS methods (exclude AVQ)
    faiss_methods = [m for m in METHOD_ORDER
                     if m != "AVQ (ScaNN)" and m in lat_df.columns]
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 9))

    data   = [lat_df[m].values for m in faiss_methods]
    colors = [PALETTE[m] for m in faiss_methods]

    bp = ax.boxplot(
        data,
        patch_artist=True,
        notch=False,
        vert=True,
        widths=0.7,                                   # ← up from 0.5
        medianprops=dict(color="#FFFFFF", linewidth=6),
        whiskerprops=dict(color="#888888", linewidth=1.5),
        capprops=dict(color="#888888",    linewidth=1.5),
        flierprops=dict(marker="o", markersize=4, alpha=0.35, markeredgecolor="none"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for flier, color in zip(bp["fliers"], colors):
        flier.set(markerfacecolor=color)

    ax.set_xticks(range(1, len(faiss_methods) + 1))
    ax.set_xticklabels([SHORT_LABELS[m] for m in faiss_methods], fontsize=12)
    ax.set_yscale("symlog")
    ax.set_ylabel("Query Latency (ms) [Log Scale]", fontsize=13)
    ax.set_title(
        "Per-Query Latency Distribution — FAISS Methods\n"
        "(1,000 queries — box = IQR, line = median, whiskers = 1.5×IQR)\n"
        "AVQ (ScaNN) excluded: latency not cross-comparable (§3.7)"
    )

    # Clip empty log-scale space at the top and bottom
    all_vals = np.concatenate(data)
    ax.set_ylim(all_vals.min() * 0.4, all_vals.max() * 5)

    # Median labels — small background box so they don't bleed into boxes
    medians = [np.median(d) for d in data]
    for i, med in enumerate(medians, start=1):
        ax.text(
            i, med * 1.15, f"{med:.3f} ms",
            ha="center", va="bottom", fontsize=9.5, color="#FFFFFF",
            bbox=dict(facecolor="#0F1117", alpha=0.55, edgecolor="none", pad=1),
        )

    path = os.path.join(FIGURES_DIR, "latency_boxplot.png")
    #fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")  # vector for LaTeX
    fig.savefig(path, dpi=300, bbox_inches="tight")   # ← 300 dpi for print
    plt.close(fig)
    #fig.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")  # vector for LaTeX
    #fig.savefig(path, dpi=300, bbox_inches="tight")                  # raster fallback
    print(f"     ✓ Saved → {path}")

# ---------------------------------------------------------------------------
# 5. Print final summary table
# ---------------------------------------------------------------------------
def print_summary(wilcoxon_rows, full_df):
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)

    print("\nFull Results Table (incl. Compression Ratio):")
    print(full_df.to_string(index=False))

    print("\nWilcoxon Signed-Rank Test Results (vs Mean Pooling baseline):")
    print(f"{'Method':<35} {'Lat p-val':>10} {'Lat Sig':>9} {'Bio p-val':>10} {'Bio Sig':>9}")
    print("-" * 77)
    for r in wilcoxon_rows:
        lat_pval = f"{r['Latency p-value']:.6f}" if isinstance(r['Latency p-value'], (float, int)) else str(r['Latency p-value'])
        bio_pval = f"{r['Bio Acc p-value']:.6f}" if isinstance(r['Bio Acc p-value'], (float, int)) else str(r['Bio Acc p-value'])
        print(
            f"{r['Method']:<35} "
            f"{lat_pval:>10} "
            f"{r['Latency Sig (p<.05)']:>9} "
            f"{bio_pval:>10} "
            f"{r['Bio Acc Sig (p<.05)']:>9}"
        )
    print("=" * 80)

    print("\nOutput files written:")
    files = [
        WILCOXON_CSV,
        FULL_RESULTS_CSV,
        os.path.join(FIGURES_DIR, "pareto_speed_vs_accuracy.png"),
        os.path.join(FIGURES_DIR, "tradeoff_ann_vs_compression.png"),
        os.path.join(FIGURES_DIR, "tradeoff_bio_vs_compression.png"),
        os.path.join(FIGURES_DIR, "scatter_ann_vs_bio.png"),
        os.path.join(FIGURES_DIR, "bubble_chart.png"),
        os.path.join(FIGURES_DIR, "grouped_bar_chart.png"),
        os.path.join(FIGURES_DIR, "radar_chart.png"),
        os.path.join(FIGURES_DIR, "latency_boxplot.png"),
    ]
    for f in files:
        status = "[OK]" if os.path.exists(f) else "[MISSING]"
        print(f"  {status}  {f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Statistical Analysis — Protein Vector Retrieval Study")
    print("=" * 60)

    # Check prerequisites
    for path, label in [
        (RAW_LATENCIES_CSV, "raw_latencies.csv"),
        (RAW_ANN_CSV,       "raw_ann_hits.csv"),
        (RAW_BIO_CSV,       "raw_bio_hits.csv"),
        (BENCHMARK_CSV,     "benchmark_results.csv"),
        (INDEXING_CSV,      "indexing_results.csv"),
    ]:
        if not os.path.exists(path):
            print(f"\n[ERROR] Required file not found: {label}")
            print("  → Run `python online_eval.py` first to generate raw data.")
            raise SystemExit(1)

    lat_df, ann_df, bio_df = load_raw_data()
    wilcoxon_rows          = run_wilcoxon(lat_df, ann_df, bio_df)
    full_df                = build_full_results()
    plot_pareto(full_df)
    plot_tradeoff_ann(full_df)
    plot_tradeoff_bio(full_df)
    plot_scatter_ann_vs_bio(full_df)
    plot_bubble_chart(full_df)
    plot_grouped_bar_line(full_df)
    plot_radar_chart(full_df)
    plot_latency_boxplot(lat_df)
    print_summary(wilcoxon_rows, full_df)

    print("\n✓ Statistical analysis complete.\n")


if __name__ == "__main__":
    main()
