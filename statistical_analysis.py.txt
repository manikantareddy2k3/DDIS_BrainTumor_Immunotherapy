"""
statistical_analysis.py
Statistical analysis and visualization of extracted tumor features.

This script expects a CSV file with rows for each image and columns at least:
 - Image Name
 - Tumor Area (px2)        # numeric
 - Shape                  # categorical (e.g., 'Lobulated','Irregular','Rounded')
 - Texture                # categorical (e.g., 'Homogeneous','Necrotic','Heterogeneous')
 - Tumor Type             # categorical label (Glioma/Meningioma/Pituitary/No Tumor)

Usage:
 python statistical_analysis.py --features_csv path/to/features.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp
from itertools import combinations

# -----------------------
# Effect size: non-parametric eta squared
# -----------------------
def eta_squared_np(H, N):
    # non-parametric eta squared approximation for Kruskal-Wallis
    return (H - 1) / (N - 1)

# -----------------------
# Cliff's Delta implementation
# -----------------------
def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size for two 1D arrays.
    Returns delta (float) and counts.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x); ny = len(y)
    greater = 0
    less = 0
    for xi in x:
        greater += np.sum(y < xi)
        less += np.sum(y > xi)
    delta = (greater - less) / (nx * ny)
    return delta

def interpret_cliffs(delta):
    a = abs(delta)
    if a < 0.147:
        return "negligible"
    elif a < 0.33:
        return "small"
    elif a < 0.474:
        return "medium"
    else:
        return "large"

# -----------------------
# Main analysis
# -----------------------
def analyze(features_csv, output_prefix="statistical"):
    df = pd.read_csv(features_csv)
    # Ensure numeric tumor area
    df['Tumor Area (px²)'] = pd.to_numeric(df['Tumor Area (px²)'], errors='coerce')
    df = df.dropna(subset=['Tumor Area (px²)'])
    print("Loaded:", df.shape)

    # Descriptive stats by tumor type
    grouped = df.groupby('Tumor Type')['Tumor Area (px²)']
    summary = grouped.agg(['median','mean','std', lambda x: np.percentile(x,75)-np.percentile(x,25)])
    summary.columns = ['median','mean','std','IQR']
    print("\nDescriptive summary:\n", summary)

    # Boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Tumor Type', y='Tumor Area (px²)', data=df, showfliers=False, palette='pastel')
    sns.swarmplot(x='Tumor Type', y='Tumor Area (px²)', data=df, color='0.25', size=3)
    plt.title('Tumor Area by Type')
    plt.savefig(f"{output_prefix}_boxplot.png")

    # Normality tests (Shapiro)
    print("\nShapiro-Wilk normality test (p-values):")
    for name, group in df.groupby('Tumor Type'):
        stat, p = stats.shapiro(group['Tumor Area (px²)'].sample(n=min(500, len(group)))) if len(group) >= 3 else (np.nan, np.nan)
        print(f"{name}: p = {p}")

    # Levene test for homogeneity of variances
    types = df['Tumor Type'].unique()
    arrays = [df.loc[df['Tumor Type']==t, 'Tumor Area (px²)'] for t in types]
    levene_stat, levene_p = stats.levene(*arrays)
    print(f"\nLevene test: p = {levene_p}")

    # Overall non-parametric comparison (Kruskal-Wallis)
    H, p_kw = stats.kruskal(*arrays)
    print(f"\nKruskal-Wallis: H = {H:.4f}, p = {p_kw:.6g}")
    N = len(df)
    eta2 = eta_squared_np(H, N)
    print("Non-parametric eta squared (η²) =", eta2)

    # Pairwise posthoc (Dunn)
    print("\nDunn's post-hoc pairwise p-values (Bonferroni corrected):")
    posthoc = sp.posthoc_dunn([df.loc[df['Tumor Type']==t, 'Tumor Area (px²)'] for t in types], p_adjust='bonferroni')
    posthoc.index = types; posthoc.columns = types
    print(posthoc)

    # Cliff's delta for each pair
    print("\nCliff's Delta (pairwise):")
    for a,b in combinations(types,2):
        xa = df.loc[df['Tumor Type']==a, 'Tumor Area (px²)']
        xb = df.loc[df['Tumor Type']==b, 'Tumor Area (px²)']
        delta = cliffs_delta(xa.values, xb.values)
        print(f"{a} vs {b}: δ = {delta:.3f}, interpretation = {interpret_cliffs(delta)}")

    # Save summary CSVs
    summary.to_csv(f"{output_prefix}_descriptive_summary.csv")
    posthoc.to_csv(f"{output_prefix}_dunn_posthoc.csv")

    print("\nPlots and CSVs saved.")

# -----------------------
# Command-line
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", type=str, required=True, help="CSV produced by feature extraction with 'Tumor Area (px²)' and 'Tumor Type' columns")
    args = parser.parse_args()
    analyze(args.features_csv)
