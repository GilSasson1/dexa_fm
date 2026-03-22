"""
Diagnostic script: analyze residuals and embeddings to understand what the model is learning.

Helps identify:
- Whether residuals are structured or mostly noise
- If embeddings are similar across visits (-> hard problem)
- Relationship between Δt and residual magnitude
- Data quality issues
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Config (match train_diffusion.py)
EMBEDDINGS_PATH = "/net/mraid20/export/genie/LabData/Analyses/gilsa/embeddings/lejepa/vit_small_new_data.pkl"
TARGETS_CSV = "/net/mraid20/export/genie/LabData/Analyses/10K_Trajectories/body_systems/Age_Gender_BMI.csv"
OUTPUT_DIR = "/net/mraid20/export/genie/LabData/Analyses/gilsa/diagnostics/"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("DIAGNOSTIC: Residual & Embedding Analysis")
    print("=" * 70)

    # ---- Load embeddings ----
    print(f"\n[1/5] Loading embeddings from {EMBEDDINGS_PATH}...")
    emb_df = pd.read_pickle(EMBEDDINGS_PATH)
    emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
    embed_dim = len(emb_cols)
    print(f"  Shape: {emb_df.shape}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Index levels: {emb_df.index.names}")

    # ---- Load targets ----
    print(f"\n[2/5] Loading targets from {TARGETS_CSV}...")
    tgt_df = pd.read_csv(TARGETS_CSV)
    tgt_df['RegistrationCode'] = tgt_df['RegistrationCode'].astype(str).apply(
        lambda x: f"10K_{x}" if not x.startswith("10K_") else x)
    tgt_df['research_stage'] = tgt_df['research_stage'].replace('00_00_visit', 'baseline')
    tgt_df.set_index(['RegistrationCode', 'research_stage'], inplace=True)
    print(f"  Shape: {tgt_df.shape}")
    print(f"  Columns: {tgt_df.columns.tolist()}")

    # ---- Find multi-visit pairs ----
    print(f"\n[3/5] Finding visit pairs...")
    emb_subjects = emb_df.index.get_level_values(0).unique()
    tgt_subjects = tgt_df.index.get_level_values(0).unique()
    common_subs = list(set(emb_subjects) & set(tgt_subjects))

    multi_visit_subs = []
    for sid in common_subs:
        n_emb = emb_df.loc[sid].shape[0] if isinstance(emb_df.loc[sid], pd.DataFrame) else 1
        n_tgt = tgt_df.loc[sid].shape[0] if isinstance(tgt_df.loc[sid], pd.DataFrame) else 1
        if min(n_emb, n_tgt) >= 2:
            multi_visit_subs.append(sid)

    print(f"  Total subjects: {len(common_subs)}")
    print(f"  Multi-visit subjects (>=2): {len(multi_visit_subs)}")

    # ---- Collect residuals & metrics ----
    print(f"\n[4/5] Computing residuals & embedding statistics...")
    residuals = []
    cosine_sims = []
    delta_ts = []
    residual_norms = []

    for sid in multi_visit_subs[:500]:  # limit for speed
        try:
            emb_visits = emb_df.loc[sid]
            tgt_visits = tgt_df.loc[sid]

            if isinstance(emb_visits, pd.Series):
                continue  # Only one visit
            if isinstance(tgt_visits, pd.Series):
                continue

            # Get common visits
            common_v = set(emb_visits.index) & set(tgt_visits.index)
            if len(common_v) < 2:
                continue

            visits_with_data = []
            for v in common_v:
                age = tgt_visits.loc[v, 'age']
                if pd.isna(age):
                    continue
                z = emb_visits.loc[v, emb_cols].values.astype(np.float32)
                visits_with_data.append((v, float(age), z))

            visits_with_data.sort(key=lambda x: x[1])

            # All pairs
            for i in range(len(visits_with_data)):
                for j in range(i + 1, len(visits_with_data)):
                    v_i, age_i, z_i = visits_with_data[i]
                    v_j, age_j, z_j = visits_with_data[j]

                    residual = z_j - z_i
                    residuals.append(residual)
                    delta_ts.append(age_j - age_i)

                    # Cosine similarity
                    cos_sim = np.dot(z_i, z_j) / (np.linalg.norm(z_i) * np.linalg.norm(z_j) + 1e-8)
                    cosine_sims.append(cos_sim)

                    # Residual norm
                    residual_norms.append(np.linalg.norm(residual))

        except Exception as e:
            pass

    residuals = np.array(residuals)
    cosine_sims = np.array(cosine_sims)
    delta_ts = np.array(delta_ts)
    residual_norms = np.array(residual_norms)

    print(f"  Collected {len(residuals)} visit pairs")

    # ---- Analysis ----
    print(f"\n[5/5] ANALYSIS RESULTS")
    print("=" * 70)

    print("\n📊 RESIDUAL STATISTICS:")
    print(f"  Mean norm: {residual_norms.mean():.6f}")
    print(f"  Std norm:  {residual_norms.std():.6f}")
    print(f"  Min norm:  {residual_norms.min():.6f}")
    print(f"  Max norm:  {residual_norms.max():.6f}")
    print(f"  Percentiles (10/50/90): {np.percentile(residual_norms, [10, 50, 90])}")

    print("\n🔗 COSINE SIMILARITY (z_i vs z_j):")
    print(f"  Mean: {cosine_sims.mean():.6f}")
    print(f"  Std:  {cosine_sims.std():.6f}")
    print(f"  Min:  {cosine_sims.min():.6f}")
    print(f"  Max:  {cosine_sims.max():.6f}")
    if cosine_sims.mean() > 0.98:
        print("  ⚠️  WARNING: Embeddings are VERY similar across visits! (>0.98 cosine sim)")
        print("     This makes the task extremely hard for a diffusion model.")
    elif cosine_sims.mean() > 0.95:
        print("  ⚠️  Embeddings are quite similar (>0.95). Task is challenging.")

    print("\n⏱️  TIME delta (Δt in years):")
    print(f"  Mean: {delta_ts.mean():.2f}")
    print(f"  Std:  {delta_ts.std():.2f}")
    print(f"  Min:  {delta_ts.min():.2f}")
    print(f"  Max:  {delta_ts.max():.2f}")

    # Correlation between Δt and residual norm
    corr = np.corrcoef(delta_ts, residual_norms)[0, 1]
    print(f"  Correlation(Δt, residual_norm): {corr:.4f}")
    if corr < 0.1:
        print("  ⚠️  WARNING: Weak correlation! Residuals don't scale with time.")
        print("     The model will struggle to learn a time-dependent relationship.")

    print("\n✅ RESIDUAL PER-ELEMENT STATS:")
    print(f"  Mean (per-dim): {residuals.mean():.8f}")
    print(f"  Std (per-dim):  {residuals.std():.8f}")
    print(f"  This should be normalized to ~0 mean, ~1 std for diffusion.")

    # ---- Visualizations ----
    print("\n📈 Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Residual norm distribution
    axes[0, 0].hist(residual_norms, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Residual Norm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Residual Norms')
    axes[0, 0].axvline(residual_norms.mean(), color='r', linestyle='--', label='Mean')
    axes[0, 0].legend()

    # Plot 2: Cosine similarity distribution
    axes[0, 1].hist(cosine_sims, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Cosine Similarity')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Distribution of Cosine Similarities (z_i vs z_j)')
    axes[0, 1].axvline(cosine_sims.mean(), color='r', linestyle='--', label='Mean')
    axes[0, 1].legend()

    # Plot 3: Δt vs residual norm
    axes[1, 0].scatter(delta_ts, residual_norms, alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Δt (years)')
    axes[1, 0].set_ylabel('Residual Norm')
    axes[1, 0].set_title(f'Time Delta vs Residual Magnitude (corr={corr:.3f})')
    
    # Add trend line
    z = np.polyfit(delta_ts, residual_norms, 1)
    p = np.poly1d(z)
    dt_sorted = np.sort(delta_ts)
    axes[1, 0].plot(dt_sorted, p(dt_sorted), "r--", linewidth=2, label='Trend')
    axes[1, 0].legend()

    # Plot 4: Cosine sim vs Δt
    axes[1, 1].scatter(delta_ts, cosine_sims, alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Δt (years)')
    axes[1, 1].set_ylabel('Cosine Similarity')
    axes[1, 1].set_title('Time Delta vs Embedding Similarity')
    axes[1, 1].axhline(0.95, color='r', linestyle='--', label='0.95 threshold')
    axes[1, 1].axhline(0.98, color='orange', linestyle='--', label='0.98 threshold')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/residual_analysis.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR}/residual_analysis.png")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("🎯 INTERPRETATION & RECOMMENDATIONS:")
    print("=" * 70)

    if cosine_sims.mean() > 0.98:
        print("\n❌ PROBLEM: Embeddings are nearly identical across visits!")
        print("   → Your encoder is not sensitive to aging/disease progression")
        print("   → The diffusion model has almost nothing to learn")
        print("   → SOLUTIONS:")
        print("      1. Check if the encoder was trained on age prediction (it should be)")
        print("      2. Try a different encoder backbone (ViT-Large instead of ViT-Small)")
        print("      3. Use a VAE or other method to amplify biological signal")
    elif corr < 0.1:
        print("\n⚠️  WEAK SIGNAL: Residuals don't depend on time!")
        print("   → Even if embeddings change, they don't correlate with Δt")
        print("   → The task is underspecified (too much random noise)")
        print("   → SOLUTIONS:")
        print("      1. Filter to only patients with large age gaps (>1 year)")
        print("      2. Use baseline probe R² as ground truth (the model should predict probe(z))")
        print("      3. Switch to predicting z_target directly instead of residual")
    elif residual_norms.mean() < 0.01:
        print("\n⚠️  VERY SMALL RESIDUALS:", f"{residual_norms.mean():.6f}")
        print("   → May need aggressive scaling or VAE preprocessing")
    else:
        print("\n✅ DATA LOOKS REASONABLE!")
        print("   → Embeddings are diverse (low cosine sim)")
        print("   → Residuals correlate with time")
        print("   → Diffusion model *should* be able to learn this")
        print("   → If performance is still poor, check model architecture or hyperparameters")

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
