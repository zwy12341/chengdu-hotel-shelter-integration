"""
Appendix X (Table X1): CDI Sensitivity to α (AHP–Entropy Combination Weight)
==============================================================================
Recomputes CDI under α = 0.3, 0.4, 0.5, 0.6, 0.7
and computes Spearman ρ relative to α = 0.5 baseline.
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ── Load data ──────────────────────────────────────────────────
risk = pd.read_excel(
    "/mnt/user-data/uploads/成都五城区_地震风险_更新坐标.xlsx",
    sheet_name="计算过程",
)
print(f"Loaded {len(risk)} grid cells")

# ── Define AHP and Entropy weights (from parameter sheet) ─────
# Weights are WITHIN-DIMENSION (sum to 1 within each dimension)
weights = {
    # Hazard dimension
    "H": {
        "indicators": ["H11_标准化", "H13_标准化"],
        "ahp":     [0.6667, 0.3333],
        "entropy": [0.4747, 0.5253],  # same for day and night
    },
    # Exposure dimension (daytime)
    "E_day": {
        "indicators": ["E11_昼_标准化", "E12_标准化", "E21_标准化", "E31_标准化"],
        "ahp":     [0.4829, 0.0882, 0.1570, 0.2720],
        "entropy": [0.4905, 0.0643, 0.2006, 0.2446],
    },
    # Exposure dimension (nighttime)
    "E_night": {
        "indicators": ["E11_夜_标准化", "E12_标准化", "E21_标准化", "E31_标准化"],
        "ahp":     [0.4829, 0.0882, 0.1570, 0.2720],
        "entropy": [0.4829, 0.0653, 0.2036, 0.2482],
    },
    # Vulnerability dimension
    "V": {
        "indicators": ["V11_标准化", "V21_标准化", "V22_标准化", "V31_标准化", "V32_标准化"],
        "ahp":     [0.1529, 0.4147, 0.2573, 0.0876, 0.0876],
        "entropy": [0.2608, 0.3083, 0.0717, 0.1888, 0.1704],
    },
}

# ── Compute CDI for each α ────────────────────────────────────
alpha_values = [0.3, 0.4, 0.5, 0.6, 0.7]

def compute_dimension_score(df, dim_config, alpha):
    """Compute dimension score with given alpha."""
    combined_w = [
        alpha * a + (1 - alpha) * e
        for a, e in zip(dim_config["ahp"], dim_config["entropy"])
    ]
    score = np.zeros(len(df))
    for col, w in zip(dim_config["indicators"], combined_w):
        score += w * df[col].values
    return score

cdi_results = {}
for alpha in alpha_values:
    H_score = compute_dimension_score(risk, weights["H"], alpha)
    E_day_score = compute_dimension_score(risk, weights["E_day"], alpha)
    E_night_score = compute_dimension_score(risk, weights["E_night"], alpha)
    V_score = compute_dimension_score(risk, weights["V"], alpha)

    cdi_day = H_score * E_day_score * V_score
    cdi_night = H_score * E_night_score * V_score

    cdi_results[alpha] = {"day": cdi_day, "night": cdi_night}

# Verify α=0.5 matches existing data
diff_day = np.abs(cdi_results[0.5]["day"] - risk["R_风险_昼"].values).max()
diff_night = np.abs(cdi_results[0.5]["night"] - risk["R_风险_夜"].values).max()
print(f"Verification - Max diff CDI(α=0.5) vs original: day={diff_day:.2e}, night={diff_night:.2e}")

# ══════════════════════════════════════════════════════════════
# Table X1: Spearman ρ between CDI(α) and CDI(α=0.5)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE X1: Spearman Rank Correlation of CDI Under Different α Values")
print("     (All compared to α = 0.5 baseline)")
print("=" * 70)

results_x1 = []
baseline_day = cdi_results[0.5]["day"]
baseline_night = cdi_results[0.5]["night"]

for alpha in alpha_values:
    rho_day, p_day = spearmanr(cdi_results[alpha]["day"], baseline_day)
    rho_night, p_night = spearmanr(cdi_results[alpha]["night"], baseline_night)

    print(f"  α = {alpha}: ρ_day = {rho_day:.6f} (p={p_day:.2e}), "
          f"ρ_night = {rho_night:.6f} (p={p_night:.2e})")

    results_x1.append({
        "alpha": alpha,
        "Spearman_rho_daytime": round(rho_day, 6),
        "p_value_daytime": f"{p_day:.2e}",
        "Spearman_rho_nighttime": round(rho_night, 6),
        "p_value_nighttime": f"{p_night:.2e}",
    })

df_x1 = pd.DataFrame(results_x1)
df_x1.to_csv("/home/claude/appendix_X/table_X1_alpha_sensitivity.csv", index=False)
print(f"\nSaved: table_X1_alpha_sensitivity.csv")


# ══════════════════════════════════════════════════════════════
# Pairwise ρ matrix (all α pairs)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUPPLEMENTARY: Full Pairwise Spearman ρ Matrix (Daytime)")
print("=" * 70)

n_alpha = len(alpha_values)
rho_matrix_day = np.ones((n_alpha, n_alpha))
rho_matrix_night = np.ones((n_alpha, n_alpha))

for i in range(n_alpha):
    for j in range(i + 1, n_alpha):
        rho_d, _ = spearmanr(cdi_results[alpha_values[i]]["day"],
                             cdi_results[alpha_values[j]]["day"])
        rho_n, _ = spearmanr(cdi_results[alpha_values[i]]["night"],
                             cdi_results[alpha_values[j]]["night"])
        rho_matrix_day[i, j] = rho_matrix_day[j, i] = rho_d
        rho_matrix_night[i, j] = rho_matrix_night[j, i] = rho_n

labels = [f"α={a}" for a in alpha_values]
df_matrix_day = pd.DataFrame(rho_matrix_day, index=labels, columns=labels).round(6)
df_matrix_night = pd.DataFrame(rho_matrix_night, index=labels, columns=labels).round(6)

print("\nDaytime:")
print(df_matrix_day.to_string())
print("\nNighttime:")
print(df_matrix_night.to_string())

df_matrix_day.to_csv("/home/claude/appendix_X/pairwise_rho_matrix_daytime.csv")
df_matrix_night.to_csv("/home/claude/appendix_X/pairwise_rho_matrix_nighttime.csv")


# ══════════════════════════════════════════════════════════════
# Additional: CDI descriptive stats under each α
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUPPLEMENTARY: CDI Descriptive Statistics by α (Daytime)")
print("=" * 70)

results_desc = []
for alpha in alpha_values:
    cdi = cdi_results[alpha]["day"]
    desc = {
        "alpha": alpha,
        "mean": round(np.mean(cdi), 6),
        "median": round(np.median(cdi), 6),
        "std": round(np.std(cdi), 6),
        "min": round(np.min(cdi), 6),
        "max": round(np.max(cdi), 6),
        "p75": round(np.percentile(cdi, 75), 6),
    }
    print(f"  α={alpha}: mean={desc['mean']:.6f}, median={desc['median']:.6f}, "
          f"std={desc['std']:.6f}, range=[{desc['min']:.6f}, {desc['max']:.6f}]")
    results_desc.append(desc)

df_desc = pd.DataFrame(results_desc)
df_desc.to_csv("/home/claude/appendix_X/cdi_descriptive_stats.csv", index=False)


# ══════════════════════════════════════════════════════════════
# Check high-demand area consistency
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUPPLEMENTARY: High-Demand Area Identification Consistency")
print("=" * 70)

# Using 75th percentile of CDI to identify high-demand areas
for alpha in alpha_values:
    cdi = cdi_results[alpha]["day"]
    threshold = np.percentile(cdi, 75)
    high_demand = cdi >= threshold

    # Compare with α=0.5
    cdi_base = cdi_results[0.5]["day"]
    threshold_base = np.percentile(cdi_base, 75)
    high_demand_base = cdi_base >= threshold_base

    overlap = (high_demand & high_demand_base).sum()
    union = (high_demand | high_demand_base).sum()
    jaccard = overlap / union if union > 0 else 0
    agree = (high_demand == high_demand_base).mean() * 100

    print(f"  α={alpha}: High-demand overlap with α=0.5: "
          f"Jaccard={jaccard:.4f}, Agreement={agree:.1f}%, "
          f"n_high={high_demand.sum()}")

print("\n✅ Appendix X analysis complete!")
