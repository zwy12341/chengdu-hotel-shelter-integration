"""
Appendix E: Sensitivity Analysis of Search-Radius Parameters
=============================================================
Computes:
  Table E2 - Pairwise Spearman ρ of ESI across scenarios A/B/C (day & night)
  Table E3 - Categorical agreement rate & district-level mean ESI
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json

# ── Load data ──────────────────────────────────────────────────
day = pd.read_csv("/mnt/user-data/uploads/grid_accessibility_daytime.csv")
night = pd.read_csv("/mnt/user-data/uploads/grid_accessibility_nighttime.csv")
risk = pd.read_excel(
    "/mnt/user-data/uploads/成都五城区_地震风险_更新坐标.xlsx",
    sheet_name="计算过程",
)

# Merge district info
day = day.merge(risk[["GRID_ID", "所属区"]], on="GRID_ID", how="left")
night = night.merge(risk[["GRID_ID", "所属区"]], on="GRID_ID", how="left")

scenarios = {"A (d₀=3000m)": "Ai_A", "B (d₀=2000m)": "Ai_B", "C (d₀=1000m)": "Ai_C"}
district_map = {"青羊区": "Qingyang", "武侯区": "Wuhou", "金牛区": "Jinniu",
                "锦江区": "Jinjiang", "成华区": "Chenghua"}

# ── Helper: categorize ESI into 4 tiers ───────────────────────
def categorize_esi(esi):
    """
    充足 (Sufficient): ESI >= 1.5
    基本平衡 (Moderate): 1.0 <= ESI < 1.5
    供给不足 (Deficient): 0.5 <= ESI < 1.0
    严重不足 (Severely deficient): ESI < 0.5
    """
    cats = pd.cut(
        esi,
        bins=[-np.inf, 0.5, 1.0, 1.5, np.inf],
        labels=["Severely deficient", "Deficient", "Moderate", "Sufficient"],
    )
    return cats


# ══════════════════════════════════════════════════════════════
# Table E2: Pairwise Spearman ρ
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TABLE E2: Pairwise Spearman Rank Correlation (ρ) of ESI Across Scenarios")
print("=" * 70)

results_e2 = []
for period_name, df in [("Daytime", day), ("Nighttime", night)]:
    print(f"\n--- {period_name} ---")
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    for s1, s2 in pairs:
        col1 = f"Ai_{s1}"
        col2 = f"Ai_{s2}"
        rho, pval = spearmanr(df[col1], df[col2])
        label = f"Scenario {s1} vs {s2}"
        print(f"  {label}: ρ = {rho:.4f}, p = {pval:.2e}, n = {len(df)}")
        results_e2.append({
            "Period": period_name,
            "Comparison": label,
            "Spearman_rho": round(rho, 4),
            "p_value": f"{pval:.2e}",
            "n": len(df),
        })

df_e2 = pd.DataFrame(results_e2)
df_e2.to_csv("/home/claude/appendix_E/table_E2_spearman.csv", index=False)
print(f"\nSaved: table_E2_spearman.csv")


# ══════════════════════════════════════════════════════════════
# Table E3a: Categorical Spatial Agreement
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE E3a: Categorical Agreement Rate (%)")
print("=" * 70)

results_e3a = []
for period_name, df in [("Daytime", day), ("Nighttime", night)]:
    print(f"\n--- {period_name} ---")
    cats = {}
    for sc_name, col in scenarios.items():
        cats[sc_name] = categorize_esi(df[col])

    pairs = [("A (d₀=3000m)", "B (d₀=2000m)"),
             ("A (d₀=3000m)", "C (d₀=1000m)"),
             ("B (d₀=2000m)", "C (d₀=1000m)")]
    for s1, s2 in pairs:
        agree = (cats[s1] == cats[s2]).sum()
        rate = agree / len(df) * 100
        print(f"  {s1} vs {s2}: Agreement = {agree}/{len(df)} = {rate:.1f}%")
        results_e3a.append({
            "Period": period_name,
            "Comparison": f"{s1} vs {s2}",
            "Agreement_count": agree,
            "Total": len(df),
            "Agreement_rate_pct": round(rate, 1),
        })

df_e3a = pd.DataFrame(results_e3a)
df_e3a.to_csv("/home/claude/appendix_E/table_E3a_categorical_agreement.csv", index=False)
print(f"\nSaved: table_E3a_categorical_agreement.csv")


# ══════════════════════════════════════════════════════════════
# Table E3b: District-level Mean ESI & Tier Distribution
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TABLE E3b: District-Level Mean ESI Across Scenarios")
print("=" * 70)

results_e3b = []
for period_name, df in [("Daytime", day), ("Nighttime", night)]:
    print(f"\n--- {period_name} ---")
    print(f"  {'District':<12}", end="")
    for sc in scenarios:
        print(f"  {sc:<16}", end="")
    print(f"  {'Rank A→B→C'}")

    for dist_cn, dist_en in district_map.items():
        mask = df["所属区"] == dist_cn
        row = {"Period": period_name, "District_CN": dist_cn, "District_EN": dist_en}
        vals = []
        for sc_name, col in scenarios.items():
            mean_val = df.loc[mask, col].mean()
            row[f"Mean_ESI_{sc_name}"] = round(mean_val, 4)
            vals.append(mean_val)

        # Rank for each scenario
        ranks = []
        for sc_name, col in scenarios.items():
            district_means = df.groupby("所属区")[col].mean()
            rank = district_means.rank(ascending=False)[dist_cn]
            ranks.append(int(rank))
        row["Rank_A"] = ranks[0]
        row["Rank_B"] = ranks[1]
        row["Rank_C"] = ranks[2]
        row["Rank_change"] = "Unchanged" if len(set(ranks)) == 1 else f"{ranks[0]}→{ranks[1]}→{ranks[2]}"

        print(f"  {dist_en:<12}", end="")
        for sc_name in scenarios:
            print(f"  {row[f'Mean_ESI_{sc_name}']:<16.4f}", end="")
        print(f"  {row['Rank_change']}")

        results_e3b.append(row)

df_e3b = pd.DataFrame(results_e3b)
df_e3b.to_csv("/home/claude/appendix_E/table_E3b_district_mean_ESI.csv", index=False)
print(f"\nSaved: table_E3b_district_mean_ESI.csv")


# ══════════════════════════════════════════════════════════════
# Additional: Tier distribution per scenario
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUPPLEMENTARY: ESI Tier Distribution by Scenario (%)")
print("=" * 70)

results_tier = []
for period_name, df in [("Daytime", day), ("Nighttime", night)]:
    print(f"\n--- {period_name} ---")
    for sc_name, col in scenarios.items():
        cats = categorize_esi(df[col])
        dist = cats.value_counts(normalize=True).sort_index() * 100
        print(f"  {sc_name}:")
        for tier, pct in dist.items():
            print(f"    {tier}: {pct:.1f}%")
            results_tier.append({
                "Period": period_name,
                "Scenario": sc_name,
                "Tier": tier,
                "Percentage": round(pct, 1),
            })

df_tier = pd.DataFrame(results_tier)
df_tier.to_csv("/home/claude/appendix_E/supplementary_tier_distribution.csv", index=False)


# ══════════════════════════════════════════════════════════════
# Summary statistics
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: Key Statistics Across Scenarios")
print("=" * 70)

for period_name, df in [("Daytime", day), ("Nighttime", night)]:
    print(f"\n--- {period_name} ---")
    for sc_name, col in scenarios.items():
        esi = df[col]
        print(f"  {sc_name}:")
        print(f"    Mean={esi.mean():.4f}, Median={esi.median():.4f}, "
              f"Std={esi.std():.4f}")
        print(f"    ESI<1.0: {(esi < 1.0).sum()} ({(esi < 1.0).mean()*100:.1f}%)")
        print(f"    ESI<0.5: {(esi < 0.5).sum()} ({(esi < 0.5).mean()*100:.1f}%)")

print("\n✅ Appendix E analysis complete!")
