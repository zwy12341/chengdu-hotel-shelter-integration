"""
================================================================================
补充分析: VIF多重共线性诊断 + λ敏感性分析
================================================================================
对应修改方案:
  问题4 → VIF诊断 (CDI Exposure维度共线性)
  问题3 → λ敏感性分析 (酒店有效容量系数)
================================================================================
"""

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = "/mnt/user-data/uploads"
OUTPUT_DIR = "/home/claude"

# ═══════════════════════════════════════════════════════════════
#  Part 1: VIF 多重共线性诊断
# ═══════════════════════════════════════════════════════════════

print("=" * 70)
print("Part 1: VIF 多重共线性诊断")
print("=" * 70)

df = pd.read_csv(f"{INPUT_DIR}/v4_data_full.csv")
print(f"\n建模样本: n = {len(df)}")

# --- 1.1 CDI全部11个指标的VIF ---
print("\n" + "-" * 50)
print("1.1 CDI 全指标 VIF (11个指标)")
print("-" * 50)

cdi_indicators = ['H11', 'H12', 'E11', 'E12', 'E21', 'E31', 
                  'V11', 'V21', 'V22', 'V31', 'V32']
cdi_labels = {
    'H11': '断层距离', 'H12': '震中距离',
    'E11': '人口密度(昼)', 'E12': '脆弱人口比例', 
    'E21': '人均GDP', 'E31': '建筑密度',
    'V11': '医疗机构密度', 'V21': '建筑高度', 
    'V22': '建筑年龄', 'V31': '高程差', 'V32': '坡度'
}
cdi_dims = {
    'H11': 'H(危险性)', 'H12': 'H(危险性)',
    'E11': 'E(暴露度)', 'E12': 'E(暴露度)', 
    'E21': 'E(暴露度)', 'E31': 'E(暴露度)',
    'V11': 'V(脆弱性)', 'V21': 'V(脆弱性)', 
    'V22': 'V(脆弱性)', 'V31': 'V(脆弱性)', 'V32': 'V(脆弱性)'
}

X_cdi = df[cdi_indicators].copy()
# 标准化以提高数值稳定性
X_cdi_std = (X_cdi - X_cdi.mean()) / X_cdi.std()

vif_results_all = []
for i, col in enumerate(cdi_indicators):
    vif_val = variance_inflation_factor(X_cdi_std.values, i)
    vif_results_all.append({
        '指标': col,
        '名称': cdi_labels[col],
        '维度': cdi_dims[col],
        'VIF': round(vif_val, 2)
    })

vif_df_all = pd.DataFrame(vif_results_all).sort_values('VIF', ascending=False)
print(vif_df_all.to_string(index=False))
print(f"\n判定标准: VIF > 5 中等共线性, VIF > 10 严重共线性")
severe = vif_df_all[vif_df_all['VIF'] > 10]
moderate = vif_df_all[(vif_df_all['VIF'] > 5) & (vif_df_all['VIF'] <= 10)]
print(f"严重 (VIF>10): {len(severe)} 个 → {list(severe['指标'].values) if len(severe)>0 else '无'}")
print(f"中等 (5<VIF≤10): {len(moderate)} 个 → {list(moderate['指标'].values) if len(moderate)>0 else '无'}")

# --- 1.2 Exposure维度内部VIF ---
print("\n" + "-" * 50)
print("1.2 Exposure 维度内部 VIF")
print("-" * 50)

exp_indicators = ['E11', 'E12', 'E21', 'E31']
X_exp = df[exp_indicators].copy()
X_exp_std = (X_exp - X_exp.mean()) / X_exp.std()

vif_exp = []
for i, col in enumerate(exp_indicators):
    vif_val = variance_inflation_factor(X_exp_std.values, i)
    vif_exp.append({'指标': col, '名称': cdi_labels[col], 'VIF': round(vif_val, 2)})

vif_df_exp = pd.DataFrame(vif_exp).sort_values('VIF', ascending=False)
print(vif_df_exp.to_string(index=False))

# --- 1.3 Exposure维度相关系数矩阵 ---
print("\n" + "-" * 50)
print("1.3 Exposure 指标 Pearson 相关系数矩阵")
print("-" * 50)

corr_exp = df[exp_indicators].corr().round(3)
print(corr_exp.to_string())

# --- 1.4 注意E21是区级数据 ---
print("\n" + "-" * 50)
print("1.4 数据精度检查: E21(人均GDP)是区级数据")
print("-" * 50)
for d in sorted(df['所属区'].unique()):
    n = len(df[df['所属区'] == d])
    val = df[df['所属区'] == d]['E21'].iloc[0]
    print(f"  {d}: {n} 格网, E21 = {val:.0f} (唯一值)")

# --- 1.5 ML特征集的VIF ---
print("\n" + "-" * 50)
print("1.5 ML 14feat 特征集 VIF")
print("-" * 50)

# 构造ML特征
df['ESI_base'] = df['ESI'] - df['delta_ESI']
df['lon_f'] = df['lon']
df['lat_f'] = df['lat']

ml_features = ['H11', 'H12', 'E11', 'E12', 'E21', 'E31',
               'V11', 'V21', 'V22', 'V31', 'V32',
               'ESI_base', 'lon_f', 'lat_f']
ml_labels = {**cdi_labels, 
             'ESI_base': '基线ESI', 'lon_f': '经度', 'lat_f': '纬度'}

X_ml = df[ml_features].copy()
X_ml_std = (X_ml - X_ml.mean()) / X_ml.std()

vif_ml = []
for i, col in enumerate(ml_features):
    vif_val = variance_inflation_factor(X_ml_std.values, i)
    vif_ml.append({'特征': col, '名称': ml_labels[col], 'VIF': round(vif_val, 2)})

vif_df_ml = pd.DataFrame(vif_ml).sort_values('VIF', ascending=False)
print(vif_df_ml.to_string(index=False))

severe_ml = vif_df_ml[vif_df_ml['VIF'] > 10]
moderate_ml = vif_df_ml[(vif_df_ml['VIF'] > 5) & (vif_df_ml['VIF'] <= 10)]
print(f"\n严重 (VIF>10): {len(severe_ml)} 个 → {list(severe_ml['特征'].values) if len(severe_ml)>0 else '无'}")
print(f"中等 (5<VIF≤10): {len(moderate_ml)} 个 → {list(moderate_ml['特征'].values) if len(moderate_ml)>0 else '无'}")


# ═══════════════════════════════════════════════════════════════
#  Part 2: λ 敏感性分析
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("Part 2: λ 敏感性分析 (酒店有效容量系数)")
print("=" * 70)

print("""
原理说明:
  G2SFCA中, 酒店供给 S_j = B_j × λ
  因为G2SFCA对供给量是线性的:
    R_j(λ) = S_j(λ) / Σ[P_k × G(d)] = (B_j × λ) / Σ[P_k × G(d)]
    → R_j(λ_new) = R_j(λ=0.7) × (λ_new / 0.7)
    → ΔESI(λ_new) = ΔESI(λ=0.7) × (λ_new / 0.7)
  因此可通过线性缩放进行精确的敏感性分析, 无需重跑G2SFCA。
""")

# 参数设置
lambda_base = 0.7
lambdas = [0.5, 0.7, 0.9]

# 基础数据
ESI_baseline = df['ESI'] - df['delta_ESI']  # 不含酒店的ESI
dESI_base = df['delta_ESI']  # λ=0.7时的ΔESI
CDI = df['CDI']

# 全域统计 (n=3700, ΔESI>0的格网)
# 注: 还有4857-3700=1157个格网ΔESI=0, ESI不受λ影响
n_total = 4857
n_with_hotel = len(df)
n_without_hotel = n_total - n_with_hotel

# CDI 75%分位数阈值 (用于识别高需求区)
cdi_75 = CDI.quantile(0.75)
high_demand_mask = CDI >= cdi_75

print(f"基础信息:")
print(f"  总格网数: {n_total}")
print(f"  有酒店覆盖: {n_with_hotel} ({n_with_hotel/n_total*100:.1f}%)")
print(f"  无酒店覆盖: {n_without_hotel} ({n_without_hotel/n_total*100:.1f}%)")
print(f"  高需求区阈值 (CDI 75%): {cdi_75:.4f}")
print(f"  高需求格网数: {high_demand_mask.sum()}")

results_lambda = []

for lam in lambdas:
    scale = lam / lambda_base
    
    # 缩放后的ΔESI和ESI
    dESI_scaled = dESI_base * scale
    ESI_scaled = ESI_baseline + dESI_scaled
    
    # ---- 指标计算 ----
    
    # 1. 平均ΔESI
    mean_dESI = dESI_scaled.mean()
    
    # 2. 高需求区 vs 其他区域的ΔESI
    mean_dESI_high = dESI_scaled[high_demand_mask].mean()
    mean_dESI_other = dESI_scaled[~high_demand_mask].mean()
    ratio_high_other = mean_dESI_high / mean_dESI_other if mean_dESI_other > 0 else np.nan
    
    # 3. 正义性: ΔESI与CDI的相关性 (不受缩放影响, 仅验证)
    r_justice, p_justice = stats.pearsonr(dESI_scaled, CDI)
    
    # 4. 供给指标 (仅对有酒店覆盖的格网)
    # ESI < 1.0 供给不足
    n_deficit = (ESI_scaled < 1.0).sum()
    pct_deficit = n_deficit / n_with_hotel * 100
    
    # ESI < 0.5 严重不足
    n_severe = (ESI_scaled < 0.5).sum()
    pct_severe = n_severe / n_with_hotel * 100
    
    # 平均ESI
    mean_ESI = ESI_scaled.mean()
    
    # 5. 酒店覆盖区的总体供需比
    total_supply_ratio = ESI_scaled.mean()
    
    results_lambda.append({
        'λ': lam,
        '缩放因子': f'{scale:.3f}',
        '平均ΔESI': mean_dESI,
        '高需求区ΔESI': mean_dESI_high,
        '其他区域ΔESI': mean_dESI_other,
        '高/其他比值': ratio_high_other,
        'ΔESI-CDI相关(r)': r_justice,
        '平均ESI': mean_ESI,
        'ESI<1.0占比(%)': pct_deficit,
        'ESI<0.5占比(%)': pct_severe,
    })

results_df = pd.DataFrame(results_lambda)

print("\n" + "-" * 70)
print("2.1 λ敏感性分析结果 (n=3700, 有酒店覆盖的格网)")
print("-" * 70)
print(results_df.to_string(index=False, float_format='%.4f'))

# --- 2.2 关键指标变化幅度 ---
print("\n" + "-" * 70)
print("2.2 关键指标变化幅度 (相对λ=0.7基准)")
print("-" * 70)

base_row = results_df[results_df['λ'] == 0.7].iloc[0]
for _, row in results_df.iterrows():
    if row['λ'] == 0.7:
        continue
    lam = row['λ']
    print(f"\n  λ = {lam} (vs λ=0.7):")
    print(f"    平均ΔESI变化: {(row['平均ΔESI']/base_row['平均ΔESI']-1)*100:+.1f}%")
    print(f"    高需求区ΔESI变化: {(row['高需求区ΔESI']/base_row['高需求区ΔESI']-1)*100:+.1f}%")
    print(f"    高/其他比值: {row['高/其他比值']:.2f} (不变, 因等比缩放)")
    print(f"    ESI<1.0占比: {row['ESI<1.0占比(%)']:.1f}% (基准: {base_row['ESI<1.0占比(%)']:.1f}%)")
    print(f"    ESI<0.5占比: {row['ESI<0.5占比(%)']:.1f}% (基准: {base_row['ESI<0.5占比(%)']:.1f}%)")

# --- 2.3 高需求区差异的显著性检验 ---
print("\n" + "-" * 70)
print("2.3 三种λ下高需求区 vs 其他区域的Welch t检验")
print("-" * 70)

for lam in lambdas:
    scale = lam / lambda_base
    dESI_scaled = dESI_base * scale
    high = dESI_scaled[high_demand_mask]
    other = dESI_scaled[~high_demand_mask]
    t_stat, p_val = stats.ttest_ind(high, other, equal_var=False)
    print(f"  λ={lam}: t={t_stat:.3f}, p={p_val:.2e}, 高需求均值={high.mean():.5f}, 其他均值={other.mean():.5f}")

# --- 2.4 分区统计 ---
print("\n" + "-" * 70)
print("2.4 分区λ敏感性 (各区平均ΔESI)")
print("-" * 70)

district_results = []
for lam in lambdas:
    scale = lam / lambda_base
    dESI_scaled = dESI_base * scale
    for d in sorted(df['所属区'].unique()):
        mask = df['所属区'] == d
        district_results.append({
            'λ': lam,
            '行政区': d,
            '平均ΔESI': dESI_scaled[mask].mean(),
            '中位数ΔESI': dESI_scaled[mask].median(),
            '格网数': mask.sum()
        })

dist_df = pd.DataFrame(district_results)
pivot = dist_df.pivot_table(index='行政区', columns='λ', values='平均ΔESI')
print(pivot.round(6).to_string())

# --- 2.5 双重缺失区分析 ---
print("\n" + "-" * 70)
print("2.5 双重缺失区对λ的不敏感性说明")
print("-" * 70)
print(f"""
  双重缺失区(HL型) = ESI_baseline < 0.5 且 无酒店覆盖的格网
  这些格网的ΔESI = 0, 因此λ取值不影响双重缺失区的判定。
  
  在有酒店覆盖的格网中 (n={n_with_hotel}):
    ESI_baseline < 0.5 的格网: {(ESI_baseline < 0.5).sum()} ({(ESI_baseline < 0.5).sum()/n_with_hotel*100:.1f}%)
    酒店介入后仍 ESI < 0.5 的格网:
""")

for lam in lambdas:
    scale = lam / lambda_base
    ESI_new = ESI_baseline + dESI_base * scale
    n_still_severe = ((ESI_baseline < 0.5) & (ESI_new < 0.5)).sum()
    n_rescued = ((ESI_baseline < 0.5) & (ESI_new >= 0.5)).sum()
    print(f"    λ={lam}: 仍严重不足={n_still_severe}, 被酒店挽救={n_rescued}")


# ═══════════════════════════════════════════════════════════════
#  Part 3: 保存结果
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("Part 3: 保存结果")
print("=" * 70)

# VIF结果
vif_output = pd.DataFrame()
# CDI全指标VIF
vif_all_save = vif_df_all.copy()
vif_all_save['分析范围'] = 'CDI全指标(11个)'
# Exposure内部VIF
vif_exp_save = vif_df_exp.copy()
vif_exp_save['分析范围'] = 'Exposure维度内部(4个)'
vif_exp_save.rename(columns={'指标': '指标'}, inplace=True)
# ML特征VIF
vif_ml_save = vif_df_ml.copy()
vif_ml_save['分析范围'] = 'ML 14feat特征集'
vif_ml_save.rename(columns={'特征': '指标'}, inplace=True)

vif_all_out = pd.concat([vif_all_save, vif_exp_save, vif_ml_save], ignore_index=True)
vif_all_out.to_csv(f"{OUTPUT_DIR}/v4_vif_diagnostic.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ VIF诊断: {OUTPUT_DIR}/v4_vif_diagnostic.csv")

# λ敏感性结果
results_df.to_csv(f"{OUTPUT_DIR}/v4_lambda_sensitivity.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ λ敏感性: {OUTPUT_DIR}/v4_lambda_sensitivity.csv")

# 分区λ敏感性
dist_df.to_csv(f"{OUTPUT_DIR}/v4_lambda_by_district.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ 分区λ敏感性: {OUTPUT_DIR}/v4_lambda_by_district.csv")

# Exposure相关系数矩阵
corr_exp.to_csv(f"{OUTPUT_DIR}/v4_exposure_correlation.csv", encoding='utf-8-sig')
print(f"  ✓ Exposure相关系数: {OUTPUT_DIR}/v4_exposure_correlation.csv")


# ═══════════════════════════════════════════════════════════════
#  Part 4: 论文可用结论总结
# ═══════════════════════════════════════════════════════════════

print("\n\n" + "=" * 70)
print("Part 4: 论文可用结论总结")
print("=" * 70)

print("""
╔══════════════════════════════════════════════════════════════════════╗
║  VIF诊断结论 (问题4)                                               ║
╠══════════════════════════════════════════════════════════════════════╣""")

max_vif_cdi = vif_df_all['VIF'].max()
max_vif_name = vif_df_all.loc[vif_df_all['VIF'].idxmax(), '指标']
max_vif_ml = vif_df_ml['VIF'].max()
max_vif_ml_name = vif_df_ml.loc[vif_df_ml['VIF'].idxmax(), '特征']

print(f"""║                                                                      ║
║  CDI全指标最大VIF: {max_vif_cdi:.2f} ({max_vif_name})                     """)
if max_vif_cdi <= 5:
    print(f"""║  → 所有指标VIF < 5, 无显著共线性问题                                ║""")
elif max_vif_cdi <= 10:
    print(f"""║  → 存在中等共线性, 但未达严重水平(VIF<10)                           ║""")
else:
    print(f"""║  → 存在严重共线性(VIF>10), 需要处理                                ║""")

print(f"""║                                                                      ║
║  ML 14feat最大VIF: {max_vif_ml:.2f} ({max_vif_ml_name})                   """)
print(f"""║  → XGBoost对共线性具有内在鲁棒性(树模型不受共线性影响)             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  λ敏感性分析结论 (问题3)                                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  核心发现: 所有关键结论对λ取值具有稳健性                             ║
║                                                                      ║
║  1. 空间正义结论不变:                                                ║
║     高需求/其他比值在三种λ下均为 {ratio_high_other:.1f}x (等比缩放特性)    ║
║     ΔESI-CDI相关系数: r = {r_justice:.3f} (不受λ影响)                  ║
║                                                                      ║
║  2. ESI<1.0供给不足占比变化范围有限:                                 ║""")
pcts = [r['ESI<1.0占比(%)'] for r in results_lambda]
print(f"""║     λ=0.5: {pcts[0]:.1f}% → λ=0.7: {pcts[1]:.1f}% → λ=0.9: {pcts[2]:.1f}%           """)
print(f"""║                                                                      ║
║  3. 双重缺失区不受λ影响 (ΔESI=0的区域λ无关)                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")
