"""
================================================================================
成都市应急避难设施供需分析 - 完整计算脚本
================================================================================
论文: 平急两用导向下城市酒店的震灾避难服务效能与空间正义

功能:
  Step 1: 数据一致性验证
  Step 2: G2SFCA基线可达性复现 (β=0.5, Dai 2010)
  Step 3: 含酒店弹性供给可达性计算
  Step 4: 供需空间关系与正义性评价
    4.1 双变量四象限分析 (CDI-ESI)
    4.2 缺口-资源双变量耦合分析
    4.3 空间正义验证

输入文件 (全部放在 INPUT_DIR):
  - demand_grids_daytime_CLEAN.csv     白天需求格网
  - demand_grids_nighttime_CLEAN.csv   夜间需求格网
  - supply_facilities_CLEAN.csv        基准供给设施(1054处)
  - hotel_facilities_filtered.csv      筛选后酒店(2264家)
  - grid_accessibility_daytime.csv     基线可达性(方案A/B/C)
  - grid_accessibility_nighttime.csv   基线可达性(夜间)
  - grid_accessibility_with_hotel_daytime.csv   含酒店可达性(白天)
  - grid_accessibility_with_hotel_nighttime.csv 含酒店可达性(夜间)
  - quadrant_analysis_data-CDI__ESI_四象限分类.csv  CDI数据

输出文件:
  - supply_demand_analysis_results.csv  完整分析结果(每格网)
  - analysis_summary.txt               分析报告

依赖: pandas, numpy, scipy
================================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
import os, time, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 路径配置 (根据实际情况修改)
# ══════════════════════════════════════════════════════
INPUT_DIR  = "/mnt/user-data/uploads"
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# 模型参数
# ══════════════════════════════════════════════════════
D0_BASE  = 3000    # 基准设施服务半径 (m)
D0_HOTEL = 1000    # 酒店服务半径 (m)
BETA     = 0.5     # Gaussian衰减参数 (Dai 2010)
A_PER    = 2.0     # 人均避难面积 (m²/人)
LAMBDA_HOTEL = 0.7 # 酒店折算系数 (基准)

# 方案A折算系数
LAMBDA_A = {
    '公园绿地': 0.60, '体育场馆': 0.65,
    '大学/高等院校': 0.60, '中学': 0.60, '小学': 0.60, '其他学校': 0.60,
}

# 坐标转米 (成都 ~30.67°N)
LAT_TO_M = 111000
LON_TO_M = 95400


# ══════════════════════════════════════════════════════
# 核心函数
# ══════════════════════════════════════════════════════
def to_meters(lon, lat):
    """经纬度→近似米坐标"""
    return lon * LON_TO_M, lat * LAT_TO_M

def gaussian_decay(d, d0, beta=BETA):
    """
    Dai (2010) 高斯距离衰减函数 (标准减法形式)
    
    G(d, d₀) = [exp(-(d/d₀)²/β) - exp(-1/β)] / [1 - exp(-1/β)]
    
    边界: G(0)=1, G(d₀)=0
    """
    r = d / d0
    num = np.exp(-r**2 / beta) - np.exp(-1.0 / beta)
    den = 1.0 - np.exp(-1.0 / beta)
    result = np.where(d <= d0, num / den, 0.0)
    return np.maximum(result, 0.0)

def run_g2sfca(grid_xy, grid_pop, supply_xy, supply_cap, d0, tag=""):
    """
    执行 Gaussian 2SFCA 两步搜索
    
    Parameters:
        grid_xy:    格网坐标 (N, 2) 米
        grid_pop:   格网人口 (N,)
        supply_xy:  设施坐标 (M, 2) 米
        supply_cap: 设施容量 (M,)
        d0:         服务半径 (m)
        tag:        标签(打印用)
    Returns:
        Ai: 每个格网的可达性指数 (N,)
    """
    N, M = len(grid_xy), len(supply_xy)
    print(f"    [{tag}] 格网={N}, 设施={M}, d₀={d0}m")
    
    t0 = time.time()
    grid_tree = cKDTree(grid_xy)
    
    # Step 1: 计算每个设施的供需比 R_j
    R_j = np.zeros(M)
    idx_grids = grid_tree.query_ball_point(supply_xy, r=d0 * 1.02)
    for j in range(M):
        nb = np.array(idx_grids[j], dtype=int)
        if len(nb) == 0:
            continue
        dist = np.linalg.norm(grid_xy[nb] - supply_xy[j], axis=1)
        w = gaussian_decay(dist, d0)
        denom = np.dot(grid_pop[nb], w)
        R_j[j] = supply_cap[j] / denom if denom > 1e-10 else 0.0
    
    # Step 2: 计算每个格网的可达性 A_i
    supply_tree = cKDTree(supply_xy)
    Ai = np.zeros(N)
    idx_supply = supply_tree.query_ball_point(grid_xy, r=d0 * 1.02)
    for i in range(N):
        nb = np.array(idx_supply[i], dtype=int)
        if len(nb) == 0:
            continue
        dist = np.linalg.norm(supply_xy[nb] - grid_xy[i], axis=1)
        w = gaussian_decay(dist, d0)
        Ai[i] = np.dot(R_j[nb], w)
    
    print(f"    [{tag}] 完成 ({time.time()-t0:.1f}s), Ai mean={Ai.mean():.4f}")
    return Ai

def classify_level(ai):
    """供需分级: 严重不足/供给不足/基本平衡/供给充足"""
    if ai >= 1.5:   return '供给充足'
    elif ai >= 1.0: return '基本平衡'
    elif ai >= 0.5: return '供给不足'
    else:            return '严重不足'


# ══════════════════════════════════════════════════════
# STEP 1: 数据加载与验证
# ══════════════════════════════════════════════════════
def step1_load_and_verify():
    print("\n" + "=" * 70)
    print("STEP 1: 数据加载与一致性验证")
    print("=" * 70)
    
    demand_day = pd.read_csv(f"{INPUT_DIR}/demand_grids_daytime_CLEAN.csv")
    demand_night = pd.read_csv(f"{INPUT_DIR}/demand_grids_nighttime_CLEAN.csv")
    supply = pd.read_csv(f"{INPUT_DIR}/supply_facilities_CLEAN.csv")
    hotels = pd.read_csv(f"{INPUT_DIR}/hotel_facilities_filtered.csv")
    acc_day = pd.read_csv(f"{INPUT_DIR}/grid_accessibility_daytime.csv")
    acc_night = pd.read_csv(f"{INPUT_DIR}/grid_accessibility_nighttime.csv")
    acc_hotel_day = pd.read_csv(f"{INPUT_DIR}/grid_accessibility_with_hotel_daytime.csv")
    acc_hotel_night = pd.read_csv(f"{INPUT_DIR}/grid_accessibility_with_hotel_nighttime.csv")
    
    # CDI数据
    cdi_path = f"{INPUT_DIR}/quadrant_analysis_data-CDI__ESI_四象限分类.csv"
    cdi_df = pd.read_csv(cdi_path) if os.path.exists(cdi_path) else None
    
    # 验证
    checks = []
    checks.append(("格网数一致", len(demand_day) == len(acc_day) == len(acc_hotel_day) == 4857))
    checks.append(("供给设施数", len(supply) == 1054))
    checks.append(("酒店数", len(hotels) == 2264))
    
    m = demand_day[['GRID_ID','population']].merge(acc_day[['GRID_ID','population']], on='GRID_ID', suffixes=('_d','_a'))
    checks.append(("人口一致", np.allclose(m['population_d'], m['population_a'])))
    
    m2 = acc_day[['GRID_ID','Ai_A']].merge(acc_hotel_day[['GRID_ID','Ai_base']], on='GRID_ID')
    checks.append(("基线ESI一致", np.allclose(m2['Ai_A'], m2['Ai_base'])))
    checks.append(("ΔESI≥0", (acc_hotel_day['Ai_improve'] >= 0).all()))
    
    supply['capacity_A'] = supply.apply(
        lambda r: r['total_area_m2'] * LAMBDA_A.get(r['facility_subtype'], 0.60) / A_PER, axis=1)
    checks.append(("总容量=9474161", abs(supply['capacity_A'].sum() - 9474161) < 1))
    
    all_pass = True
    for name, result in checks:
        status = "✓" if result else "✗"
        if not result: all_pass = False
        print(f"  {status} {name}")
    
    print(f"\n  总体: {'全部通过' if all_pass else '存在问题!'}")
    
    return {
        'demand_day': demand_day, 'demand_night': demand_night,
        'supply': supply, 'hotels': hotels,
        'acc_day': acc_day, 'acc_night': acc_night,
        'acc_hotel_day': acc_hotel_day, 'acc_hotel_night': acc_hotel_night,
        'cdi_df': cdi_df,
    }


# ══════════════════════════════════════════════════════
# STEP 2: G2SFCA基线可达性复现验证
# ══════════════════════════════════════════════════════
def step2_reproduce_baseline(data):
    print("\n" + "=" * 70)
    print("STEP 2: G2SFCA基线可达性复现验证 (方案A, 白天)")
    print("=" * 70)
    
    demand = data['demand_day']
    supply = data['supply']
    acc_day = data['acc_day']
    
    # 坐标转米
    gx, gy = to_meters(demand['lon'].values, demand['lat'].values)
    grid_xy = np.column_stack([gx, gy])
    grid_pop = demand['population'].values.astype(float)
    
    sx, sy = to_meters(supply['lon'].values, supply['lat'].values)
    supply_xy = np.column_stack([sx, sy])
    supply_cap = supply['capacity_A'].values.astype(float)
    
    Ai_repro = run_g2sfca(grid_xy, grid_pop, supply_xy, supply_cap, D0_BASE, "基线复现")
    
    # 与原始数据对比
    merged = demand[['GRID_ID']].copy()
    merged['Ai_repro'] = Ai_repro
    merged = merged.merge(acc_day[['GRID_ID','Ai_A']], on='GRID_ID')
    
    corr = merged['Ai_repro'].corr(merged['Ai_A'])
    rmse = np.sqrt(((merged['Ai_repro'] - merged['Ai_A'])**2).mean())
    max_diff = abs(merged['Ai_repro'] - merged['Ai_A']).max()
    
    print(f"\n  复现 vs 原始:")
    print(f"    Pearson r = {corr:.6f}")
    print(f"    RMSE = {rmse:.6f}")
    print(f"    最大偏差 = {max_diff:.6f}")
    print(f"    复现均值 = {Ai_repro.mean():.4f} vs 原始均值 = {acc_day['Ai_A'].mean():.4f}")
    
    if corr > 0.999:
        print("    ✓ 复现高度一致")
    elif corr > 0.99:
        print("    ○ 复现基本一致(微小数值差异)")
    else:
        print("    ✗ 存在差异，需排查")
    
    return Ai_repro


# ══════════════════════════════════════════════════════
# STEP 3: 含酒店弹性供给可达性计算
# ══════════════════════════════════════════════════════
def step3_hotel_accessibility(data):
    print("\n" + "=" * 70)
    print("STEP 3: 酒店弹性供给ΔESI计算 (λ=0.7, d₀=1000m)")
    print("=" * 70)
    
    demand = data['demand_day']
    hotels = data['hotels']
    acc_hotel = data['acc_hotel_day']
    
    # 格网坐标
    gx, gy = to_meters(demand['lon'].values, demand['lat'].values)
    grid_xy = np.column_stack([gx, gy])
    grid_pop = demand['population'].values.astype(float)
    
    # 酒店坐标与容量
    hx, hy = to_meters(hotels['lon'].values, hotels['lat'].values)
    hotel_xy = np.column_stack([hx, hy])
    hotel_cap = hotels['capacity_0.7'].values.astype(float)
    
    delta_esi = run_g2sfca(grid_xy, grid_pop, hotel_xy, hotel_cap, D0_HOTEL, "酒店ΔESI")
    
    # 与原始数据对比
    merged = demand[['GRID_ID']].copy()
    merged['delta_repro'] = delta_esi
    merged = merged.merge(acc_hotel[['GRID_ID','Ai_improve']], on='GRID_ID')
    
    corr = merged['delta_repro'].corr(merged['Ai_improve'])
    rmse = np.sqrt(((merged['delta_repro'] - merged['Ai_improve'])**2).mean())
    
    print(f"\n  复现 vs 原始 ΔESI:")
    print(f"    Pearson r = {corr:.6f}")
    print(f"    RMSE = {rmse:.6f}")
    print(f"    复现均值 = {delta_esi.mean()*1000:.3f}×10⁻³")
    print(f"    原始均值 = {acc_hotel['Ai_improve'].mean()*1000:.3f}×10⁻³")
    
    print(f"\n  酒店资源统计:")
    print(f"    酒店数: {len(hotels)}")
    print(f"    总容量(λ=0.7): {hotel_cap.sum():,.0f}人")
    print(f"    ΔESI>0格网: {(delta_esi>0).sum()} ({(delta_esi>0).mean()*100:.1f}%)")
    print(f"    ΔESI=0格网: {(delta_esi==0).sum()} ({(delta_esi==0).mean()*100:.1f}%)")
    
    return delta_esi


# ══════════════════════════════════════════════════════
# STEP 4: 供需空间关系与正义性评价
# ══════════════════════════════════════════════════════
def step4_supply_demand_analysis(data):
    print("\n" + "=" * 70)
    print("STEP 4: 供需空间关系与正义性评价")
    print("=" * 70)
    
    acc_day = data['acc_day']
    acc_hotel = data['acc_hotel_day']
    cdi_df = data['cdi_df']
    hotels_density = None
    
    # 构建核心分析表
    core = acc_day[['GRID_ID','population','Ai_A']].copy()
    core.rename(columns={'Ai_A': 'ESI_base'}, inplace=True)
    
    # 合并酒店ΔESI
    core = core.merge(acc_hotel[['GRID_ID','Ai_improve','Ai_hotel_0.7']], on='GRID_ID')
    core.rename(columns={'Ai_improve': 'delta_ESI', 'Ai_hotel_0.7': 'ESI_with_hotel'}, inplace=True)
    
    # 合并CDI (如果有)
    if cdi_df is not None and 'CDI' in cdi_df.columns:
        core = core.merge(cdi_df[['GRID_ID','CDI','所属区']], on='GRID_ID', how='left')
        has_cdi = True
    else:
        has_cdi = False
        print("  ⚠ 未找到CDI数据，跳过四象限和正义性分析")
    
    # 合并酒店密度 (如果有)
    hd_path = f"{INPUT_DIR}/hotel_kernel_density_500m-酒店核密度网格值.csv"
    if os.path.exists(hd_path):
        hd = pd.read_csv(hd_path)
        core = core.merge(hd[['GRID_ID','hotel_density']], on='GRID_ID', how='left')
        core['hotel_density'] = core['hotel_density'].fillna(0)
    
    N = len(core)
    
    # ── 4.0 基线供需统计 ──
    print(f"\n  ─── 4.0 基线供需统计 (方案A, β=0.5) ───")
    print(f"  格网总数: {N}")
    print(f"  ESI均值: {core['ESI_base'].mean():.4f}")
    print(f"  ESI中位数: {core['ESI_base'].median():.4f}")
    
    for level, lo, hi in [('严重不足',0,0.5),('供给不足',0.5,1.0),('基本平衡',1.0,1.5),('供给充足',1.5,999)]:
        n = ((core['ESI_base']>=lo) & (core['ESI_base']<hi)).sum()
        pop = core[(core['ESI_base']>=lo) & (core['ESI_base']<hi)]['population'].sum()
        print(f"    {level}: {n}格网 ({n/N*100:.1f}%), 人口{pop/10000:.1f}万")
    
    if not has_cdi:
        core.to_csv(f"{OUTPUT_DIR}/supply_demand_analysis_results.csv", index=False, encoding='utf-8-sig')
        return core
    
    # ── 4.1 双变量四象限分析 ──
    print(f"\n  ─── 4.1 双变量四象限分析 ───")
    cdi_med = core['CDI'].median()
    esi_med = core['ESI_base'].median()
    print(f"  CDI中位数: {cdi_med:.6f}")
    print(f"  ESI中位数: {esi_med:.6f}")
    
    conditions = [
        (core['CDI'] >= cdi_med) & (core['ESI_base'] >= esi_med),
        (core['CDI'] < cdi_med) & (core['ESI_base'] >= esi_med),
        (core['CDI'] >= cdi_med) & (core['ESI_base'] < esi_med),
        (core['CDI'] < cdi_med) & (core['ESI_base'] < esi_med),
    ]
    labels = ['I', 'II', 'III', 'IV']
    core['Quadrant'] = np.select(conditions, labels, default='IV')
    
    desc = {'I':'高需求-高供给','II':'低需求-高供给','III':'高需求-低供给(关键)','IV':'低需求-低供给'}
    print(f"\n  {'象限':<6}{'描述':<28}{'格网数':>6}{'占比':>8}{'人口(万)':>10}")
    for q in ['I','II','III','IV']:
        s = core[core['Quadrant']==q]
        print(f"  {q:<6}{desc[q]:<28}{len(s):>6}{len(s)/N*100:>7.1f}%{s['population'].sum()/10000:>10.1f}")
    
    # 象限III细分
    q3 = core[core['Quadrant']=='III']
    q3_risk = q3[q3['ESI_base'] < 0.5]
    print(f"\n  象限III高风险(ESI<0.5): {len(q3_risk)}格网 ({len(q3_risk)/len(q3)*100:.1f}%)")
    
    # 按行政区
    if '所属区' in core.columns:
        print(f"\n  象限III按行政区:")
        for d in sorted(core['所属区'].dropna().unique()):
            d_all = len(core[core['所属区']==d])
            d_q3 = len(core[(core['所属区']==d) & (core['Quadrant']=='III')])
            print(f"    {d}: {d_q3}格网 ({d_q3/d_all*100:.1f}%)")
    
    # ── 4.2 缺口-资源耦合分析 ──
    print(f"\n  ─── 4.2 缺口-资源耦合分析 ───")
    
    # 盲区识别
    severe = core[core['ESI_base'] < 0.5]
    n_severe = len(severe)
    print(f"  ESI<0.5 极度匮乏: {n_severe}格网")
    
    covered = severe[severe['delta_ESI'] > 0]
    dual_def = severe[severe['delta_ESI'] == 0]
    print(f"    酒店可覆盖(ΔESI>0): {len(covered)} ({len(covered)/n_severe*100:.1f}%)")
    print(f"    双重缺失(ΔESI=0):   {len(dual_def)} ({len(dual_def)/n_severe*100:.1f}%)")
    
    deficit = core[core['ESI_base'] < 1.0]
    print(f"\n  ESI<1.0 供给不足: {len(deficit)}格网")
    
    # 耦合分类 (在ESI<1.0区域)
    if 'hotel_density' in core.columns:
        gap = 1 - deficit['ESI_base']
        gap_med = gap.median()
        hd_med = deficit['hotel_density'].median()
        
        def coupling_type(row):
            g = 1 - row['ESI_base']
            h = row['hotel_density']
            if g >= gap_med and h >= hd_med: return 'HH'
            elif g >= gap_med and h < hd_med: return 'HL'
            elif g < gap_med and h >= hd_med: return 'LH'
            else: return 'LL'
        
        deficit_copy = deficit.copy()
        deficit_copy['coupling'] = deficit_copy.apply(coupling_type, axis=1)
        
        print(f"\n  耦合分类 (ESI<1.0, n={len(deficit_copy)}):")
        for ct in ['HH','HL','LH','LL']:
            n_ct = len(deficit_copy[deficit_copy['coupling']==ct])
            print(f"    {ct}: {n_ct} ({n_ct/len(deficit_copy)*100:.1f}%)")
    
    # ── 4.3 空间正义验证 ──
    print(f"\n  ─── 4.3 空间正义验证 ───")
    
    cdi_75 = core['CDI'].quantile(0.75)
    high = core[core['CDI'] >= cdi_75]
    other = core[core['CDI'] < cdi_75]
    
    mean_h = high['delta_ESI'].mean()
    mean_o = other['delta_ESI'].mean()
    ratio = mean_h / mean_o if mean_o > 0 else float('inf')
    med_h = high['delta_ESI'].median()
    med_o = other['delta_ESI'].median()
    med_ratio = med_h / med_o if med_o > 0 else float('inf')
    
    u_stat, p_mw = stats.mannwhitneyu(
        high['delta_ESI'], other['delta_ESI'], alternative='greater')
    t_stat, p_t = stats.ttest_ind(high['delta_ESI'], other['delta_ESI'])
    r_pearson, p_p = stats.pearsonr(core['CDI'], core['delta_ESI'])
    r_spearman, p_s = stats.spearmanr(core['CDI'], core['delta_ESI'])
    
    print(f"  CDI 75%分位: {cdi_75:.6f}")
    print(f"  高需求区: n={len(high)}")
    print(f"  其他区域: n={len(other)}")
    print(f"\n  {'指标':<30}{'高需求区':>15}{'其他区域':>15}{'倍数':>8}")
    print(f"  {'-'*68}")
    print(f"  {'均值ΔESI(×10⁻³)':<30}{mean_h*1000:>15.2f}{mean_o*1000:>15.2f}{ratio:>8.2f}")
    print(f"  {'中位数ΔESI(×10⁻³)':<30}{med_h*1000:>15.2f}{med_o*1000:>15.2f}{med_ratio:>8.2f}")
    print(f"  {'ΔESI>0占比':<30}{(high['delta_ESI']>0).mean()*100:>14.1f}%{(other['delta_ESI']>0).mean()*100:>14.1f}%")
    
    print(f"\n  统计检验:")
    print(f"    Mann-Whitney U: p = {p_mw:.2e}")
    print(f"    独立样本t检验:  t = {t_stat:.4f}, p = {p_t:.2e}")
    print(f"    Pearson相关:   r = {r_pearson:.3f}, p = {p_p:.2e}")
    print(f"    Spearman相关:  r = {r_spearman:.3f}, p = {p_s:.2e}")
    
    # ── 保存结果 ──
    core['level_base'] = core['ESI_base'].apply(classify_level)
    core['level_with_hotel'] = core['ESI_with_hotel'].apply(classify_level)
    core['level_improved'] = core['level_base'] != core['level_with_hotel']
    
    output_path = f"{OUTPUT_DIR}/supply_demand_analysis_results.csv"
    core.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  结果已保存: {output_path}")
    
    return core


# ══════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  成都市应急避难设施供需分析 - G2SFCA (β=0.5)")
    print("=" * 70)
    
    # Step 1
    data = step1_load_and_verify()
    
    # Step 2
    Ai_repro = step2_reproduce_baseline(data)
    
    # Step 3
    delta_esi = step3_hotel_accessibility(data)
    
    # Step 4
    results = step4_supply_demand_analysis(data)
    
    print("\n" + "=" * 70)
    print("  分析完成")
    print("=" * 70)
