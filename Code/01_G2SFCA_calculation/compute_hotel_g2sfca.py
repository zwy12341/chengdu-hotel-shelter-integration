"""
================================================================================
含酒店弹性供给的 Gaussian 2SFCA 可达性分析
================================================================================
论文: 平急两用导向下城市酒店的震灾避难服务效能与空间正义

功能: 从原始数据计算含酒店弹性供给的G2SFCA可达性指数
      → 生成 grid_accessibility_with_hotel_daytime.csv
      → 生成 grid_accessibility_with_hotel_nighttime.csv
      → 生成 含酒店弹性供给G2SFCA分析汇总表.xlsx

输入文件:
  1. supply_facilities_CLEAN.csv            基准供给设施 (1054处)
  2. demand_grids_daytime_CLEAN.csv         白天需求格网 (4857个)
  3. demand_grids_nighttime_CLEAN.csv       夜间需求格网 (4857个)
  4. grid_accessibility_daytime.csv         基线可达性 (方案A/B/C, 白天)
  5. grid_accessibility_nighttime.csv       基线可达性 (方案A/B/C, 夜间)
  6. hotel_facilities_filtered.csv          筛选后酒店 (2264家)

方法:
  - Gaussian 2SFCA (Dai 2010), β=0.5
  - 基准设施 d₀=3000m, 酒店 d₀=1000m
  - 酒店容量 = rooms × 2 × λ, λ∈{0.6, 0.7, 0.8}
  - 可达性: Ai = Ai_base(基线) + ΔAi_hotel(酒店增量)

================================================================================
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os, time, warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 1. 参数配置
# ══════════════════════════════════════════════════════
INPUT_DIR  = "/mnt/user-data/uploads"      # ← 修改为实际路径
OUTPUT_DIR = "/mnt/user-data/outputs"       # ← 修改为实际路径
os.makedirs(OUTPUT_DIR, exist_ok=True)

# G2SFCA模型参数
D0_HOTEL = 1000     # 酒店服务半径 (m), 论文§2.4.4
BETA     = 0.5      # Gaussian衰减参数, Dai (2010)

# 酒店折算系数 (敏感性分析)
LAMBDAS = [0.6, 0.7, 0.8]
LAMBDA_BASE = 0.7   # 基准值

# 坐标转换 (成都, 纬度≈30.67°N)
LAT_TO_M = 111000   # 1°纬度 ≈ 111km
LON_TO_M = 95400    # 1°经度 ≈ 95.4km (cos30.67°×111000)


# ══════════════════════════════════════════════════════
# 2. 核心函数
# ══════════════════════════════════════════════════════
def to_meters(lon, lat):
    """经纬度 → 近似平面米坐标 (以经纬度原点为基准)"""
    return lon * LON_TO_M, lat * LAT_TO_M


def gaussian_decay(dist, d0, beta=BETA):
    """
    Dai (2010) 高斯距离衰减函数

    公式:
        G(d, d₀) = [exp(-(d/d₀)²/β) - exp(-1/β)] / [1 - exp(-1/β)]

    性质: G(0)=1, G(d₀)=0, 中距离衰减最显著
    参考: Dai D. (2010) Health & Place, 16(6):1038-1052

    Parameters
    ----------
    dist : ndarray  距离数组 (m)
    d0   : float    服务半径阈值 (m)
    beta : float    衰减参数, 默认0.5

    Returns
    -------
    ndarray : 衰减权重, 范围[0, 1]
    """
    ratio = dist / d0
    numerator   = np.exp(-ratio**2 / beta) - np.exp(-1.0 / beta)
    denominator = 1.0 - np.exp(-1.0 / beta)
    result = np.where(dist <= d0, numerator / denominator, 0.0)
    return np.maximum(result, 0.0)


def g2sfca_hotel(grid_xy, grid_pop, hotel_xy, hotel_cap, d0=D0_HOTEL, tag=""):
    """
    仅计算酒店增量的 G2SFCA

    酒店与基准设施独立计算 (不同d₀), 最终可达性 = 基线 + 酒店增量
    公式见论文 §2.4.5

    Parameters
    ----------
    grid_xy   : ndarray (N, 2)  格网坐标 (米)
    grid_pop  : ndarray (N,)    格网人口
    hotel_xy  : ndarray (M, 2)  酒店坐标 (米)
    hotel_cap : ndarray (M,)    酒店容量
    d0        : float           酒店服务半径 (米)
    tag       : str             标签 (打印用)

    Returns
    -------
    delta_Ai : ndarray (N,)  酒店带来的可达性增量
    """
    N = len(grid_xy)
    M = len(hotel_xy)
    print(f"    [{tag}] 格网={N}, 酒店={M}, d₀={d0}m")
    t0 = time.time()

    # ── Step 1: 以酒店为中心, 计算供需比 R_j ──
    grid_tree = cKDTree(grid_xy)
    R_j = np.zeros(M)

    # 找每家酒店d₀范围内的格网
    idx_grids = grid_tree.query_ball_point(hotel_xy, r=d0 * 1.02)

    for j in range(M):
        neighbors = np.array(idx_grids[j], dtype=int)
        if len(neighbors) == 0:
            continue
        dist = np.linalg.norm(grid_xy[neighbors] - hotel_xy[j], axis=1)
        weights = gaussian_decay(dist, d0)
        weighted_pop = np.dot(grid_pop[neighbors], weights)
        R_j[j] = hotel_cap[j] / weighted_pop if weighted_pop > 1e-10 else 0.0

    # ── Step 2: 以格网为中心, 汇总可达性增量 ──
    hotel_tree = cKDTree(hotel_xy)
    delta_Ai = np.zeros(N)

    # 找每个格网d₀范围内的酒店
    idx_hotels = hotel_tree.query_ball_point(grid_xy, r=d0 * 1.02)

    for i in range(N):
        neighbors = np.array(idx_hotels[i], dtype=int)
        if len(neighbors) == 0:
            continue
        dist = np.linalg.norm(hotel_xy[neighbors] - grid_xy[i], axis=1)
        weights = gaussian_decay(dist, d0)
        delta_Ai[i] = np.dot(R_j[neighbors], weights)

    elapsed = time.time() - t0
    n_covered = (delta_Ai > 0).sum()
    print(f"    [{tag}] 完成 ({elapsed:.1f}s)")
    print(f"    [{tag}] ΔAi: mean={delta_Ai.mean():.6f}, max={delta_Ai.max():.4f}, "
          f"覆盖={n_covered}格网({n_covered/N*100:.1f}%)")

    return delta_Ai


def classify_level(ai_value):
    """供需分级 (论文表2-4)"""
    if ai_value >= 1.5:
        return '供给充足'
    elif ai_value >= 1.0:
        return '基本平衡'
    elif ai_value >= 0.5:
        return '供给不足'
    else:
        return '严重不足'


# ══════════════════════════════════════════════════════
# 3. 主计算流程
# ══════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  含酒店弹性供给 G2SFCA 可达性分析")
    print("  Gaussian β=0.5 | 酒店d₀=1000m | λ={0.6, 0.7, 0.8}")
    print("=" * 70)

    # ── 3.1 加载数据 ──
    print("\n[1/5] 加载数据...")

    hotels = pd.read_csv(f"{INPUT_DIR}/hotel_facilities_filtered.csv")
    demand_day   = pd.read_csv(f"{INPUT_DIR}/demand_grids_daytime_CLEAN.csv")
    demand_night = pd.read_csv(f"{INPUT_DIR}/demand_grids_nighttime_CLEAN.csv")
    acc_day   = pd.read_csv(f"{INPUT_DIR}/grid_accessibility_daytime.csv")
    acc_night = pd.read_csv(f"{INPUT_DIR}/grid_accessibility_nighttime.csv")

    print(f"  酒店: {len(hotels)}家, 总客房={hotels['rooms'].sum()}")
    print(f"  白天格网: {len(demand_day)}, 总人口={demand_day['population'].sum():,}")
    print(f"  夜间格网: {len(demand_night)}, 总人口={demand_night['population'].sum():,}")

    # ── 3.2 坐标转换 ──
    print("\n[2/5] 坐标转换...")

    hotel_x, hotel_y = to_meters(hotels['lon'].values, hotels['lat'].values)
    hotel_xy = np.column_stack([hotel_x, hotel_y])

    grid_day_x, grid_day_y = to_meters(demand_day['lon'].values, demand_day['lat'].values)
    grid_day_xy = np.column_stack([grid_day_x, grid_day_y])
    pop_day = demand_day['population'].values.astype(float)

    grid_night_x, grid_night_y = to_meters(demand_night['lon'].values, demand_night['lat'].values)
    grid_night_xy = np.column_stack([grid_night_x, grid_night_y])
    pop_night = demand_night['population'].values.astype(float)

    # ── 3.3 计算各λ下的酒店ΔESI ──
    print("\n[3/5] 计算酒店弹性供给增量 (3组λ × 2场景)...")

    results_day = {}
    results_night = {}

    for lam in LAMBDAS:
        cap_col = f"capacity_{lam}"
        hotel_cap = hotels[cap_col].values.astype(float)

        print(f"\n  === λ={lam} (容量={hotel_cap.sum():,.0f}人) ===")

        # 白天
        delta_day = g2sfca_hotel(
            grid_day_xy, pop_day, hotel_xy, hotel_cap,
            d0=D0_HOTEL, tag=f"白天λ={lam}"
        )
        results_day[lam] = delta_day

        # 夜间
        delta_night = g2sfca_hotel(
            grid_night_xy, pop_night, hotel_xy, hotel_cap,
            d0=D0_HOTEL, tag=f"夜间λ={lam}"
        )
        results_night[lam] = delta_night

    # ── 3.4 组装输出表 ──
    print("\n[4/5] 组装输出...")

    for scenario, demand, acc, results, filename in [
        ("白天", demand_day, acc_day, results_day, "grid_accessibility_with_hotel_daytime.csv"),
        ("夜间", demand_night, acc_night, results_night, "grid_accessibility_with_hotel_nighttime.csv"),
    ]:
        # 基线可达性 (方案A)
        Ai_base = acc['Ai_A'].values

        out = pd.DataFrame({
            'GRID_ID': demand['GRID_ID'],
            'lon': demand['lon'],
            'lat': demand['lat'],
            'population': demand['population'],
            'population_density_km2': demand['population_density_km2'],
            'Ai_base': Ai_base,
            'level_base': [classify_level(v) for v in Ai_base],
        })

        # 各λ含酒店可达性
        for lam in LAMBDAS:
            out[f'Ai_hotel_{lam}'] = Ai_base + results[lam]

        # 基准λ=0.7的分级
        out['level_hotel_0.7'] = [classify_level(v) for v in out['Ai_hotel_0.7']]

        # ΔESI (基准λ=0.7)
        out['Ai_improve'] = results[LAMBDA_BASE]

        # 保存
        out_path = f"{OUTPUT_DIR}/{filename}"
        out.to_csv(out_path, index=False)
        print(f"  {scenario}: {out_path}")

        # 打印统计
        print(f"    Ai_base mean={Ai_base.mean():.6f}")
        for lam in LAMBDAS:
            ai_h = out[f'Ai_hotel_{lam}'].mean()
            pct = (ai_h - Ai_base.mean()) / Ai_base.mean() * 100
            print(f"    Ai_hotel_{lam} mean={ai_h:.6f} (+{pct:.2f}%)")

    # ── 3.5 生成汇总表 ──
    print("\n[5/5] 生成Excel汇总表...")
    _generate_summary_excel(hotels, demand_day, demand_night, acc_day, acc_night,
                            results_day, results_night)

    print("\n" + "=" * 70)
    print("  计算完成!")
    print("=" * 70)


# ══════════════════════════════════════════════════════
# 4. Excel汇总表生成
# ══════════════════════════════════════════════════════
def _generate_summary_excel(hotels, demand_day, demand_night, acc_day, acc_night,
                            results_day, results_night):
    """生成 含酒店弹性供给G2SFCA分析汇总表.xlsx"""

    out_path = f"{OUTPUT_DIR}/含酒店弹性供给G2SFCA分析汇总表.xlsx"

    Ai_base_day = acc_day['Ai_A'].values
    Ai_base_night = acc_night['Ai_A'].values
    pop_day = demand_day['population'].values
    pop_night = demand_night['population'].values

    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:

        # ── Sheet 1: 参数设置 ──
        params = [
            ['含酒店弹性供给的Gaussian 2SFCA分析 - 参数设置'],
            [],
            ['一、模型参数'],
            ['参数名称', '参数值', '单位', '文献依据'],
            ['基准设施服务半径', 3000, 'm', 'GB 51143-2015; GB 50413-2007'],
            ['酒店服务半径', D0_HOTEL, 'm', '15分钟生活圈; 自然资源部(2021)'],
            ['Gaussian衰减参数β', BETA, '-', 'Dai (2010) Health & Place'],
            ['人均避难面积', 2.0, 'm²/人', 'GB 51143-2015 表3.1.10'],
            [],
            ['二、酒店筛选标准'],
            ['参数', '取值', '说明'],
            ['星级要求', '≥2星', '确保基本设施质量'],
            ['最小客房数', '≥30间', '排除小型酒店'],
            ['容量公式', '客房数×2×λ', 'FEMA指南'],
            ['基准折算系数λ', 0.7, 'Homekey项目经验'],
            ['敏感性分析λ', '0.6/0.7/0.8', '稳健性检验'],
            [],
            ['三、酒店数据统计'],
            ['指标', '数值', '单位'],
            ['筛选后酒店数', f'{len(hotels):,}', '个'],
            ['总客房数', f'{hotels["rooms"].sum():,}', '间'],
        ]
        for lam in LAMBDAS:
            cap = hotels[f'capacity_{lam}'].sum()
            params.append([f'容量(λ={lam})', f'{cap:,.0f}', '人'])

        pd.DataFrame(params).to_excel(writer, sheet_name='参数设置', index=False, header=False)

        # ── Sheet 2: 容量与可达性对比 ──
        rows = [
            ['基准供给 vs 含酒店弹性供给 对比分析'],
            [],
            ['一、总容量对比'],
            ['场景', '总容量(人)', 'vs基准变化'],
        ]

        # 基准容量
        supply = pd.read_csv(f"{INPUT_DIR}/supply_facilities_CLEAN.csv")
        lambda_a = {'公园绿地':0.60,'体育场馆':0.65,'大学/高等院校':0.60,
                     '中学':0.60,'小学':0.60,'其他学校':0.60}
        supply['cap_A'] = supply.apply(
            lambda r: r['total_area_m2'] * lambda_a.get(r['facility_subtype'], 0.60) / 2.0, axis=1)
        base_cap = supply['cap_A'].sum()
        rows.append(['基准供给(方案A)', int(base_cap), '--'])

        for lam in LAMBDAS:
            hotel_cap = hotels[f'capacity_{lam}'].sum()
            total = base_cap + hotel_cap
            pct = hotel_cap / base_cap * 100
            rows.append([f'含酒店(λ={lam})', int(total), f'+{pct:.2f}%'])

        rows += [[], ['二、可达性均值对比'],
                 ['场景', '白天均值', '夜间均值', '白天提升', '夜间提升']]
        rows.append(['基准供给(方案A)', Ai_base_day.mean(), Ai_base_night.mean(), '--', '--'])

        for lam in LAMBDAS:
            ai_d = Ai_base_day + results_day[lam]
            ai_n = Ai_base_night + results_night[lam]
            pct_d = (ai_d.mean() - Ai_base_day.mean()) / Ai_base_day.mean() * 100
            pct_n = (ai_n.mean() - Ai_base_night.mean()) / Ai_base_night.mean() * 100
            rows.append([f'含酒店(λ={lam})', ai_d.mean(), ai_n.mean(),
                        f'+{pct_d:.2f}%', f'+{pct_n:.2f}%'])

        rows += [[], ['三、供给不足(Ai<1)人口比例对比'],
                 ['场景', '白天', '夜间', '白天改善', '夜间改善']]
        base_pct_d = pop_day[Ai_base_day < 1.0].sum() / pop_day.sum() * 100
        base_pct_n = pop_night[Ai_base_night < 1.0].sum() / pop_night.sum() * 100
        rows.append(['基准供给(方案A)', f'{base_pct_d:.1f}%', f'{base_pct_n:.1f}%', '--', '--'])

        for lam in LAMBDAS:
            ai_d = Ai_base_day + results_day[lam]
            ai_n = Ai_base_night + results_night[lam]
            pct_d = pop_day[ai_d < 1.0].sum() / pop_day.sum() * 100
            pct_n = pop_night[ai_n < 1.0].sum() / pop_night.sum() * 100
            rows.append([f'含酒店(λ={lam})', f'{pct_d:.1f}%', f'{pct_n:.1f}%',
                        f'{pct_d-base_pct_d:+.1f}pp', f'{pct_n-base_pct_n:+.1f}pp'])

        pd.DataFrame(rows).to_excel(writer, sheet_name='容量与可达性对比', index=False, header=False)

        # ── Sheet 3: 酒店贡献分析 ──
        rows3 = [
            ['酒店弹性供给贡献分析（λ=0.7基准）'],
            [],
            ['一、可达性提升统计'],
            ['指标', '白天', '夜间'],
            ['均值提升', results_day[0.7].mean(), results_night[0.7].mean()],
            ['最大提升', results_day[0.7].max(), results_night[0.7].max()],
            ['提升>0.1的格网数', int((results_day[0.7]>0.1).sum()), int((results_night[0.7]>0.1).sum())],
            ['提升>0.1的格网比例',
             f'{(results_day[0.7]>0.1).mean()*100:.1f}%',
             f'{(results_night[0.7]>0.1).mean()*100:.1f}%'],
            [],
            ['二、供需状况改善人口'],
            ['指标', '白天', '夜间'],
        ]
        for label, ai_base, delta, pop in [
            ('', Ai_base_day, results_day[0.7], pop_day),
            ('', Ai_base_night, results_night[0.7], pop_night)
        ]:
            pass

        ai_d = Ai_base_day + results_day[0.7]
        ai_n = Ai_base_night + results_night[0.7]
        improved_d = (Ai_base_day < 1.0) & (ai_d >= 1.0)
        improved_n = (Ai_base_night < 1.0) & (ai_n >= 1.0)
        rows3.append(['从"不足"转为"平衡"人口',
                      int(pop_day[improved_d].sum()),
                      int(pop_night[improved_n].sum())])
        total_pop = pop_day.sum()
        rows3.append(['占总人口比例',
                      f'{pop_day[improved_d].sum()/total_pop*100:.2f}%',
                      f'{pop_night[improved_n].sum()/pop_night.sum()*100:.2f}%'])

        rows3 += [[], ['三、按星级的酒店分布'],
                  ['星级', '酒店数', '总客房', '容量(λ=0.7)', '平均客房']]
        for star in sorted(hotels['star_rating'].unique()):
            sub = hotels[hotels['star_rating']==star]
            rows3.append([f'{star}星', len(sub), sub['rooms'].sum(),
                         sub['capacity_0.7'].sum(), sub['rooms'].mean()])
        sub_all = hotels
        rows3.append(['合计', len(sub_all), sub_all['rooms'].sum(),
                     sub_all['capacity_0.7'].sum(), sub_all['rooms'].mean()])

        pd.DataFrame(rows3).to_excel(writer, sheet_name='酒店贡献分析', index=False, header=False)

        # ── Sheet 4: 敏感性分析 ──
        rows4 = [
            ['酒店折算系数敏感性分析汇总'],
            [],
            ['一、关键指标对比'],
            ['指标', 'λ=0.6', 'λ=0.7(基准)', 'λ=0.8', '敏感程度'],
        ]

        caps = {lam: hotels[f'capacity_{lam}'].sum() for lam in LAMBDAS}
        rows4.append(['总容量(人)', int(caps[0.6]), int(caps[0.7]), int(caps[0.8]),
                      f'{(caps[0.8]-caps[0.6])/caps[0.7]*100:.1f}%变幅'])

        for label, ai_base, results in [('白天均值', Ai_base_day, results_day),
                                         ('夜间均值', Ai_base_night, results_night)]:
            vals = {lam: (ai_base + results[lam]).mean() for lam in LAMBDAS}
            rng = (vals[0.8] - vals[0.6]) / vals[0.7] * 100
            rows4.append([label, f'{vals[0.6]:.6f}', f'{vals[0.7]:.6f}',
                         f'{vals[0.8]:.6f}', f'{rng:.2f}%变幅'])

        for label, ai_base, results, pop in [
            ('白天不足比例', Ai_base_day, results_day, pop_day),
            ('夜间不足比例', Ai_base_night, results_night, pop_night)
        ]:
            vals = {}
            for lam in LAMBDAS:
                ai = ai_base + results[lam]
                vals[lam] = pop[ai < 1.0].sum() / pop.sum() * 100
            rows4.append([label, f'{vals[0.6]:.1f}%', f'{vals[0.7]:.1f}%',
                         f'{vals[0.8]:.1f}%', f'{vals[0.8]-vals[0.6]:+.1f}pp变幅'])

        pd.DataFrame(rows4).to_excel(writer, sheet_name='敏感性分析汇总', index=False, header=False)

    print(f"  汇总表: {out_path}")


# ══════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
