"""
================================================================================
PDP (偏依赖图) 分析 — v4 XGBoost最优模型
================================================================================
模型: A_logΔESI_14feat × XGBoost
参数: n_estimators=200, max_depth=6, learning_rate=0.08, 
      subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1

输出:
  1. 一维PDP曲线数据 (top 8非空间特征) → Fig.9
  2. 二维PDP交互数据 (top 3交互对) → Fig.9补充
  3. ICE曲线数据 (个体条件期望, 展示异质性)
================================================================================
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = "/mnt/user-data/uploads"
OUTPUT_DIR = "/home/claude"
RANDOM_STATE = 42

# ═══════════════════════════════════════════════════════════════
#  Step 1: 重训练最优模型
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("Step 1: 重训练XGBoost最优模型")
print("=" * 70)

df = pd.read_csv(f"{INPUT_DIR}/v4_data_full.csv")
df['ESI_base'] = df['ESI'] - df['delta_ESI']
df['lon_f'] = df['lon']
df['lat_f'] = df['lat']

features_14 = ['H11', 'H12', 'E11', 'E12', 'E21', 'E31',
               'V11', 'V21', 'V22', 'V31', 'V32',
               'ESI_base', 'lon_f', 'lat_f']

feature_labels = {
    'H11': 'Fault Distance (H₁₁)', 'H12': 'Epicenter Distance (H₁₂)',
    'E11': 'Population Density (E₁₁)', 'E12': 'Vulnerable Pop. Ratio (E₁₂)',
    'E21': 'GDP per Capita (E₂₁)', 'E31': 'Building Density (E₃₁)',
    'V11': 'Medical Facility Density (V₁₁)', 'V21': 'Building Height (V₂₁)',
    'V22': 'Building Age (V₂₂)', 'V31': 'Elevation Diff. (V₃₁)',
    'V32': 'Slope (V₃₂)', 'ESI_base': 'Baseline ESI',
    'lon_f': 'Longitude', 'lat_f': 'Latitude'
}

feature_labels_cn = {
    'H11': '断层距离', 'H12': '震中距离',
    'E11': '人口密度(昼)', 'E12': '脆弱人口比例',
    'E21': '人均GDP', 'E31': '建筑密度',
    'V11': '医疗机构密度', 'V21': '建筑高度',
    'V22': '建筑年龄', 'V31': '高程差',
    'V32': '坡度', 'ESI_base': '基线ESI',
    'lon_f': '经度', 'lat_f': '纬度'
}

X = df[features_14].values
y = df['log_dESI'].values

# 全样本训练 (PDP用全样本模型)
model = XGBRegressor(
    n_estimators=200, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
)
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
print(f"  全样本 R² = {r2:.4f}")
print(f"  样本量: {len(y)}, 特征数: {len(features_14)}")


# ═══════════════════════════════════════════════════════════════
#  Step 2: 一维PDP计算
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 2: 一维PDP (top 8 非空间特征)")
print("=" * 70)

# 按重要性排序的非空间特征 (from v4_feature_importance.csv)
pdp_features = ['V11', 'E11', 'V22', 'V31', 'E21', 'H12', 'E12', 'ESI_base']
n_grid = 50  # PDP网格点数

pdp_results = {}
ice_results = {}

# 随机抽样子集用于加速ICE (取500个样本)
np.random.seed(RANDOM_STATE)
ice_sample_idx = np.random.choice(len(X), size=min(500, len(X)), replace=False)
X_ice_sample = X[ice_sample_idx]

for feat in pdp_features:
    feat_idx = features_14.index(feat)
    feat_values = X[:, feat_idx]
    
    # PDP网格: 从5%到95%分位数 (避免极端值)
    grid = np.linspace(np.percentile(feat_values, 2), 
                       np.percentile(feat_values, 98), n_grid)
    
    # PDP: 对每个网格点, 将所有样本的该特征替换为网格值, 取预测均值
    pdp_values = np.zeros(n_grid)
    ice_matrix = np.zeros((len(ice_sample_idx), n_grid))
    
    t0 = time.time()
    for g, grid_val in enumerate(grid):
        X_temp = X.copy()
        X_temp[:, feat_idx] = grid_val
        preds = model.predict(X_temp)
        pdp_values[g] = preds.mean()
        
        # ICE: 用子集
        X_ice_temp = X_ice_sample.copy()
        X_ice_temp[:, feat_idx] = grid_val
        ice_matrix[:, g] = model.predict(X_ice_temp)
    
    # 中心化ICE (c-ICE): 减去各样本在第一个网格点的预测值
    ice_centered = ice_matrix - ice_matrix[:, 0:1]
    
    pdp_results[feat] = {
        'grid': grid,
        'pdp': pdp_values,
        'pdp_centered': pdp_values - pdp_values[0],
        'ice_mean': ice_matrix.mean(axis=0),
        'ice_std': ice_matrix.std(axis=0),
        'ice_q25': np.percentile(ice_matrix, 25, axis=0),
        'ice_q75': np.percentile(ice_matrix, 75, axis=0),
        'ice_q10': np.percentile(ice_matrix, 10, axis=0),
        'ice_q90': np.percentile(ice_matrix, 90, axis=0),
    }
    ice_results[feat] = ice_centered
    
    # 打印关键特征
    total_effect = pdp_values[-1] - pdp_values[0]
    print(f"  {feat} ({feature_labels_cn[feat]}): "
          f"range=[{grid[0]:.1f}, {grid[-1]:.1f}], "
          f"PDP变化={total_effect:+.3f}, "
          f"耗时={time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════
#  Step 3: PDP形状分析 (识别非线性模式)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 3: PDP形状分析 (非线性模式识别)")
print("=" * 70)

for feat in pdp_features:
    r = pdp_results[feat]
    grid = r['grid']
    pdp = r['pdp']
    
    # 一阶差分 (边际效应)
    marginal = np.diff(pdp)
    
    # 识别拐点: 边际效应符号变化
    sign_changes = np.sum(np.diff(np.sign(marginal)) != 0)
    
    # 总效应方向
    total = pdp[-1] - pdp[0]
    direction = "正向" if total > 0 else "负向" if total < 0 else "无"
    
    # 前半段 vs 后半段的边际效应
    mid = len(marginal) // 2
    first_half_effect = pdp[mid] - pdp[0]
    second_half_effect = pdp[-1] - pdp[mid]
    
    # 判断形状
    if abs(total) < 0.05:
        shape = "弱效应/平坦"
    elif abs(first_half_effect) > 3 * abs(second_half_effect) and abs(second_half_effect) < 0.05:
        shape = "饱和型 (前段快速变化后趋平)"
    elif abs(second_half_effect) > 3 * abs(first_half_effect) and abs(first_half_effect) < 0.05:
        shape = "延迟型 (前段平后段快速变化)"
    elif sign_changes >= 2:
        shape = "非单调 (存在拐点)"
    elif abs(first_half_effect) > 2 * abs(second_half_effect):
        shape = "递减边际效应"
    elif abs(second_half_effect) > 2 * abs(first_half_effect):
        shape = "递增边际效应"
    else:
        shape = "近似线性"
    
    # 找最大边际效应区间
    max_marginal_idx = np.argmax(np.abs(marginal))
    max_marginal_range = (grid[max_marginal_idx], grid[max_marginal_idx + 1])
    
    print(f"\n  {feat} ({feature_labels_cn[feat]}):")
    print(f"    总效应: {total:+.3f} ({direction})")
    print(f"    形状: {shape}")
    print(f"    拐点数: {sign_changes}")
    print(f"    前半效应: {first_half_effect:+.3f}, 后半效应: {second_half_effect:+.3f}")
    print(f"    最大边际区间: [{max_marginal_range[0]:.1f}, {max_marginal_range[1]:.1f}]")


# ═══════════════════════════════════════════════════════════════
#  Step 4: 二维PDP (top 交互对)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 4: 二维PDP (top 3 非空间交互对)")
print("=" * 70)

# 从H-statistic中选top交互对 (排除空间坐标)
# v4_h_statistic: E11×E12=0.0195, E11×V22=0.0174, V22×E12=0.0086
interaction_pairs = [
    ('E11', 'E12', 'E₁₁×E₁₂'),
    ('E11', 'V22', 'E₁₁×V₂₂'),
    ('V11', 'E11', 'V₁₁×E₁₁'),
]

n_grid_2d = 25  # 2D网格较粗

pdp_2d_results = {}

for feat1, feat2, label in interaction_pairs:
    idx1 = features_14.index(feat1)
    idx2 = features_14.index(feat2)
    
    grid1 = np.linspace(np.percentile(X[:, idx1], 5), np.percentile(X[:, idx1], 95), n_grid_2d)
    grid2 = np.linspace(np.percentile(X[:, idx2], 5), np.percentile(X[:, idx2], 95), n_grid_2d)
    
    pdp_2d = np.zeros((n_grid_2d, n_grid_2d))
    
    t0 = time.time()
    for i, g1 in enumerate(grid1):
        for j, g2 in enumerate(grid2):
            X_temp = X.copy()
            X_temp[:, idx1] = g1
            X_temp[:, idx2] = g2
            pdp_2d[i, j] = model.predict(X_temp).mean()
    
    pdp_2d_results[(feat1, feat2)] = {
        'grid1': grid1, 'grid2': grid2,
        'pdp_2d': pdp_2d,
        'label': label
    }
    
    print(f"  {label}: range=[{pdp_2d.min():.3f}, {pdp_2d.max():.3f}], "
          f"span={pdp_2d.max()-pdp_2d.min():.3f} ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════
#  Step 5: 保存结果
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Step 5: 保存结果")
print("=" * 70)

# 5.1 一维PDP数据
pdp_rows = []
for feat in pdp_features:
    r = pdp_results[feat]
    for i in range(n_grid):
        pdp_rows.append({
            'feature': feat,
            'feature_label': feature_labels_cn[feat],
            'feature_label_en': feature_labels[feat],
            'grid_value': r['grid'][i],
            'pdp': r['pdp'][i],
            'pdp_centered': r['pdp_centered'][i],
            'ice_q10': r['ice_q10'][i],
            'ice_q25': r['ice_q25'][i],
            'ice_q75': r['ice_q75'][i],
            'ice_q90': r['ice_q90'][i],
        })

pdp_df = pd.DataFrame(pdp_rows)
pdp_df.to_csv(f"{OUTPUT_DIR}/v4_pdp_1d.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ v4_pdp_1d.csv ({len(pdp_df)}行)")

# 5.2 二维PDP数据
for (feat1, feat2), r in pdp_2d_results.items():
    fname = f"{OUTPUT_DIR}/v4_pdp_2d_{feat1}_{feat2}.csv"
    # 转为长格式
    rows_2d = []
    for i in range(n_grid_2d):
        for j in range(n_grid_2d):
            rows_2d.append({
                'feature1': feat1,
                'feature2': feat2,
                'grid1': r['grid1'][i],
                'grid2': r['grid2'][j],
                'pdp': r['pdp_2d'][i, j]
            })
    pd.DataFrame(rows_2d).to_csv(fname, index=False, encoding='utf-8-sig')
    print(f"  ✓ {fname.split('/')[-1]} ({len(rows_2d)}行)")

# 5.3 ICE曲线抽样数据 (用于展示个体异质性, 取50条)
ice_sample_export = np.random.choice(len(ice_sample_idx), size=50, replace=False)
for feat in pdp_features[:4]:  # 只导出top 4特征的ICE
    r = pdp_results[feat]
    ice_rows = []
    for s_idx in ice_sample_export:
        for g in range(n_grid):
            ice_rows.append({
                'feature': feat,
                'sample_id': int(s_idx),
                'grid_value': r['grid'][g],
                'ice_centered': ice_results[feat][s_idx, g]
            })
    ice_df = pd.DataFrame(ice_rows)
    ice_df.to_csv(f"{OUTPUT_DIR}/v4_ice_{feat}.csv", index=False, encoding='utf-8-sig')

print(f"  ✓ ICE曲线: v4_ice_V11/E11/V22/V31.csv")

# 5.4 PDP形状汇总表
shape_rows = []
for feat in pdp_features:
    r = pdp_results[feat]
    grid = r['grid']
    pdp = r['pdp']
    total = pdp[-1] - pdp[0]
    mid = len(pdp) // 2
    first_half = pdp[mid] - pdp[0]
    second_half = pdp[-1] - pdp[mid]
    marginal = np.diff(pdp)
    sign_changes = int(np.sum(np.diff(np.sign(marginal)) != 0))
    max_mg_idx = np.argmax(np.abs(marginal))
    
    shape_rows.append({
        '特征': feat,
        '名称': feature_labels_cn[feat],
        '总效应': total,
        '前半效应': first_half,
        '后半效应': second_half,
        '拐点数': sign_changes,
        '最大边际区间_左': grid[max_mg_idx],
        '最大边际区间_右': grid[max_mg_idx + 1],
        'PDP范围_min': pdp.min(),
        'PDP范围_max': pdp.max(),
    })

shape_df = pd.DataFrame(shape_rows)
shape_df.to_csv(f"{OUTPUT_DIR}/v4_pdp_shape_analysis.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ v4_pdp_shape_analysis.csv")


# ═══════════════════════════════════════════════════════════════
#  结果汇总
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PDP 结果汇总 (论文4.5.2节)")
print("=" * 70)

print(f"""
8个非空间特征的PDP关键发现:
""")

for feat in pdp_features:
    r = pdp_results[feat]
    total = r['pdp'][-1] - r['pdp'][0]
    direction = "↑" if total > 0 else "↓"
    print(f"  {feat:>8} ({feature_labels_cn[feat]:<10}): "
          f"总效应 {direction} {abs(total):.3f}")
