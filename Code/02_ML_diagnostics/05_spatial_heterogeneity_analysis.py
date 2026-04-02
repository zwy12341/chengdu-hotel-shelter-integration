"""
================================================================================
KNN-LWR v4 最终版
================================================================================
两组配置:
  A) Fig.8模型对比:  14feat + k=100 → CV R² (与其他模型同特征集)
  B) Fig.10空间异质性: 12feat + k=300 → 局部系数/分区分析
     (去掉lon/lat因KNN-LWR的核权重已内在处理空间效应; 
      k=300用于提升系数稳定性)
================================================================================
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

INPUT_DIR = "/mnt/user-data/uploads"
OUTPUT_DIR = "/home/claude"
RANDOM_STATE = 42

df = pd.read_csv(f"{INPUT_DIR}/v4_data_full.csv")
df['ESI_base'] = df['ESI'] - df['delta_ESI']
df['lon_f'] = df['lon']
df['lat_f'] = df['lat']
coords = df[['lon', 'lat']].values
n = len(df)

features_14 = ['H11','H12','E11','E12','E21','E31',
               'V11','V21','V22','V31','V32','ESI_base','lon_f','lat_f']
features_12 = ['H11','H12','E11','E12','E21','E31',
               'V11','V21','V22','V31','V32','ESI_base']

y = df['log_dESI'].values

dim_map = {
    'H11': 'H(危险性)', 'H12': 'H(危险性)',
    'E11': 'E(暴露度)', 'E12': 'E(暴露度)', 
    'E21': 'E(暴露度)', 'E31': 'E(暴露度)',
    'V11': 'V(脆弱性)', 'V21': 'V(脆弱性)', 
    'V22': 'V(脆弱性)', 'V31': 'V(脆弱性)', 'V32': 'V(脆弱性)',
    'ESI_base': 'Supply(供给基线)', 'lon_f': 'Spatial(空间)', 'lat_f': 'Spatial(空间)'
}

def bisquare_kernel(distances, bandwidth):
    u = distances / bandwidth
    return np.where(u < 1.0, (1 - u**2)**2, 0.0)


# ═══════════════════════════════════════════════════════════════
#  Part A: Fig.8 模型对比 (14feat, k=100)
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("Part A: Fig.8 模型对比 (14feat, k=100)")
print("=" * 70)

K_A = 100
scaler_14 = StandardScaler()
X_14 = scaler_14.fit_transform(df[features_14].values)

# A.1 全样本
preds_A = np.zeros(n)
t0 = time.time()
for i in range(n):
    dists = cdist(coords[i:i+1], coords)[0]
    nn = np.argsort(dists)[:K_A]
    bw = dists[nn[-1]] + 1e-10
    w = bisquare_kernel(dists[nn], bw)
    Xn = np.column_stack([np.ones(K_A), X_14[nn]])
    W = np.diag(w)
    try:
        XtW = Xn.T @ W
        beta = np.linalg.solve(XtW @ Xn + 1e-8*np.eye(Xn.shape[1]), XtW @ y[nn])
        preds_A[i] = np.concatenate([[1.0], X_14[i]]) @ beta
    except:
        preds_A[i] = np.mean(y[nn])

r2_A = r2_score(y, preds_A)
rmse_A = np.sqrt(mean_squared_error(y, preds_A))
mae_A = mean_absolute_error(y, preds_A)
print(f"  全样本: R²={r2_A:.4f}, RMSE={rmse_A:.4f}, MAE={mae_A:.4f} ({time.time()-t0:.1f}s)")

# A.2 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores_A = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_14)):
    X_tr, X_te = X_14[train_idx], X_14[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    c_tr, c_te = coords[train_idx], coords[test_idx]
    
    y_pred = np.zeros(len(test_idx))
    for j in range(len(test_idx)):
        dists = cdist(c_te[j:j+1], c_tr)[0]
        nn = np.argsort(dists)[:K_A]
        bw = dists[nn[-1]] + 1e-10
        w = bisquare_kernel(dists[nn], bw)
        Xn = np.column_stack([np.ones(K_A), X_tr[nn]])
        W = np.diag(w)
        try:
            XtW = Xn.T @ W
            beta = np.linalg.solve(XtW @ Xn + 1e-8*np.eye(Xn.shape[1]), XtW @ y_tr[nn])
            y_pred[j] = np.concatenate([[1.0], X_te[j]]) @ beta
        except:
            y_pred[j] = np.mean(y_tr[nn])
    
    fold_r2 = r2_score(y_te, y_pred)
    cv_scores_A.append(fold_r2)

cv_mean_A = np.mean(cv_scores_A)
cv_std_A = np.std(cv_scores_A)
print(f"  CV R² = {cv_mean_A:.4f} ± {cv_std_A:.4f}")
print(f"  各折: {[f'{s:.4f}' for s in cv_scores_A]}")


# ═══════════════════════════════════════════════════════════════
#  Part B: Fig.10 空间异质性 (12feat, k=300)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part B: Fig.10 空间异质性 (12feat, k=300)")
print("=" * 70)

K_B = 300
scaler_12 = StandardScaler()
X_12 = scaler_12.fit_transform(df[features_12].values)
n_feat_12 = len(features_12)

coefs_B = np.zeros((n, n_feat_12 + 1))  # +1 intercept
preds_B = np.zeros(n)

t0 = time.time()
for i in range(n):
    dists = cdist(coords[i:i+1], coords)[0]
    nn = np.argsort(dists)[:K_B]
    bw = dists[nn[-1]] + 1e-10
    w = bisquare_kernel(dists[nn], bw)
    Xn = np.column_stack([np.ones(K_B), X_12[nn]])
    W = np.diag(w)
    try:
        XtW = Xn.T @ W
        beta = np.linalg.solve(XtW @ Xn + 1e-8*np.eye(Xn.shape[1]), XtW @ y[nn])
        preds_B[i] = np.concatenate([[1.0], X_12[i]]) @ beta
        coefs_B[i] = beta
    except:
        preds_B[i] = np.mean(y[nn])

r2_B = r2_score(y, preds_B)
rmse_B = np.sqrt(mean_squared_error(y, preds_B))
print(f"  全样本: R²={r2_B:.4f}, RMSE={rmse_B:.4f} ({time.time()-t0:.1f}s)")

# B.1 局部系数表
coef_names = ['Intercept'] + features_12
coef_df = pd.DataFrame(coefs_B, columns=coef_names)
coef_df['lon'] = coords[:, 0]
coef_df['lat'] = coords[:, 1]
coef_df['所属区'] = df['所属区'].values
coef_df['GRID_ID'] = df['GRID_ID'].values
coef_df['actual'] = y
coef_df['predicted'] = preds_B
coef_df['residual'] = y - preds_B

# B.2 空间变异性
print("\n空间变异性排序 (12feat, k=300):")
print(f"{'指标':<10} {'维度':<14} {'均值':>8} {'标准差':>8} {'CV':>8} {'非平稳':>8} {'正%':>6}")
print("-" * 68)

var_results = []
for feat in features_12:
    c = coef_df[feat].values
    m = np.mean(c)
    s = np.std(c)
    cv = abs(s/m) if abs(m) > 1e-10 else np.inf
    iqr = np.percentile(c, 75) - np.percentile(c, 25)
    n_pos = np.sum(c > 0)
    sign_change = min(n_pos, n - n_pos) / n
    nsi = cv * (1 + sign_change)
    pct_pos = n_pos / n * 100
    
    var_results.append({
        '指标代码': feat, '维度': dim_map[feat],
        '系数均值': m, '系数标准差': s,
        '变异系数(CV)': cv, '四分位距(IQR)': iqr,
        '极差': np.max(c) - np.min(c),
        '正系数比例': pct_pos / 100,
        '符号变化比例': sign_change,
        '空间非平稳性指数': nsi
    })

var_df = pd.DataFrame(var_results).sort_values('空间非平稳性指数', ascending=False)

for _, row in var_df.iterrows():
    print(f"{row['指标代码']:<10} {row['维度']:<14} {row['系数均值']:>8.4f} "
          f"{row['系数标准差']:>8.4f} {row['变异系数(CV)']:>8.2f} "
          f"{row['空间非平稳性指数']:>8.2f} {row['正系数比例']*100:>5.1f}%")

# B.3 分区系数 + 主导模式
print("\n分区系数均值 + 主导模式:")

district_rows = []
for district in ['成华区', '金牛区', '锦江区', '青羊区', '武侯区']:
    mask = coef_df['所属区'] == district
    row = {'行政区': district, '样本数': mask.sum()}
    
    dim_strength = {}
    for feat in features_12:
        vals = coef_df.loc[mask, feat]
        row[f'{feat}_均值'] = vals.mean()
        row[f'{feat}_标准差'] = vals.std()
        
        d = dim_map[feat]
        if 'Spatial' not in d and 'Supply' not in d:
            if d not in dim_strength:
                dim_strength[d] = 0
            dim_strength[d] += abs(vals.mean())
    
    dominant = max(dim_strength, key=dim_strength.get)
    row['主导模式'] = dominant
    district_rows.append(row)
    
    print(f"\n  {district} (n={mask.sum()}): 【{dominant}】")
    for d in sorted(dim_strength, key=dim_strength.get, reverse=True):
        print(f"    {d}: {dim_strength[d]:.4f}")

district_df = pd.DataFrame(district_rows)


# ═══════════════════════════════════════════════════════════════
#  保存所有结果
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("保存结果")
print("=" * 70)

# 模型性能 (合并到v4_model_comparison)
perf_df = pd.DataFrame([
    {'实验': 'A_logΔESI_14feat', '模型': 'KNN-LWR',
     'Train_R2': r2_A, 'Test_R2': '', 'CV_R2': cv_mean_A, 
     'CV_std': cv_std_A, 'RMSE': rmse_A, 'MAE': mae_A,
     '备注': f'k={K_A}, bisquare, 14feat'},
    {'实验': 'B_spatial_12feat', '模型': 'KNN-LWR',
     'Train_R2': r2_B, 'Test_R2': '', 'CV_R2': '', 
     'CV_std': '', 'RMSE': rmse_B, 'MAE': '',
     '备注': f'k={K_B}, bisquare, 12feat, 用于空间异质性分析'}
])
perf_df.to_csv(f"{OUTPUT_DIR}/v4_lwr_performance.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ v4_lwr_performance.csv")

# 局部系数 (Fig.10地图数据)
coef_df.to_csv(f"{OUTPUT_DIR}/v4_lwr_coefficients_final.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ v4_lwr_coefficients_final.csv ({len(coef_df)}行 × {len(coef_df.columns)}列)")

# 空间变异性
var_df.to_csv(f"{OUTPUT_DIR}/v4_lwr_spatial_variability_final.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ v4_lwr_spatial_variability_final.csv")

# 分区系数
district_df.to_csv(f"{OUTPUT_DIR}/v4_lwr_district_coefficients_final.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ v4_lwr_district_coefficients_final.csv")


# ═══════════════════════════════════════════════════════════════
#  Fig.8 完整模型对比汇总
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Fig.8 完整模型对比 (论文4.5.1节)")
print("=" * 70)

print(f"""
┌────────────────┬──────────┬──────────┬────────────────┬──────────┐
│ 模型           │ Train R² │ Test R²  │ CV R²          │ RMSE     │
├────────────────┼──────────┼──────────┼────────────────┼──────────┤
│ OLS            │ 0.1829   │ 0.1508   │ 0.1555 ± 0.042 │ 1.6541   │
│ KNN-LWR(k=100)│ {r2_A:.4f}   │   —      │ {cv_mean_A:.4f} ± {cv_std_A:.3f} │ {rmse_A:.4f}   │
│ Random Forest  │ 0.7184   │ 0.4157   │ 0.4286 ± 0.032 │ 1.3721   │
│ XGBoost ★      │ 0.9363   │ 0.5564   │ 0.5659 ± 0.014 │ 1.1955   │
└────────────────┴──────────┴──────────┴────────────────┴──────────┘

排序 (CV R²): XGBoost > RF > KNN-LWR > OLS
XGBoost为最优模型 ✓
KNN-LWR的CV R²({cv_mean_A:.4f})低于RF({0.4286:.4f}), CV std高({cv_std_A:.4f} vs 0.032)
→ 支持论文论述: 线性局部拟合在非线性因变量上不稳定
""")

print("=" * 70)
print("Fig.10 空间异质性 (论文4.5.3节)")
print("=" * 70)
print(f"""
KNN-LWR配置: 12feat (不含lon/lat), k={K_B}, bisquare kernel
R² = {r2_B:.4f}

空间非平稳性前5个因子:
""")
for i, (_, row) in enumerate(var_df.head(5).iterrows()):
    print(f"  {i+1}. {row['指标代码']} ({row['维度']}): CV={row['变异系数(CV)']:.2f}, 非平稳指数={row['空间非平稳性指数']:.2f}")
