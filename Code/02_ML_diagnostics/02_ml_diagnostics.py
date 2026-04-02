"""
================================================================================
ΔESI驱动机制分析 — v6 超参数优化版
================================================================================
修复内容 (相比v5):
  v5正则化过于激进 (CV R² 从v4的0.566降至0.466)，本版通过系统网格搜索
  找到正则化与拟合之间的最优平衡点。

  参数调整 (v5 → v6):
    n_estimators:     100 → 250   # 增加树数量，配合较低学习率
    max_depth:          5 → 6     # 恢复v4深度，允许捕获更多非线性
    learning_rate:   0.08 → 0.06  # 降低学习率，配合更多树
    min_child_weight:   5 → 3     # 适度放松叶节点约束
    reg_alpha:        1.0 → 0.3   # 减弱L1正则化 (v5过于激进)
    reg_lambda:       1.0 → 2.0   # 加强L2正则化 (比L1更平滑)
    subsample:        0.7 → 0.75  # 适度提高采样比例
    colsample_bytree: 0.7 → 0.75  # 适度提高列采样

  预期效果:
    Train R²: 0.713 → ~0.85  (v4=0.936, 仍有充分正则化)
    CV R²:    0.466 → ~0.55  (接近v4的0.566, 远超v5)
    Gap:      0.247 → ~0.30  (合理的泛化差距)

输入: v4_data_full.csv (3700行, 昼间E11, 14特征)
输出: v6_fig8.png, v6_model_comparison.csv, v6_feature_importance.csv
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# 全局配置
# ═══════════════════════════════════════════════════
RANDOM_STATE = 42
TEST_SIZE = 0.3      # 论文中写的 7:3
CV_FOLDS = 5
K_NEIGHBORS = 150    # KNN-LWR邻居数

# XGBoost v6 优化参数 (通过系统网格搜索选定)
XGB_V6_PARAMS = dict(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.06,
    min_child_weight=3,
    reg_alpha=0.3,
    reg_lambda=2.0,
    subsample=0.75,
    colsample_bytree=0.75,
)

FEAT_14 = ['H11','H12','E11','E12','E21','E31',
           'V11','V21','V22','V31','V32',
           'ESI_base','lon_f','lat_f']

DIMENSION_MAP = {
    'H11': 'H(危险性)', 'H12': 'H(危险性)',
    'E11': 'E(暴露度)', 'E12': 'E(暴露度)',
    'E21': 'E(暴露度)', 'E31': 'E(暴露度)',
    'V11': 'V(脆弱性)', 'V21': 'V(脆弱性)',
    'V22': 'V(脆弱性)', 'V31': 'V(脆弱性)', 'V32': 'V(脆弱性)',
    'ESI_base': 'Supply(供给基线)',
    'lon_f': 'Spatial(空间)', 'lat_f': 'Spatial(空间)',
}


# ═══════════════════════════════════════════════════
# KNN-LWR 实现
# ═══════════════════════════════════════════════════
def knn_lwr_predict(X_fit, y_fit, X_pred, k=K_NEIGHBORS):
    """KNN-LWR: 对每个预测点, 用K个最近邻构建局部加权线性回归"""
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(X_fit)
    dists, idxs = nn.kneighbors(X_pred)
    preds = np.zeros(len(X_pred))
    n_feat = X_fit.shape[1]
    
    for i in range(len(X_pred)):
        nb_dist = dists[i]
        bw = nb_dist[-1] + 1e-10
        w = np.exp(-0.5 * (nb_dist / bw)**2)
        w = w / (w.sum() + 1e-10)
        
        X_nb = X_fit[idxs[i]]
        y_nb = y_fit[idxs[i]]
        W = np.diag(w)
        X_aug = np.column_stack([np.ones(k), X_nb])
        
        try:
            beta = np.linalg.solve(
                X_aug.T @ W @ X_aug + 1e-5 * np.eye(n_feat + 1),
                X_aug.T @ W @ y_nb
            )
            preds[i] = np.concatenate([[1], X_pred[i]]) @ beta
        except:
            preds[i] = np.average(y_nb, weights=w)
    
    return preds


# ═══════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════
def main():
    print("="*70)
    print("  v6: ΔESI驱动机制分析 (超参数优化版)")
    print("="*70)
    
    # ── 1. 数据加载 ──
    df = pd.read_csv('v4_data_full.csv')
    if 'ESI_base' not in df.columns: df['ESI_base'] = df['ESI']
    if 'lon_f' not in df.columns: df['lon_f'] = df['lon']
    if 'lat_f' not in df.columns: df['lat_f'] = df['lat']
    
    X = df[FEAT_14].values
    y = df['log_dESI_w'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n  样本: {len(X)} | 训练: {len(X_train)} | 测试: {len(X_test)}")
    
    # ── 2. OLS ──
    print("\n  [1/4] OLS (Linear)...")
    ols = LinearRegression().fit(X_train, y_train)
    ols_cv = cross_val_score(LinearRegression(), X, y, cv=CV_FOLDS, scoring='r2', n_jobs=-1)
    ols_r = dict(
        name='OLS (Linear)',
        train_r2=round(r2_score(y_train, ols.predict(X_train)), 3),
        test_r2=round(r2_score(y_test, ols.predict(X_test)), 3),
        rmse=round(np.sqrt(mean_squared_error(y_test, ols.predict(X_test))), 2),
        mae=round(mean_absolute_error(y_test, ols.predict(X_test)), 2),
        cv_mean=round(ols_cv.mean(), 3),
        cv_std=round(ols_cv.std(), 3),
    )
    print(f"    Train R²={ols_r['train_r2']}, CV R²={ols_r['cv_mean']}±{ols_r['cv_std']}")
    
    # ── 3. KNN-LWR ──
    print("\n  [2/4] KNN-LWR...")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    y_knn_train = knn_lwr_predict(X_train_sc, y_train, X_train_sc)
    y_knn_test = knn_lwr_predict(X_train_sc, y_train, X_test_sc)
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    knn_cv_scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_sc)):
        yp = knn_lwr_predict(X_sc[tr_idx], y[tr_idx], X_sc[val_idx])
        knn_cv_scores.append(r2_score(y[val_idx], yp))
        print(f"    Fold {fold+1}: R²={knn_cv_scores[-1]:.4f}")
    
    knn_cv = np.array(knn_cv_scores)
    knn_r = dict(
        name='GWR (KNN-LWR)',
        train_r2=round(r2_score(y_train, y_knn_train), 3),
        test_r2=round(r2_score(y_test, y_knn_test), 3),
        rmse=round(np.sqrt(mean_squared_error(y_test, y_knn_test)), 2),
        mae=round(mean_absolute_error(y_test, y_knn_test), 2),
        cv_mean=round(knn_cv.mean(), 3),
        cv_std=round(knn_cv.std(), 3),
    )
    print(f"    Train R²={knn_r['train_r2']}, CV R²={knn_r['cv_mean']}±{knn_r['cv_std']}")
    
    # ── 4. Random Forest ──
    print("\n  [3/4] Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    rf_cv = cross_val_score(
        RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
                              random_state=RANDOM_STATE, n_jobs=-1),
        X, y, cv=CV_FOLDS, scoring='r2', n_jobs=-1
    )
    rf_r = dict(
        name='Random Forest',
        train_r2=round(r2_score(y_train, rf.predict(X_train)), 3),
        test_r2=round(r2_score(y_test, rf.predict(X_test)), 3),
        rmse=round(np.sqrt(mean_squared_error(y_test, rf.predict(X_test))), 2),
        mae=round(mean_absolute_error(y_test, rf.predict(X_test)), 2),
        cv_mean=round(rf_cv.mean(), 3),
        cv_std=round(rf_cv.std(), 3),
    )
    print(f"    Train R²={rf_r['train_r2']}, CV R²={rf_r['cv_mean']}±{rf_r['cv_std']}")
    
    # ── 5. XGBoost (v6优化) ──
    print("\n  [4/4] XGBoost (v6 optimized)...")
    xgb = XGBRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **XGB_V6_PARAMS
    ).fit(X_train, y_train)
    xgb_cv = cross_val_score(
        XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0, **XGB_V6_PARAMS),
        X, y, cv=CV_FOLDS, scoring='r2', n_jobs=-1
    )
    xgb_r = dict(
        name='XGBoost',
        train_r2=round(r2_score(y_train, xgb.predict(X_train)), 3),
        test_r2=round(r2_score(y_test, xgb.predict(X_test)), 3),
        rmse=round(np.sqrt(mean_squared_error(y_test, xgb.predict(X_test))), 2),
        mae=round(mean_absolute_error(y_test, xgb.predict(X_test)), 2),
        cv_mean=round(xgb_cv.mean(), 3),
        cv_std=round(xgb_cv.std(), 3),
    )
    print(f"    Train R²={xgb_r['train_r2']}, CV R²={xgb_r['cv_mean']}±{xgb_r['cv_std']}")
    
    # ── 6. 特征重要性 ──
    fi = xgb.feature_importances_
    fi_pct = fi / fi.sum() * 100
    imp = dict(zip(FEAT_14, fi_pct))
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
    
    dims = {}
    for f, v in imp.items():
        d = DIMENSION_MAP.get(f, '?')
        dims[d] = dims.get(d, 0) + v
    
    print(f"\n  XGBoost 特征重要性:")
    for i, (f, v) in enumerate(sorted_imp):
        print(f"    {i+1:2d}. {f:<12}{v:5.1f}% [{DIMENSION_MAP.get(f,'')}]")
    
    print(f"\n  维度贡献:")
    for d in ['V(脆弱性)', 'E(暴露度)', 'Spatial(空间)', 'H(危险性)', 'Supply(供给基线)']:
        print(f"    {d}: {dims.get(d,0):.1f}%")
    
    # ── 7. v4/v5/v6 对比 ──
    print(f"\n{'='*75}")
    print(f"  版本对比:")
    print(f"  {'版本':<20}{'Train R²':>10}{'CV R²':>16}{'Gap':>8}")
    print(f"  {'-'*50}")
    print(f"  {'v4 (原版)':20s}{'0.936':>10}{'0.566±0.014':>16}{'0.370':>8}")
    print(f"  {'v5 (过度正则化)':20s}{'0.713':>10}{'0.466±0.022':>16}{'0.247':>8}")
    print(f"  {'v6 (优化版)':20s}{xgb_r['train_r2']:>10.3f}  {xgb_r['cv_mean']:.3f}±{xgb_r['cv_std']:.3f}    {xgb_r['train_r2']-xgb_r['cv_mean']:.3f}")
    
    # ── 8. 汇总 ──
    models = [ols_r, knn_r, rf_r, xgb_r]
    
    print(f"\n{'='*75}")
    print(f"  {'Model':<20}{'Train R²':>10}{'Test R²':>10}{'CV R²':>16}{'RMSE':>8}{'MAE':>8}")
    print(f"  {'-'*70}")
    for m in models:
        print(f"  {m['name']:<20}{m['train_r2']:>10.3f}{m['test_r2']:>10.3f}"
              f"  {m['cv_mean']:.3f}±{m['cv_std']:.3f}{m['rmse']:>8.2f}{m['mae']:>8.2f}")
    
    # ── 9. 出图 ──
    print("\n  生成Figure 8...")
    plot_fig8(models)
    
    # ── 10. 保存 ──
    pd.DataFrame(models).to_csv('v6_model_comparison.csv', index=False)
    
    imp_rows = [{'Feature': f, 'Importance_pct': round(v, 2),
                 'Dimension': DIMENSION_MAP.get(f, '')} for f, v in sorted_imp]
    pd.DataFrame(imp_rows).to_csv('v6_feature_importance.csv', index=False)
    
    print("\n  ✓ 所有文件已保存")
    return models, sorted_imp, dims


def plot_fig8(models):
    """生成论文Figure 8"""
    names = [m['name'] for m in models]
    colors = ['#7EB6D9', '#1A5276', '#E67E22', '#C0392B']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    
    # (a) R²
    ax = axes[0, 0]
    vals = [m['train_r2'] for m in models]
    bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.axhline(0.9, color='green', ls='--', alpha=0.4)
    ax.axhline(0.8, color='orange', ls='--', alpha=0.4)
    ax.text(3.45, 0.91, 'Excellent (0.9)', fontsize=8, color='green', alpha=0.6, ha='right')
    ax.text(3.45, 0.81, 'Good (0.8)', fontsize=8, color='orange', alpha=0.6, ha='right')
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('(a) Model Explanatory Power (R²)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=12, labelsize=9)
    
    # (b) RMSE
    ax = axes[0, 1]
    vals = [m['rmse'] for m in models]
    bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('(b) Prediction Error (RMSE)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.2)
    ax.tick_params(axis='x', rotation=12, labelsize=9)
    
    # (c) MAE
    ax = axes[1, 0]
    vals = [m['mae'] for m in models]
    bars = ax.bar(names, vals, color=colors, width=0.6, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('(c) Mean Absolute Error (MAE)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.2)
    ax.tick_params(axis='x', rotation=12, labelsize=9)
    
    # (d) CV R²
    ax = axes[1, 1]
    cv_vals = [m['cv_mean'] for m in models]
    cv_errs = [m['cv_std'] for m in models]
    bars = ax.bar(names, cv_vals, color=colors, width=0.6, edgecolor='white',
                  yerr=cv_errs, capsize=5, error_kw={'linewidth':1.5, 'color':'black'})
    for i, (bar, val, err) in enumerate(zip(bars, cv_vals, cv_errs)):
        ax.text(bar.get_x()+bar.get_width()/2, val+err+0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    best_i = np.argmax(cv_vals)
    ax.text(bars[best_i].get_x()+bars[best_i].get_width()/2,
            cv_vals[best_i]+cv_errs[best_i]+0.045,
            'Best', ha='center', fontsize=10, fontweight='bold', color='red')
    ax.set_ylabel('Cross-Validation R²', fontsize=12)
    ax.set_title('(d) 5-Fold Cross-Validation R²', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 0.65)
    ax.tick_params(axis='x', rotation=12, labelsize=9)
    
    plt.savefig('v6_fig8.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('v6_fig8.pdf', bbox_inches='tight', facecolor='white')
    print("    ✓ v6_fig8.png / .pdf")


if __name__ == '__main__':
    models, sorted_imp, dims = main()
