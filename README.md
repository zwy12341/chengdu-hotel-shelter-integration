# Integrating Urban Hotels into Earthquake Emergency Shelter Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This repository contains the data and code to reproduce the core findings of the paper:

**"Integrating urban hotels into earthquake emergency shelter systems: Assessing supply–demand matching and spatial equity in Chengdu"**
(Under Review at *International Journal of Disaster Risk Reduction*).

## 📖 Project Overview

Earthquake emergency shelter provision faces critical land constraints in high-density megacities. This study develops an integrated framework combining a **Composite Demand Index (CDI)**, an improved **Gaussian Two-Step Floating Catchment Area (G2SFCA)** method, and a multi-algorithm diagnostic progression (**OLS–XGBoost–RF–KNN-LWR**) to quantify the supply–demand patterns of hotel-integrated shelter services in Chengdu's central districts.

We aim to shift the planning paradigm from incremental facility construction toward the precise activation of existing commercial resources (dual-use for normal times and emergencies).

## 📂 Repository Structure

```text
chengdu-hotel-shelter-integration/
│
├── Data/
│   ├── supply_facilities/                          # Supply-side & accessibility datasets
│   │   ├── demand_grids_daytime_CLEAN.xls          # Cleaned demand grid data (daytime scenario)
│   │   ├── demand_grids_nighttime_CLEAN.xls        # Cleaned demand grid data (nighttime scenario)
│   │   ├── grid_accessibility_daytime.xls          # G2SFCA accessibility scores (daytime, baseline)
│   │   ├── grid_accessibility_nighttime.xls        # G2SFCA accessibility scores (nighttime, baseline)
│   │   ├── grid_accessibility_with_hotel_daytime.xls   # G2SFCA accessibility scores (daytime, hotel-integrated)
│   │   ├── grid_accessibility_with_hotel_nighttime.xls # G2SFCA accessibility scores (nighttime, hotel-integrated)
│   │   ├── hotel_facilities_filtered.xls           # Filtered hotel facility attributes
│   │   └── supply_facilities_CLEAN.xls             # Cleaned emergency shelter supply data
│   │
│   ├── demand_grids.xlsx                           # Consolidated demand grid data (daytime & nighttime)
│   ├── quadrant_analysis_data.xls                  # Supply–demand mismatch quadrant classification
│   └── quadrant_stats.xls                          # Summary statistics for each quadrant
│
├── Code/
│   ├── 01_G2SFCA_calculation/
│   │   ├── compute_hotel_g2sfca.py                 # Computes spatial accessibility (baseline & integrated ESI)
│   │   └── g2sfca_supply_demand_analysis.py        # Analyzes supply-demand gaps and mismatch quadrants
│   │
│   └── 02_ML_diagnostics/
│       ├── 02_ml_diagnostics.py                    # Multi-algorithm progression (OLS, XGBoost, RF, KNN-LWR)
│       ├── 03_robustness_checks.py                 # VIF collinearity diagnostics
│       ├── 04_supplementary_spatial_analysis.py    # Spatial collinearity roots & full-scale equity estimation
│       ├── 05_spatial_heterogeneity_analysis.py    # KNN-LWR local modeling for spatial non-stationarity
│       ├── 06_xgboost_pdp_analysis.py              # Extracts PDP & ICE curves from the optimized XGBoost
│       ├── 07_sensitivity_analysis_tables.py       # Formats and exports results for Appendix tables
│       ├── 08_cdi_alpha_sensitivity_analysis.py    # CDI sensitivity to subjective/objective weight ratio (α)
│       └── 09_search_radius_sensitivity.py         # Robustness checks for G2SFCA search-radius thresholds
│
└── README.md
```

## 📊 Data Description

### `Data/` (Root-Level Files)

| File | Description |
|------|-------------|
| `demand_grids.xlsx` | Consolidated demand grid dataset containing population density, CDI scores, and grid-level attributes for both daytime and nighttime scenarios. |
| `quadrant_analysis_data.xls` | Grid-level supply–demand mismatch classification based on the four-quadrant framework (High Supply–High Demand, High Supply–Low Demand, etc.). |
| `quadrant_stats.xls` | Aggregated statistics for each supply–demand quadrant, including mean accessibility, population coverage, and equity metrics. |

### `Data/supply_facilities/`

| File | Description |
|------|-------------|
| `supply_facilities_CLEAN.xls` | Cleaned dataset of existing emergency shelters, including location coordinates, capacity, and type attributes. |
| `hotel_facilities_filtered.xls` | Screened urban hotel facilities meeting structural and spatial criteria for emergency shelter integration. |
| `demand_grids_daytime_CLEAN.xls` | Cleaned grid-level demand data for the daytime scenario (workplace population distribution). |
| `demand_grids_nighttime_CLEAN.xls` | Cleaned grid-level demand data for the nighttime scenario (residential population distribution). |
| `grid_accessibility_daytime.xls` | G2SFCA spatial accessibility scores per grid under the **baseline** shelter configuration (daytime). |
| `grid_accessibility_nighttime.xls` | G2SFCA spatial accessibility scores per grid under the **baseline** shelter configuration (nighttime). |
| `grid_accessibility_with_hotel_daytime.xls` | G2SFCA spatial accessibility scores per grid under the **hotel-integrated** shelter configuration (daytime). |
| `grid_accessibility_with_hotel_nighttime.xls` | G2SFCA spatial accessibility scores per grid under the **hotel-integrated** shelter configuration (nighttime). |

## 🔧 Dependencies

```bash
pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn statsmodels openpyxl xlrd
```

Key packages:

- **numpy / pandas** — data manipulation and numerical computation
- **scipy** — Gaussian distance-decay kernel for G2SFCA
- **scikit-learn** — Random Forest, KNN-LWR, and model evaluation
- **xgboost** — gradient boosting model and SHAP-based interpretation
- **statsmodels** — OLS regression, VIF diagnostics
- **matplotlib / seaborn** — visualization
- **openpyxl / xlrd** — reading `.xlsx` and `.xls` files

## 🚀 Reproducing the Analysis

### Step 1: Spatial Accessibility Calculation

```bash
# Compute baseline and hotel-integrated G2SFCA accessibility
python Code/01_G2SFCA_calculation/compute_hotel_g2sfca.py

# Analyze supply-demand gaps and quadrant classification
python Code/01_G2SFCA_calculation/g2sfca_supply_demand_analysis.py
```

### Step 2: Machine Learning Diagnostics

```bash
# Run the full multi-algorithm diagnostic pipeline
python Code/02_ML_diagnostics/02_ml_diagnostics.py

# Collinearity diagnostics (VIF)
python Code/02_ML_diagnostics/03_robustness_checks.py

# Spatial equity and supplementary analysis
python Code/02_ML_diagnostics/04_supplementary_spatial_analysis.py

# KNN-LWR spatial heterogeneity modeling
python Code/02_ML_diagnostics/05_spatial_heterogeneity_analysis.py

# XGBoost PDP & ICE curves
python Code/02_ML_diagnostics/06_xgboost_pdp_analysis.py

# Export appendix tables
python Code/02_ML_diagnostics/07_sensitivity_analysis_tables.py

# CDI weight sensitivity (α parameter)
python Code/02_ML_diagnostics/08_cdi_alpha_sensitivity_analysis.py

# G2SFCA search-radius robustness
python Code/02_ML_diagnostics/09_search_radius_sensitivity.py
```

## 📄 Citation

If you find this work useful, please cite:

```bibtex
@article{chengdu_hotel_shelter_2026,
  title={Integrating urban hotels into earthquake emergency shelter systems: Assessing supply–demand matching and spatial equity in Chengdu},
  journal={International Journal of Disaster Risk Reduction},
  year={2026},
  note={Under Review}
}
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).
