# Instacart Customer Segmentation — clustering grocery shoppers into behavioral segments

CMPE 255 Data Mining (Section 01), San José State University · Spring 2026 · Team: Maxim Dokukin, Megha Gangal, Mansi Gupta · Status: Completed

## Overview

Instacart's public grocery basket dataset holds 3.4 million orders and 32.4 million order-product rows from 206,209 customers, but no customer-level view. This project turns the transaction logs into one behavioral profile per customer (purchase frequency, basket size, reorder rate, days between orders, tenure, recency, time-of-day and category concentration), checks the feature space for skew, redundancy and outliers, and compares four clustering families: K-Means, agglomerative (Ward) clustering, DBSCAN and Gaussian Mixture Models. K-Means, hierarchical clustering and GMM all found four interpretable segments (power users, loyal regulars, casual shoppers, new/inactive users). DBSCAN put almost every customer into one density group. The team picked GMM as the primary model because its soft membership fits customer behavior that shifts gradually from one segment to the next.

## Highlights

- 206,209 customers profiled from 3,421,083 orders and 32,434,489 prior order-product rows (`EDA.ipynb`, `notebooks/04-06_clustering_max.ipynb`)
- K-Means silhouette swept for k = 2–9 (0.279 → 0.205); k = 4 chosen for interpretability at 0.2574 (`models/K-Means Model/MODEL_1_KMeans.ipynb`)
- Full-dataset runs on HPC: hierarchical and GMM both give 4 clusters; DBSCAN gives 1 cluster + 345 noise points; GMM silhouette ≈ 0.25 (final deck, slide 4)
- Outlier consensus: Mahalanobis, Isolation Forest and LOF each flag 4,125 customers (2%), but only 118 (0.06%) are flagged by all three (`notebooks/04_outlier_analysis.ipynb`)
- A customer feature mart with 15 candidate features and an automated keep/log/clip policy: 14 features kept; 9 PCA components retain 90.7% of the variance (`notebooks/04-06_clustering_max.ipynb`)

## How it works

```
Kaggle CSVs (orders, order_products__prior/train, products, aisles, departments)
  → EDA (volume, peak hours, reorder gaps, top departments/aisles/products)
  → customer-level features (frequency, basket size, reorder rate, gaps, tenure, recency, hour/weekday, category HHI)
  → audit (skew, IQR outliers, |r| ≥ 0.92 redundancy) → log1p / 1–99% clipping → StandardScaler
  → outlier detection (IQR, z-score, Mahalanobis, Isolation Forest, LOF) and PCA check
  → K-Means · Hierarchical (Ward) · DBSCAN · GMM → silhouette / Davies-Bouldin / cluster profiles
  → four named segments → marketing actions
```

- **Data loading** (`dataLoading.py`): downloads the Kaggle dataset with `kagglehub` into `./data` if files are missing, then loads all six CSVs.
- **Preprocessing** (`dataPreprocessing.py`): fills the first-order `days_since_prior_order` gap with 0, samples 50,000 orders for a DBSCAN outlier check, and standardizes the order columns.
- **Feature engineering** (`featureEngineering.py`, `notebooks/03_feature_engineering.ipynb`): builds user-level features from the leakage-safe prior orders only, plus a 45-feature user–product matrix (8.47 M rows).
- **Clustering workbench** (`notebooks/04-06_clustering_max.ipynb`): customer feature mart, feature audit and retention policy, scaled and PCA views, a parameter sweep over all four algorithms with subsample and seed stability (ARI), and a composite model-selection score.
- **PCA** (`pca.py`, `notebooks/04_ExploringPCA.ipynb`): cumulative explained variance and 90% thresholding. The full feature set was kept for clustering.
- **Outlier analysis** (`notebooks/04_outlier_analysis.ipynb`): univariate and multivariate detectors, agreement matrix, consensus profile.
- **Models** (`models/`): K-Means, hierarchical and DBSCAN notebooks on customer subsets, plus `*_hpc.py` scripts that run DBSCAN, GMM and hierarchical clustering on all 206,209 customers.

## Results

| Model | Customers | Clusters | Silhouette | Role in the final recommendation |
|---|---|---|---|---|
| K-Means (k = 4) | 188,816 | 4 | 0.2574 | Baseline; k chosen from the elbow and silhouette sweep |
| Hierarchical (Ward) | 5,000 sample / 206,209 on HPC | 4 | 0.1712 (sample) | Validation of k = 4 via dendrogram |
| DBSCAN (eps 0.8, min_samples 10) | 206,209 on HPC | 1 + 345 noise | n/a | Outlier monitor, not a segmentation |
| GMM (4 components, full covariance) | 206,209 on HPC | 4 (soft) | ≈ 0.25 | Primary model |

The low silhouette values (0.17–0.28) show that customer behavior forms a continuum, not well-separated groups. The same four behavioral profiles recur across models: power users (an order every ~5 days, ~73% reorder), loyal regulars (~10–12 days, ~55–64%), casual shoppers (~18–19 days, ~28–34%) and new or inactive users (near-zero reorder), per final-deck slide 12. The GMM split is 41.5% disengaged, 21.6% occasional, 24.8% regular and 12.4% loyal (deck slide 6). Sources: model notebooks and `docs/slides.pdf`. The HPC scripts print their metrics, but no run log is committed.

## Getting started

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt        # numpy, pandas, scikit-learn, matplotlib, seaborn, kagglehub, kneed, nbformat

python dataLoading.py                  # downloads the Kaggle dataset into ./data (needs Kaggle access via kagglehub)
python dataVisualization.py            # EDA figures → ./visualizations
jupyter notebook notebooks/            # 02_eda → 03_feature_engineering → 04-06_clustering_max, 04_ExploringPCA, 04_outlier_analysis
python "models/GMM Model/gmm_model_hpc.py"   # full-dataset runs; expect orders.csv and order_products__prior.csv in the working directory
```

Requirements: Python 3 with the packages above; about 32 M prior order rows are loaded into memory, so full-dataset hierarchical clustering and DBSCAN need an HPC-class machine. `data/` is git-ignored. Note: `main.py` imports a `clustering` module that is not in the repository, so run the steps individually as shown.

Dataset: [Instacart Online Grocery Basket Analysis (Kaggle)](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset). The project proposal also listed Mall Customers, UCI Online Retail, E-commerce Customer Behaviour and Customer Personality Analysis as candidate datasets. Only Instacart was used.

## Documents

- [Slides](docs/slides.pdf) — 12-slide final presentation
- Notebooks: [`EDA.ipynb`](EDA.ipynb), [`notebooks/`](notebooks/), [`models/`](models/)
- Figures: [`visualizations/`](visualizations/)
- References: Han, Kamber & Pei, *Data Mining: Concepts and Techniques*; Wedel & Kamakura, *Market Segmentation*; Agrawal & Srikant (1994), "Fast Algorithms for Mining Association Rules", VLDB; Jain (2010), "Data Clustering: 50 Years Beyond K-Means", *Pattern Recognition Letters* 31(8); Rousseeuw (1987), "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis", *J. Comput. Appl. Math.* 20
