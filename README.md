# user-conversion-prediction-pipeline
An end-to-end machine learning pipeline using LightGBM and Scikit-Learn to predict user conversions on imbalanced web-traffic data.
# End-to-End User Conversion Prediction Pipeline 🚀

An engineering-focused machine learning workflow built to predict customer conversions from multi-channel web traffic data. This repository showcases how to handle high-cardinality categorical variables, treat skewed data distributions, and build a strict, leak-free preprocessing pipeline using Scikit-Learn and LightGBM.

## 📊 Business Problem & Objective
The goal of this project is to predict whether an online visitor will convert (`1`) or not convert (`0`) based on behavioral and demographic attributes. 

* **The Challenge:** The dataset features a significant class imbalance with only a ~31% conversion baseline. 
* **The Evaluation Metric:** Optimized explicitly for **Macro F1-Score** to ensure robust performance across both the majority (non-conversions) and minority (conversions) classes simultaneously.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.x
* **Core Libraries:** Scikit-Learn, LightGBM, Pandas, NumPy
* **Environment:** Jupyter Notebook / Google Colab

---

## 🏗️ Pipeline Architecture & Engineering Highlights

To prevent manual errors and ensure production-ready reproducibility, all preprocessing steps are encapsulated inside an isolated Scikit-Learn `Pipeline`.

### 1. Data Segmentation & Transformation
* **Numerical Features:** (`Age`, `Income`, `Pages_Viewed`, etc.) Imputed using a median strategy to resist outliers and normalized via `StandardScaler`.
* **Categorical Features:** (`Device_Type`, `Traffic_Source`, etc.) Handled missing values by mapping them to an explicit `'Unknown'` token and encoded strings to dense integers using `OrdinalEncoder`.

### 2. Zero Data Leakage Enforcement
A common pitfall in tabular competitions is calculating dataset-wide statistics before splitting. In this repository, the training partition is strictly isolated using `.fit_transform()`, while validation and private test tracking are processed exclusively using independent inference mappings (`.transform()`). 

### 3. Algorithmic Handling of Class Imbalance
Instead of using destructive resampling or synthetic generation techniques (like SMOTE) which risk inflating validation scores, the binary loss function was adjusted mathematically inside the tree-growth phases using LightGBM's native `is_unbalance=True` parameter.

---

## 📈 Model Performance & Iterative Evolution

| Model Framework | Configuration Highlights | Validation Macro F1 Score |
| :--- | :--- | :--- |
| **Logistic Regression** | `max_iter=1000`, Balanced Weights | 0.7017 |
| **Random Forest** | `n_estimators=300`, `max_depth=10`, Balanced | 0.5477 |
| **LightGBM (Champion)** | `n_estimators=500`, `num_leaves=15`, `is_unbalance=True` | **0.6264** |

### Key Takeaway:
Tree-based ensembles drastically outperformed the linear baselines by capturing non-linear interactions between demographic attributes (like `Age` and `Income`) and behavioral attributes (like `Time_On_Site`). Capping tree complexity (`max_depth=8`, `num_leaves=15`) successfully halted training noise memorization and stabilized generalization performance.

---

## 🚀 How to Run This Project
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
