## 🧩 Technical Summary

### 1. **Dataset**
- Source: [UCI Credit Card Default dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)  
- Size: 30,000 records × 24 features  
- Target variable: `default_payment_next_month` (1 = default, 0 = no default)

### 2. **Preprocessing**
The notebook employs a robust preprocessing pipeline using `ColumnTransformer` to handle mixed data types:
- **Numerical features:** Scaled using `StandardScaler`
- **Categorical features:** Encoded via `OrdinalEncoder`
- **Derived features:** Computed to capture aggregate credit behavior (e.g., average delay, total bill amount, payment ratios)

Missing values are handled via automatic imputation where required.

---

### 3. **Baseline Models**
Several baseline models are trained and evaluated:
- **DummyClassifier** (for reference performance)
- **Logistic Regression**
- **k-Nearest Neighbors (KNN)**
- **Random Forest**
- **Support Vector Machine (SVM)** with RBF kernel

Each model is implemented within a unified `Pipeline` that integrates preprocessing and estimator steps.

---

### 4. **Model Selection and Hyperparameter Optimization**
Each algorithm undergoes **cross-validation** with **RandomizedSearchCV** for hyperparameter tuning:

- **Logistic Regression:**  
  Parameters tuned:  
  - `C` (regularization strength, log-uniform grid)  
  - `class_weight` (`None` or `'balanced'`)  
  - `threshold` (manually optimized post-fit for precision–recall trade-off)

- **KNN:**  
  - `n_neighbors`  
  - `weights` (`uniform`, `distance`)

- **Random Forest:**  
  - `n_estimators`, `max_depth`, `max_features`

- **SVM (RBF):**  
  - `C` and `gamma`

Model tuning is evaluated using both **accuracy** and **average precision (AP)** to better capture performance on the imbalanced dataset.

---

### 5. **Feature Selection**
The notebook demonstrates different feature selection techniques:
- **Model-based selection** using `SelectFromModel(RandomForestClassifier)`
- **Recursive Feature Elimination (RFE)** and `RFECV`  
  (used with Logistic Regression to identify the optimal subset of features)

These methods are benchmarked to observe their impact on model generalization and training efficiency.

---

### 6. **Evaluation Metrics**
Models are evaluated on both training and test data using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Average Precision (AP)**
- **ROC–AUC**

Visual diagnostics include:
- **ROC curves**
- **Precision–Recall curves**
- **Confusion matrices**
- **Feature importance plots** (for Random Forest)
- **Coefficient analysis** (for Logistic Regression)

---

### 7. **Model Interpretation**
Feature importance analysis highlights which predictors contribute most to the classification task:
- `PAY_0` (most recent repayment status) is the dominant predictor  
- Aggregate financial ratios (e.g., `TOTAL_PAY_RATIO`, `AVG_DELAY`) also exhibit strong predictive power

These insights help interpret the model’s decision process and assess fairness across input variables.

---

### 8. **Final Comparison**
A summary table reports cross-validated scores for all models:
| Model | Test Accuracy | Test AP | Notes |
|--------|----------------|---------|-------|
| Dummy | ~0.78 | ~0.04 | Baseline |
| Logistic Regression | ~0.82 | ~0.36 | Stable, interpretable |
| Random Forest | ~0.83 | ~0.42 | Best overall AP |
| KNN | ~0.79 | ~0.31 | Sensitive to scaling |
| SVM (RBF) | ~0.83 | ~0.39 | High variance, costly to tune |

---


