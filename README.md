# Algerian Forest Fires Regression Model

This project predicts the Fire Weather Index (FWI) using various regression models on the Algerian Forest Fires dataset. It includes correlation-based feature selection, feature scaling, and multiple regression models with cross-validation.

## 🔧 Workflow Breakdown

### 1. Data Preprocessing
- Reads `Algerian_forest_fires_dataset_UPDATE.csv` (skips first row).
- Strips column names, drops `day`, `month`, `year`.
- Encodes `Classes`: `not fire` → 1, `fire` → 0.
- Converts all columns to numeric (handles coercion issues).
- Drops all rows with missing values after conversion.

### 2. Train/Test Split
- Uses `train_test_split()` (80% train / 20% test).
- Splits features (X) and target (y = FWI).

### 3. Correlation-Based Feature Elimination
- Drops features with correlation > 0.85 (domain-based threshold).
- Applied only to training data, then mirrored to test set.

### 4. Feature Scaling
- Applies `StandardScaler` to normalize feature values.
- Prevents scale issues in regularized regression.

---

## 🤖 Models Used

### ✅ Linear Regression
- Baseline model, simple and interpretable.

### ✅ LassoCV
- Automatically selects best `alpha` using 5-fold CV.

### ✅ ElasticNet
- Combines Lasso and Ridge penalties.
- Evaluated on test set.

### ✅ RidgeCV (with multiple CV strategies)
- KFold (default)
- ShuffleSplit
- RepeatedKFold
- (LeaveOneOut is included but commented out for performance)

---

## 📊 Evaluation Metrics
- **Mean Absolute Error (MAE)**
- **R² Score**
- **Scatter plots** for visual comparison of predicted vs. actual FWI

---

## 🧪 Additional Cross-Validation
- `cross_val_score` example for validating any model (e.g., Lasso)
- Shows how to extract R² scores across folds

---

## 🚀 Requirements
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
