import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset and skip first row if it’s a junk header
df = pd.read_csv('Algerian_forest_fires_dataset_UPDATE.csv', skiprows=1)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop non-numeric date columns
df.drop(['day', 'month', 'year'], axis=1, inplace=True)

# Encode 'Classes': 'not fire' -> 1, 'fire' -> 0
df['Classes'] = np.where(df['Classes'] == 'not fire', 1, 0)

# Convert all feature columns to numeric (coerce invalid entries to NaN)
X = df.drop('FWI', axis=1)
X = X.apply(pd.to_numeric, errors='coerce')  # handles 'Temperature' or other column issues

# Target variable
y = pd.to_numeric(df['FWI'], errors='coerce')  # ensure target is also numeric

# Drop rows with any NaNs caused by coercion
combined = pd.concat([X, y], axis=1).dropna()
X = combined.drop('FWI', axis=1)
y = combined['FWI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# print("Train/Test shapes:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# # Show correlation matrix
# print("Correlation matrix:\n", X_train.corr())

# Plot correlation heatmap
# plt.figure(figsize=(12, 10))
# sns.heatmap(X_train.corr(), annot=True, fmt=".2f",  square=True)
# plt.title("Correlation Matrix - Training Set")
# plt.tight_layout()
# plt.show()

def correlation(datset,threshold):
    col_corr=set()
    corr_matrix=datset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
#threshold domain expert's work
corr_features = correlation(X_train, 0.85)


#drop this features where correlation is greater than threshold
X_train.drop(corr_features, axis=1,  inplace=True)
X_test.drop(corr_features, axis=1, inplace=True)
# print(X_train.shape, X_test.shape)

# Standardize the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   
# print("Standardized Train/Test ", X_train, X_test)
# #boxplot
# plt.subplots(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# sns.boxplot(data=X_train)
# plt.title("Before scaling")
# plt.subplot(1, 2, 2)
# sns.boxplot(data=X_train_scaled)
# plt.title("After scaling")
# plt.tight_layout()
# # plt.show()

#MODEL TRAINING
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
linreg= LinearRegression()
linreg.fit(X_train_scaled, y_train)
# Predictions
y_pred = linreg.predict(X_test_scaled)
mae= mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# print(f"MAE: {mae:.2f}, R^2: {r2:.2f}")
# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual FWI")
# plt.ylabel("Predicted FWI")
# plt.title("Actual vs Predicted FWI")
# plt.show()

#LASSO REGRESSION
# from sklearn.linear_model import Lasso
# from sklearn.metrics import mean_absolute_error, r2_score
# lasso = Lasso()
# lasso.fit(X_train_scaled, y_train)
# # Predictions
# y_pred_lasso = lasso.predict(X_test_scaled)
# mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
# r2_lasso = r2_score(y_test, y_pred_lasso)
# print(f"Lasso MAE: {mae_lasso:.2f}, R^2: {r2_lasso:.2f}")
# plt.scatter(y_test, y_pred_lasso)
# plt.xlabel("Actual FWI")
# plt.ylabel("Predicted FWI (Lasso)")
# plt.title("Actual vs Predicted FWI (Lasso)")
# plt.show()
#Cross-validation for Lasso
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train_scaled, y_train)
# Predictions
lasso_cv_pred = lasso_cv.predict(X_test_scaled)
print(lasso_cv.alpha_ )

y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)
plt.scatter(y_test, y_pred_lasso_cv)
plt.xlabel("Actual FWI")        
plt.ylabel("Predicted FWI (LassoCV)")
plt.title("Actual vs Predicted FWI (LassoCV)")
plt.show()







# #RIDGE REGRESSION
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_absolute_error, r2_score
# ridge = Ridge()
# ridge .fit(X_train_scaled, y_train)
# # Predictions
# y_pred_ridge  = ridge .predict(X_test_scaled)
# mae_ridge  = mean_absolute_error(y_test, y_pred_ridge )
# r2_ridge  = r2_score(y_test, y_pred_ridge )
# print(f"ridge  MAE: {mae_ridge :.2f}, R^2: {r2_ridge :.2f}")
# plt.scatter(y_test, y_pred_ridge )
# plt.xlabel("Actual FWI")
# plt.ylabel("Predicted FWI (ridge )")
# plt.title("Actual vs Predicted FWI (ridge )")
# plt.show()

#ElasticNet regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
elastic_net = ElasticNet()
elastic_net.fit(X_train_scaled, y_train)
# Predictions
y_pred_en = elastic_net.predict(X_test_scaled)
mae_en = mean_absolute_error(y_test, y_pred_en)
r2_en = r2_score(y_test, y_pred_en)
print(f"ElasticNet MAE: {mae_en:.2f}, R^2: {r2_en:.2f}")
plt.scatter(y_test, y_pred_en)
plt.xlabel("Actual FWI")
plt.ylabel("Predicted FWI (ElasticNet)")
plt.title("Actual vs Predicted FWI (ElasticNet)")
plt.show()
