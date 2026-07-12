import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score

# Task 1: Load Cleaned Data
print("--- Task 1: Loading Data & Target Definition ---")
df = pd.read_csv('cleaned_car_data.csv')
print("Data loaded successfully.")

# Feature matrix x, drop targets
X = df.drop(columns=['Price'])
print("Feature matrix created successfully.")

# Regression Target
y_reg = df['Price']
print("Regression target created successfully.")

# Classification Target (Binarized at median)
median_price = y_reg.median()
y_clf = (y_reg > median_price).astype(int)
print("Classification target created successfully.")

print(f"Dataset Shape: {X.shape}")
print(f"Regression Median Price: {median_price:.2f}")
print(f"Class 0 (<= Median): {X[y_clf == 0].shape[0]} rows, Class 1 (> Median): {X[y_clf == 1].shape[0]} rows\n")


#  Task 2: Encode categorical columns:
print("--- Task 2: Encoding Categorical Columns ---")

X_encoded = pd.get_dummies(X, columns=['Car_Brand', 'Fuel_Type', 'Transmission'], drop_first=True, dtype=int)
print(f"Shape after One-Hot Encoding: {X_encoded.shape}")
print("Columns after encoding:\n", X_encoded.columns.tolist(), "\n")

# Task 3:Leak-free train-test split and scaling:
print("--- Task 3: Train-Test Split & Scaling ---")
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X_encoded, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X_encoded, y_clf, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

print(f"Training features shape: {X_train_scaled.shape}")
print(f"Test features shape: {X_test_scaled.shape}\n")

# Task 4 :Regression model — Linear Regression:
print("--- Task 4: Regression Model - Linear Regression ---")
# OLS Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_lr = lr_model.predict(X_test_scaled)

mse_lr = mean_squared_error(y_reg_test, y_pred_reg_lr)
r2_lr = r2_score(y_reg_test, y_pred_reg_lr)

print(f"OLS Linear Regression - MSE: {mse_lr:.2f}, R2: {r2_lr:.4f}")

# Extract Coefficients
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr_model.coef_
})

coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

print("\nTop 3 Features with largest absolute coefficients:")
print(coef_df.head(3))

# Ridge Regression Experiment
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_reg_train)
y_pred_reg_ridge = ridge_model.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_reg_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_reg_ridge)
print(f"Ridge Regression - MSE: {mse_ridge:.2f}, R2: {r2_ridge:.4f}\n")

# Task 5: Classification model — Logistic Regression:
print("--- Task 5: Classification Model - Logistic Regression ---")
# Check Class Imbalance
print("Training Class Distribution:")
print(y_clf_train.value_counts(normalize=True))

# class ka distribution puri tarah se balanced hai kyonki humne median par bineriz kiya hai.
log_reg_baseline = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
log_reg_baseline.fit(X_train_scaled, y_clf_train)

y_pred_clf_base = log_reg_baseline.predict(X_test_scaled)
y_prob_clf_base = log_reg_baseline.predict_proba(X_test_scaled)[:, 1]

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_clf_test, y_pred_clf_base))

print("\n--- Classification Report ---")
print(classification_report(y_clf_test, y_pred_clf_base))

# ROC Curve & AUC
fpr_base, tpr_base, _ = roc_curve(y_clf_test, y_prob_clf_base)
auc_base = auc(fpr_base, tpr_base)

plt.figure(figsize=(7, 5))
plt.plot(fpr_base, tpr_base, color='darkorange', lw=2, label=f'Baseline Logistic Reg (AUC = {auc_base:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Baseline Model (C=1.0)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# Task 5 b: Decision-threshold sensitivity
print("--- Task 5b: Decision Threshold Sensitivity ---")
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
threshold_results = []
for t in thresholds:
    preds = (y_prob_clf_base >= t).astype(int)
    p = precision_score(y_clf_test, preds, zero_division=0)
    r = recall_score(y_clf_test, preds, zero_division=0)
    f1 = f1_score(y_clf_test, preds, zero_division=0)
    threshold_results.append([t, p, r, f1])
threshold_df = pd.DataFrame(threshold_results, columns=['Threshold', 'Precision', 'Recall', 'F1'])
print(threshold_df.to_string(index=False))

# Task 6 : Regularization experiment on Logistic Regression:
print("--- Task 6: Regularization Experiment on Logistic Regression ---")
log_reg_strong = LogisticRegression(C=0.01, max_iter=1000, random_state=42)
log_reg_strong.fit(X_train_scaled, y_clf_train)

y_pred_clf_strong = log_reg_strong.predict(X_test_scaled)
y_prob_clf_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]

fpr_strong, tpr_strong, _ = roc_curve(y_clf_test, y_prob_clf_strong)
auc_strong = auc(fpr_strong, tpr_strong)

print(f"Strong Regularization (C=0.01) -> AUC: {auc_strong:.4f}")

# Task 7: AUC antar ke liye bootstrap confidence interval
print("--- Task 7: Bootstrap Confidence Interval for AUC Difference ---")
np.random.seed(42)
n_bootstraps = 500
bootstrap_diffs = []

y_clf_test_arr = np.array(y_clf_test)

for i in range(n_bootstraps):
    # Sample indices with replacement
    indices = np.random.choice(len(y_clf_test_arr), size=len(y_clf_test_arr), replace=True)
    # target aur anumanit sambhavnaon ke liye bootstrap sample nikalen
    y_true_boot = y_clf_test_arr[indices]
    prob_base_boot = y_prob_clf_base[indices]
    prob_strong_boot = y_prob_clf_strong[indices]

#AUC ki ganana tabhi karen jab bootstrap sample mein donon class maujood hon.
if len(np.unique(y_true_boot)) == 2:
        fpr_b, tpr_b, _ = roc_curve(y_true_boot, prob_base_boot)
        auc_b = auc(fpr_b, tpr_b)

        fpr_s, tpr_s, _ = roc_curve(y_true_boot, prob_strong_boot)
        auc_s = auc(fpr_s, tpr_s)

        bootstrap_diffs.append(auc_b - auc_s)

mean_diff = np.mean(bootstrap_diffs)
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)

print(f"mean auc difference(c=1.0 - c=0.01): {mean_diff:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

if ci_lower > 0 or ci_upper < 0:
     print("Result: 95% CI mein zero nahi aa raha hai. Matlab C=1.0 ka advantage bilkul solid aur statistically reliable hai!")
else:
    print("Result: 95% CI ke beech mein zero aa gaya. Iska matlab dono ke beech ka difference har sample mein reliable nahi hoga.")