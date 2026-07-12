import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline

# Setup & Reload Data from Parts 1 & 2
print("--- Setup & Reload Data from Parts 1 & 2 ---")

df = pd.read_csv('cleaned_car_data.csv')

X = df.drop(columns=['Price'])
y_reg = df['Price']
y_clf = (y_reg > y_reg.median()).astype(int)

# categorical encoding
X_encoded = pd.get_dummies(X, columns=['Car_Brand', 'Fuel_Type', 'Transmission'], drop_first=True, dtype=int)

# Train-Test Split (Same as Part 2)
X_train, X_test, y_clf_train, y_clf_test = train_test_split(X_encoded, y_clf, test_size=0.2, random_state=42)

# standard scaled Data for direct model evaluation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Task 1: Decision Tree Baseline 
print("--- Task 1: Decision Tree Baseline ---")
dt_unconstrained = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_unconstrained.fit(X_train_scaled, y_clf_train)

dt_un_train_acc = accuracy_score(y_clf_train, dt_unconstrained.predict(X_train_scaled))
dt_un_test_acc = accuracy_score(y_clf_test, dt_unconstrained.predict(X_test_scaled))
print(f"Decision Tree (Unconstrained) - Train Accuracy: {dt_un_train_acc:.4f}, Test Accuracy: {dt_un_test_acc:.4f}")

# Task2. Controlled Tree
print("--- Task 2: Controlled Decision Tree ---")
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)

dt_con_train_acc = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
dt_con_test_acc = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))
print(f"Decision Tree (Controlled) - Train Accuracy: {dt_con_train_acc:.4f}, Test Accuracy: {dt_con_test_acc:.4f}\n")

# Task 3. Gini vs Entropy comparison:
print("--- Task 3: Gini vs Entropy Comparison ---")
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42).fit(X_train_scaled, y_clf_train)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42).fit(X_train_scaled, y_clf_train)

print(f"Gini DT Test Accuracy: {accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)):.4f}")
print(f"Entropy DT Test Accuracy: {accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)):.4f}")

# Task 4. Random Forest:
print("--- Task 4: Random Forest---")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_scaled, y_clf_train)

rf_train_acc = accuracy_score(y_clf_train, rf_model.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf_model.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf_model.predict_proba(X_test_scaled)[:, 1])

print(f"Random Forest -> Train Acc: {rf_train_acc:.4f} | Test Acc: {rf_test_acc:.4f} | ROC-AUC: {rf_test_auc:.4f}")

# Top 5 Features by Importance
print("Top 5 Features by Importance:")
importances = rf_model.feature_importances_
feature_names = X_train.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Features by Random Forest Importance:")
print(fi_df.head(5).to_string(index=False))

# Task 4a. Gradient Boosting.
print("--- Task 4a: Gradient Boosting ---")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train_scaled, y_clf_train)

gb_train_acc = accuracy_score(y_clf_train, gb_model.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb_model.predict(X_test_scaled))
gb_test_auc = roc_auc_score(y_clf_test, gb_model.predict_proba(X_test_scaled)[:, 1])

print(f"\nGradient Boosting -> Train Acc: {gb_train_acc:.4f} | Test Acc: {gb_test_acc:.4f} | ROC-AUC: {gb_test_auc:.4f}")

# Task 4b: Feature Ablation Study
print("--- Task 4b: Feature Ablation Study ---")
lowest_5_features = fi_df.tail(5)['Feature'].tolist()
print("5 Lowest Importance Features to be removed:", lowest_5_features)

# Drop lowest 5 features from train/test
X_train_reduced = X_train.drop(columns=lowest_5_features)
X_test_reduced = X_test.drop(columns=lowest_5_features)

# Scale reduced features
scaler_red = StandardScaler()
X_train_red_scaled = scaler_red.fit_transform(X_train_reduced)
X_test_red_scaled = scaler_red.transform(X_test_reduced)

# Train second Random Forest on reduced set
rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_red_scaled, y_clf_train)
rf_red_test_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_red_scaled)[:, 1])

print(f"Full Model Test ROC-AUC: {rf_test_auc:.4f}")
print(f"Reduced Model (Dropped 5 Features) Test ROC-AUC: {rf_red_test_auc:.4f}")

# Task5. Cross-validated comparison:
print("--- Task 5: Cross-Validated Comparison ---")

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Controlled DT': DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = (scores.mean(), scores.std())
    print(f"{name:20} -> Mean AUC: {scores.mean():.4f} | Std AUC: {scores.std():.4f}")

# Task 6. Hyperparameter tuning with GridSearchCV:
print("--- Task 6: Hyperparameter Tuning with GridSearchCV ---")

# Pipeline ke liye double underscore (__) syntax ka use karke parameter grid ko define karna.
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
#Maine bina scale kiye hue raw training data (X_train) ka use karke ek mazboot (robust) end-to-end pipeline banayi hai

pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler(),
    RandomForestClassifier(random_state=42)
)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(X_train, y_clf_train)

print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Best Cross-Validated AUC: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_

# Task 7. Manual learning curve.
print("--- Task 7: Manual Learning Curve ---")

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_data = []

for f in fractions:
    n_rows = int(f * len(X_train))
    X_train_sub = X_train.iloc[:n_rows]
    y_clf_train_sub = y_clf_train.iloc[:n_rows]

# Tuned pipeline ko data ke subset par fit karna.
best_pipeline.fit(X_train_sub, y_clf_train_sub)

# AUC value compute karna aur evaluation metrics ko track karna.
train_auc = roc_auc_score(y_clf_train_sub, best_pipeline.predict_proba(X_train_sub)[:, 1])
test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])

learning_curve_data.append([f, train_auc, test_auc])

lc_df = pd.DataFrame(learning_curve_data, columns=['Training Fraction', 'Training AUC', 'Test AUC'])
print(lc_df.to_string(index=False))

# Task 8. Serialize the best model:
print("---Task 8: Serialize the best model---")

# Save model to disk
joblib.dump(best_pipeline, 'best_model.pkl')
print("Model 'best_model.pkl' successfully saved to disk.")

# 5-line verification block: Reload and predict on mock rows
loaded_model = joblib.load('best_model.pkl')
mock_data = X_test.iloc[0:2].copy()  # Emulate 2 genuine production payload rows
mock_preds = loaded_model.predict(mock_data)
mock_probs = loaded_model.predict_proba(mock_data)[:, 1]
print("Loaded Pipeline Inference execution successful!")
print(f"Predictions for 2 custom validation rows: {mock_preds} | Class Probabilities: {mock_probs}")




