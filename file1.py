# ===============================
# CAR INSURANCE CLAIM PREDICTION
# NUMERIC MODEL PIPELINE
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# -------------------------
# LOAD DATA
# -------------------------
file_path = "train.csv"   # <-- change path if needed
df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)

# -------------------------
# DROP ID COLUMN
# -------------------------
if "policy_id" in df.columns:
    df = df.drop(columns=["policy_id"])

# -------------------------
# TARGET
# -------------------------
TARGET = "is_claim"

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\nClass distribution:")
print(y.value_counts(normalize=True))

# -------------------------
# COLUMN TYPES
# -------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

print("\nNumeric columns:", len(num_cols))
print("Categorical columns:", len(cat_cols))

# -------------------------
# PREPROCESSING
# -------------------------
num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols)
])

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------
# BASELINE MODEL (Random Forest)
# -------------------------
rf_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
rf_pipeline.fit(X_train, y_train)

# Evaluation
rf_pred = rf_pipeline.predict(X_val)
rf_proba = rf_pipeline.predict_proba(X_val)[:, 1]

print("\n===== RANDOM FOREST RESULTS =====")
print(classification_report(y_val, rf_pred))
print("ROC-AUC:", round(roc_auc_score(y_val, rf_proba), 4))


# -------------------------
# FINAL MODEL (XGBoost)
# -------------------------
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_pipeline = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss"
    ))
])

print("\nTraining XGBoost...")
xgb_pipeline.fit(X_train, y_train)

# Evaluation
xgb_pred = xgb_pipeline.predict(X_val)
xgb_proba = xgb_pipeline.predict_proba(X_val)[:, 1]

print("\n===== XGBOOST RESULTS =====")
print(classification_report(y_val, xgb_pred))
print("ROC-AUC:", round(roc_auc_score(y_val, xgb_proba), 4))


# -------------------------
# SAVE FINAL MODEL
# -------------------------
joblib.dump(xgb_pipeline, "car_claim_numeric_model.pkl")
print("\nModel saved as car_claim_numeric_model.pkl")