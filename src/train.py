import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print(" Loading data...")

df = pd.read_csv("Data/india_housing_prices.csv")

df = df.drop_duplicates()

if "ID" in df.columns:
    df = df.drop("ID", axis=1)

# -------------------------------
#  DEFINE COLUMNS
# -------------------------------
numeric_cols = [
    "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Year_Built",
    "Floor_No", "Total_Floors",
    "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space"
]

categorical_cols = [
    "State", "City", "Property_Type",
    "Furnished_Status", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status"
]

# -------------------------------
#  FIX TYPES
# -------------------------------
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill numeric
df[numeric_cols] = df[numeric_cols].fillna(0)

# Fill categorical
for col in categorical_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].replace("nan", "Unknown")
    df[col] = df[col].fillna("Unknown")

# -------------------------------
#  FEATURE ENGINEERING
# -------------------------------
df["Age_of_Property"] = 2026 - df["Year_Built"]
df["Price_per_SqFt"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

# -------------------------------
#  TARGETS
# -------------------------------
median_price = df["Price_in_Lakhs"].median()
df["Good_Investment"] = (df["Price_in_Lakhs"] <= median_price).astype(int)
df["Future_Price"] = df["Price_in_Lakhs"] * (1.08 ** 5)

# -------------------------------
#  FEATURES
# -------------------------------
X = df.drop(["Good_Investment", "Future_Price"], axis=1)
y_class = df["Good_Investment"]
y_reg = df["Future_Price"]

# Save schema
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# -------------------------------
#  PIPELINE
# -------------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=20, max_depth=10, n_jobs=-1))
])

reg = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=20, max_depth=10, n_jobs=-1))
])

# -------------------------------
#  TRAIN
# -------------------------------
X_train, X_test, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.2, random_state=42)
_, _, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

print(" Training...")
clf.fit(X_train, y_train_c)
reg.fit(X_train, y_train_r)

joblib.dump(clf, "models/classifier_pipeline.pkl")
joblib.dump(reg, "models/regressor_pipeline.pkl")

print(" DONE")
