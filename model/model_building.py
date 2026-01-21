# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("Data/train.csv")

# Select relevant features
features = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars",
    "BedroomAbvGr", "FullBath", "YearBuilt", "Neighborhood"
]
target = "SalePrice"

X = df[features]
y = df[target]

# Define numerical and categorical features
numerical_features = [
    "OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars",
    "BedroomAbvGr", "FullBath", "YearBuilt"
]
categorical_features = ["Neighborhood"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# Full pipeline: preprocessing + model
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_estimators=100))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.2f}")

# Save pipeline (includes preprocessing + model)
joblib.dump(pipeline, "model/house_price_model.pkl")
print("Pipeline saved as 'house_price_model.pkl'")
