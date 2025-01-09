import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib 
 

# Upload data 
data = pd.read_csv("data/housing.csv")

# Data Preprocessing 
data["total_bedrooms"].fillna(data["total_bedrooms"].median(), inplace=True)
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model comparison
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
} 

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse
    print(f"{name} - MSE: {mse}") 

# Saving the best model
best_model_name = min(results, key=results.get)
best_model = models[best_model_name]
print(best_model) 
joblib.dump(best_model, "models/best_housing_price_model.pkl") 