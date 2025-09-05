# Week 2 - Real-Time Air Quality Prediction System (Model Training & Evaluation)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Example: Simulated dataset for one city (Delhi)
# In actual project, replace with OpenAQ or prepared dataset
np.random.seed(42)
date_rng = pd.date_range(end=pd.Timestamp.now(), periods=200, freq="H")
df = pd.DataFrame({
    "datetime": date_rng,
    "PM2.5": np.random.normal(60, 15, 200),
    "PM10": np.random.normal(100, 25, 200),
    "O3": np.random.normal(30, 10, 200),
    "CO": np.random.normal(0.5, 0.1, 200),
    "SO2": np.random.normal(20, 5, 200),
    "NO2": np.random.normal(40, 10, 200),
})

# Shift data by 1 hour for prediction
results = []
for pollutant in ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]:
    df[f"{pollutant}_next"] = df[pollutant].shift(-1)

    data = df[[pollutant, f"{pollutant}_next"]].dropna()
    X = data[[pollutant]].values
    y = data[f"{pollutant}_next"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([pollutant, round(mae, 3), round(r2, 3)])

# Show evaluation results
eval_df = pd.DataFrame(results, columns=["Pollutant", "MAE", "R2_Score"])
print("Model Evaluation Results:")
print(eval_df)
