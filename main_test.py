import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np 
import joblib
import pickle
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("stock_data.db")

# Load historical stock prices
query_prices = "SELECT * FROM historical_prices"
df_prices = pd.read_sql(query_prices, conn)

# Load news articles with sentiment scores
query_news = "SELECT Date, Sentiment FROM news_articles"
df_news = pd.read_sql(query_news, conn)

# Close connection
conn.close()

# Convert Date column to datetime format
df_prices["Date"] = pd.to_datetime(df_prices["Date"])
df_news["Date"] = pd.to_datetime(df_news["Date"])

# Display first few rows
#print(df_prices.head())
#print(df_news.head())

# Aggregate sentiment score per day (average sentiment for multiple articles)
df_news_grouped = df_news.groupby("Date").mean().reset_index()

# Merge stock prices with news sentiment
df_merged = pd.merge(df_prices, df_news_grouped, on="Date", how="left")

# Fill missing sentiment values (some days may not have news)
df_merged["Sentiment"]=df_merged["Sentiment"].fillna(0)

# Display merged dataset
#print(df_merged.tail())

# Sort by date
df_merged = df_merged.sort_values(by="Date")

# Create previous day's close price
df_merged["Previous_Close"] = df_merged["Close"].shift(1)

# Create rolling averages
df_merged["Moving_Avg_3d"] = df_merged["Close"].rolling(window=3).mean()
df_merged["Moving_Avg_7d"] = df_merged["Close"].rolling(window=7).mean()

# Define target variable (next day's close price)
df_merged["Next_Day_Close"] = df_merged["Close"].shift(-1)

# Drop rows with NaN values (caused by shifting/rolling operations)
df_final = df_merged.dropna()

# Display final dataset
#print(df_final.columns)


# Selecting Features (X) and Target (y)
X = df_merged[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment','Previous_Close', 'Moving_Avg_3d', 'Moving_Avg_7d']]
y = df_merged['Next_Day_Close']

X = X.fillna(X.median())
y=y.fillna(y.median())

# Normalizing Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

#print(f"Training Data Shape: {X_train.shape}, Test Data Shape: {X_test.shape}")



# Function to Train & Evaluate a Model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, model

# Models Dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Support Vector Regressor": SVR(kernel='rbf'),
    "XGBoost Regressor": XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Store results
results = {}

# Train & Evaluate each model
for name, model in models.items():
    mae, rmse, r2, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2 Score": r2}
    print(f"{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2 Score={r2:.4f}")

# Convert results to DataFrame for comparison
import pandas as pd
results_df = pd.DataFrame(results).T
#print("\nModel Performance Comparison:")
#print(results_df)

# Get predictions from each model
y_pred_lr = models["Linear Regression"].predict(X_test)
y_pred_rf = models["Random Forest"].predict(X_test)
y_pred_xgb = models["XGBoost Regressor"].predict(X_test)

# Averaging ensemble
y_pred_ensemble = (y_pred_lr + y_pred_rf + y_pred_xgb) / 3

# Evaluate the ensemble model
mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
rmse_ensemble = mean_squared_error(y_test, y_pred_ensemble, squared=False)
r2_ensemble = r2_score(y_test, y_pred_ensemble)

# Print results
#print(f"Ensemble Model: MAE={mae_ensemble:.4f}, RMSE={rmse_ensemble:.4f}, R2 Score={r2_ensemble:.4f}")


# Predictions on train set
y_train_pred_ensemble = (models["Linear Regression"].predict(X_train) + 
                         models["Random Forest"].predict(X_train) + 
                         models["XGBoost Regressor"].predict(X_train)) / 3

# Evaluate on training set
mae_train = mean_absolute_error(y_train, y_train_pred_ensemble)
rmse_train = mean_squared_error(y_train, y_train_pred_ensemble, squared=False)
r2_train = r2_score(y_train, y_train_pred_ensemble)

# Print results
#print(f"Ensemble Model (Train): MAE={mae_train:.4f}, RMSE={rmse_train:.4f}, R2 Score={r2_train:.4f}")
#print(f"Ensemble Model (Test): MAE={mae_ensemble:.4f}, RMSE={rmse_ensemble:.4f}, R2 Score={r2_ensemble:.4f}")

ensemble_model = StackingRegressor(
    estimators=[
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42))
    ],
    final_estimator=XGBRegressor(n_estimators=50, learning_rate=0.05, random_state=42)
)

ensemble_model.fit(X_train, y_train)

y_pred_test_ensemble = ensemble_model.predict(X_test)

# Evaluation
ensemble_mae = mean_absolute_error(y_test, y_pred_test_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))
ensemble_r2 = r2_score(y_test, y_pred_test_ensemble)

print(f"Ensemble Model: MAE={ensemble_mae:.4f}, RMSE={ensemble_rmse:.4f}, R2 Score={ensemble_r2:.4f}")

# Save the model
with open("final_ensemble_model.pkl", "wb") as model_file:
    pickle.dump(ensemble_model, model_file)

# Load the model
with open("final_ensemble_model.pkl", "rb") as model_file:
    ensemble_model = pickle.load(model_file)
# Predict
y_pred_test_ensemble = ensemble_model.predict(X_test)

# Evaluate (optional, for debugging)
ensemble_mae = mean_absolute_error(y_test, y_pred_test_ensemble)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))
ensemble_r2 = r2_score(y_test, y_pred_test_ensemble)

print(f"Final Ensemble Model: MAE={ensemble_mae:.4f}, RMSE={ensemble_rmse:.4f}, R2 Score={ensemble_r2:.4f}")
