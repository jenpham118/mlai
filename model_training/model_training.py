import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

train_df = pd.read_csv("amd_train_scaled.csv", parse_dates=["date"])
test_df  = pd.read_csv("amd_test_scaled.csv", parse_dates=["date"])

feature_cols = [c for c in train_df.columns if c not in ("date", "target")]

X_train, y_train = train_df[feature_cols], train_df["target"]
X_test, y_test   = test_df[feature_cols], test_df["target"]

tscv = TimeSeriesSplit(n_splits=5)

# Linear Regression

lr = LinearRegression()
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
print(f"Linear Regression Test MSE: {lr_mse:.4f}")

# Random Forest Regressor
# hyperparameter for grid
rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid=rf_param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_
print("Random Forest Best params:", rf_grid.best_params_)

rf_pred = rf_best.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
print(f"Random Forest Test MSE: {rf_mse:.4f}")
