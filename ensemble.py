import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv("data/train.csv")
y = df["log_hours"].to_numpy()

models = {
    "xgboost": np.load("xgb_oof.npy"),
    "ridge": np.load("ridge_oof.npy"),
    "extra_trees": np.load("extra_oof.npy")
}

predictions = np.column_stack(list(models.values()))

def objective(weights):

    ensemble = predictions @ weights

    return root_mean_squared_error(y, ensemble)


n = len(models)

result = minimize(
    objective,
    x0=np.ones(n) / n,
    method="SLSQP",
    bounds=[(0, 1)] * n,
    constraints={
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1
    }
)

if not result.success:
    raise RuntimeError(result.message)

for name, preds in models.items():
    print(f"{name} error: {root_mean_squared_error(y, preds)}")

for name, weight in zip(models.keys(), result.x):
    print(f"{name} weight: {weight:.4f}")

print(f"Best ensemble score: {result.fun}")


