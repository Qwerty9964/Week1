import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

df=pd.read_csv("data/train.csv")
y = df["log_hours"]

xgb_preds = np.load("xgb_oof.npy")
ridge_preds = np.load("ridge_oof.npy")

best_score = float("inf")
best_weight = 0

for weight in np.linspace(0,1,101):
    preds=(weight*xgb_preds)+((1-weight)*ridge_preds)

    score = root_mean_squared_error(y, preds)

    if score<best_score:
        best_score=score
        best_weight=weight

print(f"best score: {best_score}")
print(f"best xgb weight: {best_weight}")


