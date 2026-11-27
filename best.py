import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

scores=[]

best_params={'max_depth': 9,
            'learning_rate': 0.01496992977418088,
            'subsample': 0.4551244532473209,
            'colsample_bytree': 0.9000338599376193,
            'reg_lambda': 8.86045464059279,
            'reg_alpha': 0.0009822935926439825,
            'gamma': 2.2726785097950764}
    


df = pd.read_csv("data/train.csv")

df["commit_cadence"]=df["commit_cadence"].map({
    "sporadic": 1,
    "daily": 2,
    "frequent": 3
}).fillna(0)

#engineered features
df["incremental_lines_added"] = (
    df["lines_added_total"] - df["max_lines_in_one_commit"].fillna(0)
).clip(lower=0)

df["dev_commitment"] = (
    df["commit_cadence"]*df["active_commit_days"]*df["num_commits"]*df["median_lines_per_commit"]
).clip(lower=0)

df["non_active_days"] = (
    df["project_span_days"] - df["active_commit_days"]
)

df["commits_a_day"] = (
    df["num_commits"]/df["active_commit_days"].clip(lower=1)
)

df["non_first_commit_code"] = (
    df["lines_of_code"]*(1-df["pct_code_in_first_commit"])
)

df["incremental_intesity"] = (
    df["incremental_lines_added"]*df["active_commit_days"]
)

df["commit_imbalance"] = (
  df["max_lines_in_one_commit"] / df["median_lines_per_commit"].clip(lower=1)
)

x = df.drop(columns=["project_id",
"pct_commits_weekend",
"pct_commits_late_night",
"num_commit_authors",
"num_branches",
"num_merge_commits",
"pct_primary_language",
"num_dependencies",
"uses_game_engine",
"has_tests",
"num_test_files",
"has_ci_config",
"has_dockerfile",
"has_database",
"comment_line_ratio",
"num_image_files",
"num_audio_files",
"num_font_files",
"readme_has_install_instructions",
"has_license",
"has_docs_folder",
"url_is_live",
"hosting_type",
"page_weight_kb",
"num_internal_pages",
"is_mobile_responsive",
"has_custom_domain",
"load_time_ms",
"is_multiplayer",
"mentions_ai_api",
"is_clone_or_tutorial",
"is_forked_from_template",
"author_prior_projects",
"author_account_age_years",
"log_hours",
"hours_bucket",
"hours_spent",

])

#preproccess anything thats not a number into a number or category to be handled by xgboost
for col in x.select_dtypes(include=["bool","boolean"]).columns:
    x[col] = x[col].astype(float)
for col in x.select_dtypes(include=["string","object","category"]).columns:
    x[col] = x[col].astype("string").astype("category")

y = df["log_hours"]

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=30
)

xgb_oof_preds = np.zeros(len(x))

for train_col_ids, val_col_ids in cv.split(x):

    x_train = x.iloc[train_col_ids]
    x_val = x.iloc[val_col_ids]
    y_train= y.iloc[train_col_ids]
    y_val = y.iloc[val_col_ids]

    model = XGBRegressor(
        **best_params,
        n_estimators=2000,
        early_stopping_rounds=80,
        eval_metric="rmse",
        enable_categorical=True,
        tree_method="hist",
        random_state=1
    )

    model.fit(x_train,y_train,eval_set=[(x_val,y_val)],verbose=False)

    xgb_oof_preds[val_col_ids]=model.predict(x_val)



    scores.append(model.best_score)

    importances= pd.Series(
        model.feature_importances_,
        index=x.columns
    )

    print(importances.sort_values(ascending=False))

print(np.mean(scores))

np.save("xgb_oof.npy", xgb_oof_preds)

analysis = df.copy()
analysis["predictions"] = xgb_oof_preds
analysis["error"] = analysis["log_hours"] - analysis["predictions"]
analysis["absolute_error"] = np.abs(analysis["error"])
analysis["squared_error"] = analysis["error"] * analysis["error"]

print(analysis.groupby("project_category")["absolute_error"].mean().sort_values(ascending=False))
print(analysis.groupby("hours_bucket")["absolute_error"].mean().sort_values(ascending=False))

analysis["predicted_hours"] = np.expm1(
    analysis["predictions"]
)

analysis["predicted_bucket"] = pd.cut(
    analysis["predicted_hours"],
    bins=[-np.inf, 10, 30, 100, np.inf],
    labels=["<10", "10-30", "30-100", "100+"]
)

print(
    analysis.groupby("predicted_bucket", observed=True)
    .agg(
        count=("error", "size"),
        bias=("error", "mean"),
        mse=("squared_error", "mean")
    )
)

print(scores)
print(np.mean(scores))


