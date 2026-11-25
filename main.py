import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


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
"hours_spent"
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

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=30)

def objective(trial):

    study_scores=[]

    params={
        "max_depth" : trial.suggest_int("max_depth",1,9),
        "learning_rate" : trial.suggest_float("learning_rate",0.01,0.11),
        "subsample" : trial.suggest_float("subsample",0.3,1),
        "colsample_bytree" : trial.suggest_float("colsample_bytree",0.5,1),
        "reg_lambda" : trial.suggest_float("reg_lambda",0.001,100, log=True),
        "reg_alpha" : trial.suggest_float("reg_alpha",0.001,100, log=True),
        "gamma" : trial.suggest_float("gamma",0,5),
    }

    for train_col_ids, val_col_ids in cv.split(x):

        x_train = x.iloc[train_col_ids]
        x_val = x.iloc[val_col_ids]
        y_train= y.iloc[train_col_ids]
        y_val = y.iloc[val_col_ids]

        model = XGBRegressor(
            **params,
            n_estimators=2000,
            early_stopping_rounds=80,
            eval_metric="rmse",
            enable_categorical=True,
            tree_method="hist",
            random_state=1
        )

        model.fit(x_train,y_train,eval_set=[(x_val,y_val)],verbose=False)

        study_scores.append(model.best_score)

    return np.mean(study_scores)


study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=30))

study.optimize(objective, n_trials=60, show_progress_bar=True)

print(study.best_value)
print(study.best_params)