import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

best_params={'max_depth': 18,
             'min_samples_leaf': 1,
             'min_samples_split': 9,
             'max_features': 0.7946005321469658,
             'bootstrap': False}


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

#preproccess
x = x.select_dtypes(include="number")
x = x.replace([np.inf, -np.inf], np.nan)
x = np.sign(x) * np.log1p(np.abs(x))
x = x.fillna(0)

y = df["log_hours"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=30
)

buckets = df["hours_bucket"]

extra_oof_preds=np.zeros(len(x))



study_scores=[]


for train_col_ids, val_col_ids in cv.split(x,buckets):

    x_train = x.iloc[train_col_ids]
    x_val = x.iloc[val_col_ids]
    y_train= y.iloc[train_col_ids]
    y_val = y.iloc[val_col_ids]

    model = ExtraTreesRegressor(
        n_estimators=250,
        **best_params
    )

    model.fit(x_train,y_train)
    preds = model.predict(x_val)

    extra_oof_preds[val_col_ids]=preds

    error = root_mean_squared_error(y_val, preds)

    study_scores.append(error)



print(f"best score: {np.mean(study_scores)}")

np.save("extra_oof.npy", extra_oof_preds)