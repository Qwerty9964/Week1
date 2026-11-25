import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


df = pd.read_csv("data/train.csv")

df["commit_cadence"]=df["commit_cadence"].map({
    "sporadic": 1,
    "daily": 2,
    "frequent": 3
}).fillna(0)

df["incremental_lines_added"] = (
    df["lines_added_total"] - df["max_lines_in_one_commit"].fillna(0)
).clip(lower=0)

df["dev_commitment"] = (
    df["commit_cadence"]*df["active_commit_days"]*df["num_commits"]*df["median_lines_per_commit"]
).clip(lower=0)

df["non_first_commit_code"] = (
    df["lines_of_code"]*(1-df["pct_code_in_first_commit"])
)


x = df.drop(columns=["project_id",
"pct_commits_weekend",
"pct_commits_late_night",
"num_commit_authors",
"num_branches",
"num_merge_commits",
"primary_language",
"pct_primary_language",
"num_dependencies",
"framework",
"uses_game_engine",
"has_tests",
"num_test_files",
"has_ci_config",
"has_dockerfile",
"has_backend",
"has_database",
"comment_line_ratio",
"num_image_files",
"num_audio_files",
"num_3d_model_files",
"num_font_files",
"asset_size_mb",
"has_custom_art",
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
"project_category",
"is_multiplayer",
"mentions_ai_api",
"is_clone_or_tutorial",
"is_forked_from_template",
"author_prior_projects",
"author_account_age_years"
])

x=np.log1p(x.fillna(0))

y = df["log_hours"]


x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=30)

for weight in [4,4.5,5,5.5,6]:

    model = XGBRegressor(
        n_estimators=2000,
        max_depth=4,
        learning_rate=0.06,
        early_stopping_rounds=80,
        eval_metric="rmse",
        random_state=30
    )

    model.fit(x_train,y_train,eval_set=[(x_val,y_val)],verbose=50)

    print(f"With weight {weight} at iteration {model.best_iteration} the score was {model.best_score}")


