import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv("data/train.csv")

x = df[["num_commits", "active_commit_days", "project_span_days","lines_of_code","forks","max_dir_depth","pct_code_in_first_commit","pct_generic_commit_msgs","churn_ratio","readme_word_count","avg_commit_msg_length"]]
x=np.log1p(x.fillna(0))


y = df["log_hours"]

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2,random_state=30)

model = LinearRegression()
model.fit(x_train,y_train)

predictions = model.predict(x_val)

error = root_mean_squared_error(y_val,predictions)

print(f"error: {error}")

correlations = df.select_dtypes(include = "number").corr()["log_hours"]

print(correlations.abs().sort_values())