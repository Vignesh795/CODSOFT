import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

movie_df = pd.read_csv("Movies task 2.csv", encoding="latin1")

movie_df = movie_df[["Year", "Duration", "Rating"]]

movie_df = movie_df.dropna()

movie_df["Year"] = movie_df["Year"].str.replace("(", "")
movie_df["Year"] = movie_df["Year"].str.replace(")", "")

movie_df["Duration"] = movie_df["Duration"].str.replace(" min", "")

movie_df["Year"] = movie_df["Year"].astype(int)
movie_df["Duration"] = movie_df["Duration"].astype(int)

X = movie_df[["Year", "Duration"]]
y = movie_df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg_model = LinearRegression()

reg_model.fit(X_train, y_train)

rating_pred = reg_model.predict(X_test)

error = mean_squared_error(y_test, rating_pred)

print("Movie Rating Model Error:", error)
