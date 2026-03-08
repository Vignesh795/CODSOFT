import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv("Titanic-Dataset.csv")

titanic_df = titanic_df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

titanic_df["Sex"] = titanic_df["Sex"].replace({"male": 0, "female": 1})

titanic_df["Age"] = titanic_df["Age"].fillna(titanic_df["Age"].mean())
titanic_df["Fare"] = titanic_df["Fare"].fillna(titanic_df["Fare"].mean())

titanic_df = titanic_df.dropna()

X = titanic_df[["Pclass", "Sex", "Age", "Fare"]]
y = titanic_df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

predicted_values = lr_model.predict(X_test)

acc = accuracy_score(y_test, predicted_values)

print("Titanic Prediction Accuracy:", acc)
