import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

iris_df = pd.read_csv("IRIS task 3.csv")

features = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
target = iris_df["species"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

pred = rf_model.predict(X_test)

result = accuracy_score(y_test, pred)

print("Iris Classification Accuracy:", result)
