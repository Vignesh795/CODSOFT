# Titanic Survival Prediction 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Select useful columns
data = data[['Survived','Pclass','Sex','Age','Fare']]

# Convert Sex column to numbers
data['Sex'] = data['Sex'].map({'male':0,'female':1})

# Fill missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Remove any remaining NaN rows
data = data.dropna()

# Split data
X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy

print("Titanic Model Accuracy:", accuracy_score(y_test, predictions))
