# Movie Rating Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("Movies task 2.csv", encoding="latin1")

# Select important columns
data = data[['Year','Duration','Rating']]

# Remove missing values
data = data.dropna()

# Convert text values
data['Year'] = data['Year'].str.replace("(","").str.replace(")","")
data['Duration'] = data['Duration'].str.replace(" min","")

data['Year'] = data['Year'].astype(int)
data['Duration'] = data['Duration'].astype(int)

# Split data
X = data[['Year','Duration']]
y = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Error
print("Movie Rating Model Error:", mean_squared_error(y_test, pred))