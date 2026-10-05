"""
Trains a Logistic Regression model on the Titanic dataset
and saves it as model.pkl for the Flask API to use.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- Load data ----
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(URL)

# ---- Features ----
FEATURES = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[FEATURES].copy()
y = df['Survived']

# ---- Preprocess ----
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# ---- Train ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---- Evaluate ----
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing  Accuracy: {test_acc  * 100:.2f}%")

# ---- Save ----
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model.pkl saved successfully.")