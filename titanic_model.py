import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
X = df[features].copy()
y = df['Survived']

X['Age'] = X['Age'].fillna(X['Age'].median())
X['Sex'] = X['Sex'].map({'female': 0, 'male': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

intercept = model.intercept_[0]
coefficients = model.coef_[0]

print(f"Intercept: {intercept:.4f}")
print(f"Pclass Weight: {coefficients[0]:.4f}")
print(f"Sex(Male) Weight: {coefficients[1]:.4f}")
print(f"Age Weight: {coefficients[2]:.4f}")
print(f"SibSp Weight: {coefficients[3]:.4f}")
print(f"Parch Weight: {coefficients[4]:.4f}")

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Testing Accuracy: {test_acc * 100:.2f}%")
print(classification_report(y_test, y_test_pred))