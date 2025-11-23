import pandas as pd

df = pd.read_csv('task-4/BMW sales data (2010-2024) (1).csv')
print(df.head())


import numpy as np

median_sales = df["Sales_Volume"].median()
df["Sales_Class"] = np.where(df["Sales_Volume"] > median_sales, 1, 0)


df_encoded = pd.get_dummies(df.drop("Sales_Class", axis=1), drop_first=True)

X = df_encoded                         
y = df["Sales_Class"]                  

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#model fitting
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_train)

#model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = log_model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score

y_prob = log_model.predict_proba(x_test)[:, 1]
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))









