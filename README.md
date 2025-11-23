# task-4

>> imported BMW dataset using pandas data frame.
>> converted the 'Sales_volume' target column from categorical to numerical.
>> Encoded the target column to drop dummies.

>> Splitted the datasets into training and test sets with 80:20 ratio respectively using sklearn model_selection train_test_split.
>> Fitted Logistic regression model for numeric classification

>>Evaluated the model with accuracy, confusion matrix and created the classification report of 'Precision', 'f-1 Score', 'support'
  and achieved following metric reults:
  Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00      5013
           1       1.00      1.00      1.00      4987

    accuracy                           1.00     10000
   macro avg       1.00      1.00      1.00     10000
weighted avg       1.00      1.00      1.00     10000

>> ROC-AOC score: 0.9999959599726895
