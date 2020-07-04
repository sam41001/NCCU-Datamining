## Import Library ##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## For Data Encoding ##
from sklearn.preprocessing import LabelEncoder, StandardScaler

## For Model Evaluation ##
from sklearn.model_selection import KFold, train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
## Machine Learning Model ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

## For Model Performance ##
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


## Model Performance
def evaluation(gt, pred):
    acc = accuracy_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    f1 = f1_score(gt, pred)
    matrix = confusion_matrix(gt, pred)

    return acc, precision, recall, f1, matrix


raw_data = pd.read_csv("./EmployeeAttrition.csv", index_col=0)

categorical_col = []
numeric_col = []
for col in raw_data.columns:
    if raw_data[col].dtype == object and col != "Attrition":
        categorical_col.append(col)
        print(col, raw_data[col].unique())
        print("========================================================================")
    elif raw_data[col].dtype == int and col != "Attrition":
        numeric_col.append(col)

## Encode Label
raw_data["Attrition"] = raw_data["Attrition"].astype("category").cat.codes
raw_data["Attrition"].value_counts()

## Data Encoding (one-hot encoding)
one_hot_encoding_df = pd.get_dummies(raw_data, columns=categorical_col)

print(set(one_hot_encoding_df.columns) - set(numeric_col))

## Data Splitting and Model Learning (Decision Tree)
avg_acc = 0
avg_precision = 0
avg_recall = 0
avg_f1 = 0
avg_confusion_matrix = []
avg_feature_importance = []

kf = KFold(n_splits=5)
fold_count = 0

for train_index, test_index in kf.split(one_hot_encoding_df):
    print("Training Data: %d, Testing Data: %d" % (len(train_index), len(test_index)))
    train_X = one_hot_encoding_df.iloc[train_index, one_hot_encoding_df.columns != 'Attrition']
    train_y = one_hot_encoding_df.iloc[train_index]["Attrition"]
    test_X = one_hot_encoding_df.iloc[test_index, one_hot_encoding_df.columns != 'Attrition']
    test_y = one_hot_encoding_df.iloc[test_index]["Attrition"]
    #只使用Randomforest 進行參數調整
   # model = RandomForestClassifier(n_estimators=21, random_state=10, max_depth=6)
    model = RandomForestClassifier()
    model = model.fit(train_X, train_y)

    test_predict = model.predict(test_X)
    avg_feature_importance.append(model.feature_importances_)
    acc, precision, recall, f1, matrix = evaluation(test_y, test_predict)
    # acc, precision, recall, f1, matrix = evaluation(test_y, test_predict)

    print("Fold: %d, Accuracy: %f, Precision: %f, Recall: %f, F1: %f" % (
    fold_count + 1, round(acc, 3), round(precision, 3), round(recall, 3), round(f1, 3)))
    avg_acc += acc
    avg_precision += precision
    avg_recall += recall
    avg_f1 += f1
    avg_confusion_matrix.append(matrix)
    fold_count += 1

print("=================================================================================")
print("Avg Accuracy: %f, Avg Precision: %f, Avg Recall: %f, Avg F1: %f" % (round(avg_acc / kf.get_n_splits(), 3), \
                                                                           round(avg_precision / kf.get_n_splits(), 3), \
                                                                           round(avg_recall / kf.get_n_splits(), 3), \
                                                                           round(avg_f1 / kf.get_n_splits(), 3)))

importance_dict = {}
for col, importance in zip(train_X.columns, np.mean(np.array(avg_feature_importance), axis=0)):
    importance_dict[col] = importance

print(sorted(importance_dict.items(), key=lambda x: -x[1])[:10])
