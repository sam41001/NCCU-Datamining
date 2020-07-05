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
# counter Xsmote, ysmote
from collections import Counter

raw_data_train0 = pd.read_csv("./fold/fold_0_train.csv", index_col=0)
raw_data_test0 = pd.read_csv("./fold/fold_0_test.csv", index_col=0)
raw_data_train1 = pd.read_csv("./fold/fold_1_train.csv", index_col=0)
raw_data_test1 = pd.read_csv("./fold/fold_1_test.csv", index_col=0)
raw_data_train2 = pd.read_csv("./fold/fold_2_train.csv", index_col=0)
raw_data_test2 = pd.read_csv("./fold/fold_2_test.csv", index_col=0)
raw_data_train3 = pd.read_csv("./fold/fold_3_train.csv", index_col=0)
raw_data_test3 = pd.read_csv("./fold/fold_3_test.csv", index_col=0)
raw_data_train4 = pd.read_csv("./fold/fold_4_train.csv", index_col=0)
raw_data_test4 = pd.read_csv("./fold/fold_4_test.csv", index_col=0)

temp=[raw_data_train0,raw_data_train1,raw_data_train2,raw_data_train3,raw_data_train4]
test_set=[raw_data_test0,raw_data_test1,raw_data_test2,raw_data_test3,raw_data_test4]
for i in range(5):
    raw_data = temp[i]
    test_raw_data = test_set[i]
    categorical_col = []
    numeric_col = []
    test_categorical_col = []
    test_numeric_col = []
    #訓練資料集轉換
    for col in raw_data.columns:
        if raw_data[col].dtype == object and col != "Attrition":
            categorical_col.append(col)
           # print(col, raw_data[col].unique())
           # print("========================================================================")
        elif raw_data[col].dtype == int and col != "Attrition":
            numeric_col.append(col)


    ## Encode Label
    raw_data["Attrition"] = raw_data["Attrition"].astype("category").cat.codes
    raw_data["Attrition"].value_counts()
    print(raw_data["Attrition"].value_counts())

    ## Data Encoding (one-hot encoding)
    one_hot_encoding_df = pd.get_dummies(raw_data, columns=categorical_col)

    #print(set(one_hot_encoding_df.columns) - set(numeric_col))

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
        # print(test_index)
        train_X = one_hot_encoding_df.iloc[train_index, one_hot_encoding_df.columns != 'Attrition']
        train_y = one_hot_encoding_df.iloc[train_index]["Attrition"]

        test_X = one_hot_encoding_df.iloc[test_index, one_hot_encoding_df.columns != 'Attrition']
        test_y = one_hot_encoding_df.iloc[test_index]["Attrition"]

        ###SMOTE 增加會離職的訓練樣本數變成1:1
        smo = SMOTE(random_state=42)
        # X_smo, y_smo = BorderlineSMOTE(random_state=42, kind='borderline-2').fit_resample(train_X, train_y)
        X_smo, y_smo = smo.fit_sample(train_X, train_y)
        print("Before SMOTE: ", Counter(train_y))
        print("After SMOTE:  ", Counter(y_smo))
        # 訓練模型最終答案
        model = RandomForestClassifier(n_estimators=21, random_state=10, max_depth=6, max_features=10)
        model = model.fit(X_smo, y_smo)
        X_smo.iloc[train_index], y_smo.iloc[train_index]

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
                                                                               round(avg_precision / kf.get_n_splits(),
                                                                                     3), \
                                                                               round(avg_recall / kf.get_n_splits(), 3), \
                                                                               round(avg_f1 / kf.get_n_splits(), 3)))

    importance_dict = {}

    ##測試資料集轉換

    for col in test_raw_data.columns:
        if test_raw_data[col].dtype == object and col != "Attrition":
            test_categorical_col.append(col)

        elif test_raw_data[col].dtype == int and col != "Attrition":
            test_numeric_col.append(col)

    test_raw_data["Attrition"] = test_raw_data["Attrition"].astype("category").cat.codes
    test_raw_data["Attrition"].value_counts()
    print( test_raw_data["Attrition"].value_counts())
    #fit testing data
    test_one_hot_encoding_df = pd.get_dummies(test_raw_data, columns=test_categorical_col)
    #print(set(test_one_hot_encoding_df.columns) - set(test_categorical_col))
    fit_testing_index=test_raw_data["Age"].values
    #print(fit_testing_index)

    #for p2_testing_index in  kf.split(test_one_hot_encoding_df):

    test_X = test_one_hot_encoding_df.iloc[fit_testing_index, test_one_hot_encoding_df.columns != 'Attrition']
    test_y = test_one_hot_encoding_df.iloc[fit_testing_index]["Attrition"]
    #print(test_X)
    test_predict = model.predict(test_X)

    avg_feature_importance.append(model.feature_importances_)
    acc, precision, recall, f1, matrix = evaluation(test_y, test_predict)
    fold_count=999
    print("Final Fold: %d, Final Accuracy: %f, Fianl Precision: %f, Final Recall: %f, Final F1: %f" % (
        1, round(acc, 3), round(precision, 3), round(recall, 3), round(f1, 3)))
    avg_acc = acc
    avg_precision = precision
    avg_recall = recall
    avg_f1 = f1
    avg_confusion_matrix.append(matrix)
    #fold_count = 999

    print("==================================final===============================================")


    for col, importance in zip(train_X.columns, np.mean(np.array(avg_feature_importance), axis=0)):
        importance_dict[col] = importance

    print(sorted(importance_dict.items(), key=lambda x: -x[1])[:10])
