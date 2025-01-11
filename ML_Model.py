import random
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV, KFold
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import warnings
import time

warnings.filterwarnings("ignore")

seed = 42
random.seed(seed)
np.random.seed(seed)

# read data
data = pd.read_csv("data_path", encoding='UTF-8')
x, y = data.iloc[:, 4:-1], data.iloc[:, -1]    # Class
# x, y = data.iloc[:, 5:-1], data.iloc[:, -1]  # Method

# Min-Max
# scaler = MinMaxScaler()
# x = scaler.fit_transform(x)
# x = pd.DataFrame(x)

# data balance
# sm = SMOTE(random_state=42)
# x, y = sm.fit_resample(x, y)

# PCA-FST
# pca = PCA(n_components=35, random_state=42)    # The threshold for the method is 25, for the class is 35
# x_pca = pca.fit_transform(x)
# x_pca = pd.DataFrame(x_pca)
# x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

# Chi-square-FST
k_best = SelectKBest(score_func=chi2, k=35)
x_chi = k_best.fit_transform(x, y)
x_chi = pd.DataFrame(x_chi)
print(x_chi.head())

cv = KFold(n_splits=5, random_state=42, shuffle=True)

# RF
# ml = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=42)

# DT
# ml = DecisionTreeClassifier(max_depth=5, random_state=42)  # initialization max_depth=5, min_samples_split=20, min_samples_leaf=5, max_features=5, max_leaf_nodes=80

# MLP
# ml = MLPClassifier(activation='relu', solver='adam', alpha=0.1)

# LR
# ml = LogisticRegression(random_state=42)

# KNN
# ml = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=45, p=2)

# XGBoost
# ml = XGBClassifier(num_class=3, colsample_bytree=0.8, learning_rate=0.1, max_depth=10, n_estimators=200, objective='multi:softmax', subsample=0.6)

# AdaBoost
# ml = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=5),
#                         random_state=42,
#                         algorithm="SAMME.R",
#                         n_estimators=200, learning_rate=0.5)

# GraindentBoost
ml = GradientBoostingClassifier(n_estimators=150, learning_rate=0.5, max_depth=3, random_state=42)
# *********************************************** GridSearch-GradientBoost ******************************************************
# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'learning_rate': [0.01, 0.1, 0.5, 1],
#     'max_depth': [3, 4, 5, 10],
# }
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(ml, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
# grid_result = grid_search.fit(x_chi, y)
#
# print("Best parameters found: ", grid_result.best_params_)
# print("Best accuracy score: ", grid_result.best_score_)
# *********************************************** GridSearch-AdaBoost ******************************************************
# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'learning_rate': [0.01, 0.1, 0.5, 0.8, 1],
#     'base_estimator': ['DecisionTreeClassifier', 'deprecated'],
# }
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(ml, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
# grid_result = grid_search.fit(x_train, y_train)
#
# print("Best parameters found: ", grid_result.best_params_)
# print("Best accuracy score: ", grid_result.best_score_)
# *********************************************** GridSearch-XGBoost ******************************************************
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [3, 4, 5, 10, 20],
#     'learning_rate': [0.001, 0.01, 0.1],
#     'subsample': [0.6, 0.8, 1],
#     'colsample_bytree': [0.6, 0.8, 1],
#     'objective': ['multi:softmax'],
# }
#
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# grid_search = GridSearchCV(ml, param_grid, cv=kf, scoring='accuracy', n_jobs=-1)
# grid_result = grid_search.fit(x_train, y_train)
#
# print("Best parameters found: ", grid_result.best_params_)
# print("Best accuracy score: ", grid_result.best_score_)
# *********************************************** GridSearch-RF / DT******************************************************
# param_grid = {
#     'n_estimators': [10, 50, 100, 200],    # RF
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [None, 5, 10, 20, 30],       # RF
#     'min_samples_split': [2, 5, 10],       # RF
#     'min_samples_leaf': [1, 2, 3, 4, 5],         # RF
#     'max_features': [1, 2, 4, 5, 10, 20, 30, 40, 50, 60, 80, 90, 100, 120, 150, 200],    # DT
#     'max_leaf_nodes': [1, 2, 4, 5, 10, 20, 30, 40, 50, 60, 80, 90, 100, 120, 150, 200],  # DT
# }
# grid_search = GridSearchCV(estimator=ml, param_grid=param_grid, cv=5, scoring='accuracy')
# grid_search.fit(x_train, y_train)
# print("Best parameters found：", grid_search.best_params_)
# *****************************************************************************************************************
y_pred_list = []
y_test_list = []

for train_index, test_index in cv.split(x_chi, y):
    x_train, x_test = x_chi.iloc[train_index], x_chi.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    ml.fit(x_train, y_train)

    y_pred = ml.predict(x_test)  # KNN np.array(x_test)
    y_pred_list.append(y_pred)
    y_test_list.append(y_test)

for i in range(len(y_pred_list)):
    y_test = y_test_list[i]
    y_pred = y_pred_list[i]
    print(f"The{i + 1} fold classification report：")
    print(classification_report(y_test, y_pred, digits=5))
    print(f"The{i + 1} fold MCC：", matthews_corrcoef(y_test, y_pred))
