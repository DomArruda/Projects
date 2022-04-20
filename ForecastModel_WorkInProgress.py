# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:25:14 2022

@author: darruda2.ga
"""

import pandas as pd 

import numpy as np 

data = pd.read_excel('ForecastProject3.xlsx', sheet_name = 'Model_Data')


data = data.dropna()
data = data.drop_duplicates()

#Num and Cat pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


cat_attribs = []
num_attribs = ['Day', 'Month', 'Percent Of Senior Members Assigned', 'Fixed Cost', 'Total Cost', 'Percent Offshore', 'Total Project Hours']
num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])


cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False))
    ])


preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])


x = data
y= data.drop('Total Cost', axis = 1)


from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 24)


X_train = preprocess_pipeline.fit_transform(
    X_train[num_attribs + cat_attribs])




X_test = preprocess_pipeline.fit_transform(
    X_test[num_attribs + cat_attribs])


# to do a little parameter tuning 
from sklearn.ensemble import RandomForestRegressor 
param_grid = {'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'n_estimators': [200, 400, 1200, 1400, 1600, 1800, 2000]}

from sklearn.model_selection import GridSearchCV

model =GridSearchCV(RandomForestRegressor(), param_grid, cv = 2)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
from sklearn.metrics import r2_score



print(r2_score(y_test, y_pred))