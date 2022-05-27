# -*- coding: utf-8 -*-
"""
Created on Sun May  1 10:16:57 2022

@author: Domin
"""


decision = input('Please Select An Option:\n\nPress 2 To Predict RAG Status\n\nPress 3 to Predict Project Price\n\nSelection: ')

import datetime
from datetime import date 
import numpy as np 
import pandas as pd 
import os 

print(os.getcwd())

decision = int(decision)


## Reading the Data 
if decision == 3:
    date = datetime.datetime.now()
    
    
    file = input('Would you like to read an excel file? If so write the file name.\n\nSelection: ')
    sheet = input('Please type the name of the sheet: ')
    data = pd.read_excel(file + '.xlsx', sheet)
    """
    data = pd.read_excel('Project3Data_Updated.xlsx', sheet_name = 'Model_Data')"""
    data = data[['Start Date', 'Department', 'Month', 'Day', 'Fixed Cost', 
                 'Percent Offshore', 'Total Project Hours', 'Total Cost']]
    data_copy = data 
    
    data = data.drop('Start Date', axis = 1)
    

    
    
    
    
    
    
    from sklearn.model_selection import train_test_split
    
    
    
    
    #Num and Cat pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    
    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    
    
    cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(sparse=False))
        ])
    
    
    y = data['Total Cost']
    x = data.drop(['Total Cost'], axis = 1)
    
    
    
    num_attribs = ['Day', 'Percent Offshore',  'Total Project Hours', 'Fixed Cost', 'Month']
    cat_attribs = ['Department']
    
    
    
    
    
    preprocess_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.50, random_state = 31)
    
    
    
    X_train = preprocess_pipeline.fit_transform(
        X_train[num_attribs + cat_attribs])
    
    
    
    
    X_test = preprocess_pipeline.fit_transform(
        X_test[num_attribs + cat_attribs])
    
    
    from sklearn.ensemble import RandomForestRegressor
    
    param_grid = { 
        'n_estimators': [200, 500]
    }
    from sklearn.model_selection import GridSearchCV
    
    
    # to do a little parameter tuning 
    from sklearn.model_selection import GridSearchCV
    
    model = GridSearchCV(RandomForestRegressor(), param_grid, cv = 3)
    
    
    model.fit(X_train, y_train)
    
    
    
    y_pred = model.predict(X_test)
    
    print('\nModel Score: ',model.score(X_test, y_test),'\n')
    
    
    
    file = input('Please type the name of the file with the values you would like to predict?\nSelection: ')
    sheet = input('Please type the name of the sheet: ')
    
    test_data = pd.read_excel( file + '.xlsx' , sheet_name = sheet)
    test_data = test_data[['Start Date', 'Department', 'Month', 'Day', 'Fixed Cost', 
                 'Percent Offshore', 'Total Project Hours', 'Total Cost']]
    test_data_copy = test_data
    test_data = test_data.drop(['Start Date', 'Total Cost'], axis = 1)
    
    
    
    
    processed_test_data = preprocess_pipeline.fit_transform(
        test_data[num_attribs + cat_attribs])
    
    forecast_values = model.predict(processed_test_data)
    
    
    test_data_copy['Predicted Values'] = forecast_values
    name = 'Project3PredictionV9.xlsx'
    
    test_data_copy.to_excel(name)
    print('Results sent to Excel under file name: ', name)
    
    
    
    
    
    

    import datetime
    from datetime import date 
    import numpy as np 
    import pandas as pd 
    import os 

    print(os.getcwd())
    
    
if decision == 2:   
    
    ## Reading the Data 
    date = datetime.datetime.now()
    file = input('Would you like to read an excel file? If so write the file name.\n\nSelection: ')
    sheet = input('What is the name of the sheet? ')
        
    data = pd.read_excel(file + '.xlsx', sheet_name = sheet)
    """data = pd.read_excel('Project2UpdatedVersion.xlsx', sheet_name = 'Train Data')"""
    data = data[['Start_Date', 'Project_ID', 'Project_Name', 'Early Setbacks', 'Program', 'Project Manager Years Of Experience', 
                      'Flag Status ']]
    data_copy = data 
    
    data = data.drop(['Start_Date','Project_ID', 'Project_Name'], axis = 1)
    
    
    
    
    
    
    from sklearn.model_selection import train_test_split
    
    
    
    
    #Num and Cat pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    
    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    
    
    cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(sparse=False))
        ])
    
    
    y = data['Flag Status ']
    x = data.drop(['Flag Status '], axis = 1)
    
    num_attribs = ['Project Manager Years Of Experience']
    cat_attribs = ['Early Setbacks', 'Program']
    
    
    
    
    
    preprocess_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.90, random_state = 24)
    
    
    
    X_train = preprocess_pipeline.fit_transform(
        X_train[num_attribs + cat_attribs])
    
    
    
    
    X_test = preprocess_pipeline.fit_transform(
        X_test[num_attribs + cat_attribs])
    
    
    from sklearn.ensemble import RandomForestClassifier
    
    param_grid = { 
        'n_estimators': [200, 500, 1000]
    }
    from sklearn.model_selection import GridSearchCV
    
    
    # to do a little parameter tuning 
    from sklearn.model_selection import GridSearchCV
    
    model = GridSearchCV(RandomForestClassifier(), param_grid, cv = 3)
    
    
    model.fit(X_train, y_train)
    
    
    
    y_pred = model.predict(X_test)
    
    print('Model Score: ', model.score(X_test, y_test), '\n')
    
    
    
    
    file = input('Please type the name of the file with the values you would like to predict?\nSelection: ')
    sheet = input('What is the name of the sheet? ')
    test_data = pd.read_excel(file + '.xlsx', sheet_name = sheet)
    """test_data = pd.read_excel('Project2UpdatedVersion.xlsx', sheet_name = 'Prediction Data')"""
    test_data = test_data[['Start_Date', 'Project_ID', 'Project_Name','Early Setbacks', 'Program', 'Project Manager Years Of Experience']]
    test_data_copy = test_data
    test_data = test_data.drop(['Start_Date','Project_ID', 'Project_Name'], axis = 1)
    
    
    
    
    processed_test_data = preprocess_pipeline.fit_transform(
        test_data[num_attribs + cat_attribs])
    
    forecast_values = model.predict(processed_test_data)
    
    
    test_data_copy['Flag Status '] = forecast_values
    
    name = 'Project2PredictionV7.xlsx'
    
    test_data_copy.to_excel(name)
    print('Results sent to excel!\nFile Name: ', name)

