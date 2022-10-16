#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from PIL import Image
from xgboost import plot_importance
import matplotlib.pyplot as plt

image = Image.open('SymbolicRegression/symbo.jpeg')
st.image(image,caption = 'Python Symbolic Regression', use_column_width = True)
import csv
import random
import math
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
import pandas as pd
from numpy import add, multiply as mul, subtract as sub, divide as div
import sympy as smp
from sympy.core import sympify
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
from xgboost import XGBRegressor 
from sympy import *
init_printing()





def run_regression(uploaded_file):
    

    with st.sidebar.header('**2. Set Train Size**'):
        train_size = st.sidebar.slider('Data split ratio (% for Training Set)', min_value = 20, max_value = 90, step = 10)
   

         
    with st.sidebar.header('**3. Set Genetic Algorithm Parameters**'):
        pop_size = st.sidebar.slider('Choose Population Size', min_value = 500, max_value = 10000, step = 100)
        tournament_size = st.sidebar.slider('Select Tournament Size', min_value = 0, max_value = pop_size, 
                                            step = int(pop_size/10))
        generation_num = st.sidebar.slider('Choose Number of Generations', min_value = 1, max_value = 100, step = 1)
       
      
        
 
    with st.sidebar.header('**4. Choose Allowed Functions:'):
        functions_gp = ('add', 'sub', 'mul', 'div', 'log', 'cos', 'sin', 'tan', 'inv')
        functions_select = st.multiselect('Choose Your Functions: ', functions_gp, 
                                               default = ('add', 'sub', 'mul', 'div', 'log'))
        
    data = pd.read_csv(uploaded_file)
    data = pd.get_dummies(data)
    data_copy = data.copy()
    for items in data.columns: 
        if 'Unnamed' in items:
            data.drop(items, axis = 1, inplace = True)
    
    variable_list = tuple(data.columns.insert(0, 'None Selected'))
    
    variables = st.multiselect('Choose Which Variables To Include:', data.columns, default = tuple(data.columns))
    st.write('Note: Categorical Variables Will Automatically Be Converted To Dummies')
    data = data[list(variables)]

    
    
    st.write('**A Peek At The Data**')
    st.write(data.head(5))
    
    
    variables_list = variables 
    variables_list.insert(0, 'None Selected')
    y_value = st.selectbox('Select The Value You Would Like To Predict:', variables_list)
    
 
    
    
    if y_value is None or y_value == 'None Selected': 
        st.stop()
        
    y = data[y_value]
    X = data.drop([y_value], axis = 1)
    num_variables = len(X.columns)
    
 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = train_size)
    
    button_1 = st.button('Not Sure Which Variables To Include? Click Here For Feature Importance Scores!')
    if button_1 != False:
        model = XGBRegressor(num_estimators = 300, random_state = 69) 
        model.fit(X_train, y_train)
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(figure= feat_importances.nlargest(20).sort_values(ascending = True).plot(kind='barh') )

    
    st.text('')
    st.text('')

    button_2 = st.button('Ready To Run The Program? Click Here!')
    
    if button_2 != False: 
        st.write('Running Your Program! Note: Larger Datsets May Take Several Minutes.')
        est_gp = SymbolicRegressor(population_size=pop_size,
                           generations=generation_num, stopping_criteria=0.001,
                           p_crossover=0.2, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.15, p_point_mutation=0.2,
                           tournament_size = tournament_size, function_set= functions_select,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.00001, random_state=42, warm_start = True)
        est_gp.fit(X_train, y_train)
        
        
       
        r2_score(y_test,est_gp.predict(X_test.iloc[:, :num_variables])) 
        
        MAPE(y_test,est_gp.predict(X_test))
    
    
        converter = {
        'sub': lambda x, y : x - y,
        'div': lambda x, y : x/y,
        'mul': lambda x, y : x*y,
        'add': lambda x, y : x + y,
        'neg': lambda x    : -x,
        'pow': lambda x, y : x**y
        }
        
        
        simplify(sympify(str(est_gp._program), locals = converter))
        
        #Our formula!!!
    
    
        approx_formula = str((simplify(sympify(str(est_gp._program), locals = converter))))
        
        
    
        error = round(MAPE(y_test,est_gp.predict(X_test)) * 100, 5)
        
        r2 = round(r2_score(y_test,est_gp.predict(X_test.iloc[:, :num_variables])),5)
     
        
        
        variable_list = ['X' + str(i) for i in range(0, len(X.columns))]
        og_variables = X.columns
        replace_variables = tuple(zip(variable_list, og_variables))
        
        
        replace_df = pd.DataFrame(replace_variables, columns = ['Keys', 'Values'])
        
        
        for i in range(len(replace_df)): 
          approx_formula = approx_formula.replace(replace_df['Keys'].iloc[i], replace_df['Values'].iloc[i])
        
        st.write('Formula: ')
        st.text('')
        st.text('')
        st.latex(approx_formula)
      
        mae_error = round(MAE(y_test,est_gp.predict(X_test)) * 100, 5)
        
        st.latex(f'MAPE: {error:.2f}%')
        st.latex(f'MAE: {mae_error:.2f}')
        st.latex(f'R-Squared : {r2:.5f}')




with st.sidebar.header('**1.Upload your CSV data**'):
    uploaded_file = st.sidebar.file_uploader("Please Upload a CSV file", type=["csv"])
    uploaded_file = pd.read_csv('SymbolicRegression/PhysicsData.csv')
    
    
 
    
 
    
if uploaded_file != None:
    run_regression(uploaded_file)
    
    
    
        
    
