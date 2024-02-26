import pandas as pd
import numpy as np 
import streamlit as st
from code_editor import code_editor
from sqlalchemy import create_engine
import duckdb
from PIL import Image


image = Image.open("SQLSimulator/database_schema.png")
st.title("SQL SIMULATOR ⚙️")
st.image(image, caption = '', use_column_width = True)
st.text('')
st.text('')

# Helper Functions...
def read_data():
    df1 = pd.read_csv("SQLSimulator/Salespeople.csv")
    df2 = pd.read_csv("SQLSimulator/Salespeople_Data.csv")
    df3 = pd.read_csv("SQLSimulator/Transactions.csv")
    return df1, df2, df3

salespeople, sales, transactions = read_data()


def create_tables():
    conn = duckdb.connect()
    conn.register("Salespeople", salespeople)
    conn.register("Sales", sales)
    conn.register("Transactions", transactions)

create_tables()


def query_database(query):
    conn = duckdb.connect()
    df = conn.execute(query).df()
    return df


button_bool = st.button("CLICK HERE TO GRAB THE FIRST 5 ROWS OF THE SALES TABLE")
st.text('')

if button_bool:
    df = query_database(query = "SELECT * FROM SALES LIMIT 5".lower())
    st.markdown("*Query: SELECT * FROM SALES LIMIT 5*")
    st.dataframe(df, use_container_width = False)  
    st.markdown(f'*Number of rows:{df.shape[0]:,}*')  
    st.markdown(f'*Number of cols:{df.shape[1]:,}*')
    st.text('')
    st.text('')



# User Input

# there's better ways to do this but oh well. 

st.markdown("*Write Your SQL Code Below. Press ctrl + enter to run the query.*")
query = code_editor(code="/*Write your code here!*/\n\n\n\n\n\n", lang="sql", key="editor", height = 500, theme= "light")

if str(query['text']) != '':
    try:
        for query in query['text'].lower().split(';'):
            st.text('')
            df = query_database(query = query)
            st.markdown(f'*Number of rows:{df.shape[0]:,}*')
            st.markdown(f'*Number of cols:{df.shape[1]:,}*')
            st.dataframe(df, use_container_width = False)
           
    except Exception as e:
        st.markdown(f"Oops! Looks like we've encountered an error. Try checking your query. (Error: {e})")


