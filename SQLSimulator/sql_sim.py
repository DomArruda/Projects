import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import duckdb
from PIL import Image


image = Image.open("SQLSimulator/database_schema.png")
st.title("SQL SIMULATOR ⚙️")
st.image(image, caption = '', use_column_width = True)

# *** Data Loading ***
def read_data():
    df1 = pd.read_csv("SQLSimulator/Salespeople.csv")
    df2 = pd.read_csv("SQLSimulator/Salespeople_Data.csv")
    df3 = pd.read_csv("SQLSimulator/Transactions.csv")
    return df1, df2, df3

salespeople, sales, transactions = read_data()

# *** DuckDB Setup ***
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


button_bool = st.button("CLICK HERE TO GRAB THE FIRST 5 ROWS")

if button_bool: 
    df = query_database(query = "SELECT * FROM SALES LIMIT 5")
    st.markdown("*Query: SELECT * FROM SALES LIMIT 5*")
    st.dataframe(df, use_container_width = False)  # Adjusted for Streamlit display
    st.markdown(f'*Number of rows:     {df.shape[0]:,}*')  # Use .shape for Pandas
    st.markdown(f'*Number of cols:     {df.shape[1]:,}*')
    st.text('')
    st.text('')

# User Input
query = st.text_area(label = "Write your SQL Query here")

if str(query) != '':
    try: 
        st.text('')
        df = query_database(query = query)
        st.dataframe(df, use_container_width = False)
        st.markdown(f'*Number of rows:     {df.shape[0]:,}*')
        st.markdown(f'*Number of cols:     {df.shape[1]:,}*')
    except Exception as e: 
        st.markdown(f"Oops! Looks like we've encountered an error. Try checking your query. (Error: {e})")
