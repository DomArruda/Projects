import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from pyspark.sql import SparkSession
from PIL import Image
image = Image.open("SQLSimulator/database_schema.png")
spark = SparkSession.builder.appName("SQL_Sim").getOrCreate()
st.title("SQL SIMULATOR ⚙️")
st.image(image, caption = '', use_column_width = True)

def read_data():
    df1 = spark.read.csv(r"SQLSimualtor/Salespeople.csv", header=True, inferSchema=True)
    df2 = spark.read.csv(r"SQLSimulator/Salespeople_Data.csv", header=True, inferSchema=True)
    df3 = spark.read.csv(r"SQLSimulator/Transactions.csv", header=True, inferSchema=True)
    return df1, df2, df3

salespeople, sales, transactions= read_data()
@st.cache
def create_tables():
    salespeople.createOrReplaceTempView("Salespeople")
    sales.createOrReplaceTempView("Sales")
    transactions.createOrReplaceTempView("Transactions")

create_tables()
button_bool = st.button("CLICK HERE TO GRAB THE FIRST 5 ROWS")

def query_database(query):
    df = spark.sql(query)
    return df

if button_bool: 
    df = query_database(query = "SELECT * FROM SALES LIMIT 5")
    st.markdown("*Query: SELECT * FROM SALES LIMIT 5*")
    st.dataframe(df.toPandas())
    button_bool = False
    st.markdown(f'*Number of rows:         {df.count():,}*')
    st.markdown(f'*Number of cols:         {len(df.columns):,}*')
    st.text('')
    st.text('')
    

query = st.text_area(label = "Write your SQL Query here")
if str(query) != '':
    try: 
        st.text('')
        df = query_database(query = query)
        st.dataframe(df.toPandas(), use_container_width = False)
        st.markdown(f'*Number of rows:         {df.count():,}*')
        st.markdown(f'*Number of cols:         {len(df.columns):,}*')
    except: 
        st.markdown("Oops! Looks like we've encountered an error. Try checking your query")
