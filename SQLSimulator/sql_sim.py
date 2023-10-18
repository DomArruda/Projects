
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
st.title("SQL SIMULATOR ⚙️")
try: 
    st.markdown('## Please write a query from the SALES table into the text input below')
    st.text('')
    st.text('')
    @st.cache
    def read_data():
        data = pd.read_csv(r"SQLSimulator/SaleTransactions.csv", delimiter = ',')
        if 'index' in list(data.columns):
            data = data.drop('index', axis = 1)
        return data

    data = read_data()
    engine = create_engine('sqlite://memory', echo=False)
    data.to_sql(name = 'Sales', con = engine, index = False)

    button_bool = st.button("CLICK HERE TO GRAB THE FIRST 5 ROWS")

    def query_database(engine, query):
        conn = engine.connect()
        execution_result = conn.execute(query)
        df = pd.DataFrame(execution_result.fetchall())
        df.columns = execution_result.keys()
        return df

    if button_bool: 
        df = query_database(engine = engine, query = "SELECT * FROM SALES LIMIT 5")
        st.dataframe(df)
        button_bool = False
        st.markdown(f'*Number of rows:         {df.shape[0]:,}*')
        st.markdown(f'*Number of cols:         {df.shape[1]:,}*')
        st.text('')
        st.text('')
        

    query = st.text_area(label = "Write your SQL Query here")
    if str(query) != '':
        try: 
            st.text('')
            df = query_database(engine = engine, query = query)
            st.dataframe(df, use_container_width = False)
            st.markdown(f'*Number of rows:         {df.shape[0]:,}*')
            st.markdown(f'*Number of cols:         {df.shape[1]:,}*')
        except: 
            st.markdown("Oops! Looks like we've encountered an error. Try checking your query")
except Exception as e:
    st.text('')
    st.text('')
    st.markdown('# Looks like the website/app may be down! Hang tight!')
    st.text(e)




    
