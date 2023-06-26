
import streamlit as st 
from bs4 import BeautifulSoup
import numpy as np
import re
import pandas as pd
import requests
import numpy 
import torch, torchvision
from transformers import AutoTokenizer, AutoModelForSequenceClassification 
from PIL import Image
import csv
import plotly.express as px
import pyodbc
downloaded_mock = False
#%%

@st.cache(allow_output_mutation = True)

def load_model(boolean): 
    if boolean is True: 
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        return model 
@st.cache(allow_output_mutation = True)
def load_tokenizer(boolean): 
    if boolean is True: 
        tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        return tokenizer
#%% 

model = load_model(True)
tokenizer = load_tokenizer(True)

sentimentDict = {'1': 'Very Negative', '2': 'Negative', '3': 'Neutral', '4': 'Positive', '5': 'Very Positive'}


def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1 

def reviews_scrape(reviews: str, num_pages): 
        
        link_list = []
        df_list = []
        link_list.append(reviews)
        for i in range(1, num_pages): 
            next_page = reviews + '?start=' + str(i) + '0'
            link_list.append(next_page)
        for j in link_list:
            r = requests.get(j)
            soup = BeautifulSoup(r.text, 'html.parser')
            regex = re.compile('.*comment.*')
            results = soup.find_all('p', {'class':regex})
            reviews = [result.text for result in results] 
            df = pd.DataFrame(np.array(reviews), columns=['review'])
            df['score'] = df['review'].apply(lambda x: sentiment_score(x[-512:])) 
            df_list.append(df)
        merged = pd.concat(df_list)
        return merged

def reviews_csv(reviews): 
    data = pd.read_csv(reviews, header = "infer")
    data_columns = data.columns 
    data_columns = data_columns.insert(0, 'None Selected')
    additional_instructs = ''
    st.text(downloaded_mock)
    if downloaded_mock != False: 
        additional_instructs = 'Use reviews.text for Mock Data.' 
        
    selected_column = st.selectbox(f'Which column contains the reviews?{additional_instructs}', data_columns)
    totalReviews = len(data)
    st.text(f'Total number of reviews detected: {totalReviews}')
    numReviews = int(st.number_input('How many reviews should I analyze? '))
    #df = pd.DataFrame(data, columns=['review'])
    if selected_column == data_columns[0] or numReviews <= 0 or numReviews > totalReviews  : 
        st.stop()
    else:
        data = data.iloc[:numReviews]
        data['score'] = data[selected_column].apply(lambda x: sentiment_score(x[-512:])) 
        return data
    
def SQL_scrape(query, conn): 
    data = pd.read_sql(query, conn)
    data_columns = data.columns 
    data_columns = data_columns.insert(0, 'None Selected')
    selected_column = st.selectbox('Which column contains the reviews?', data_columns)
    totalReviews = len(data)
    st.text(f'Total number of reviews detected: {totalReviews}')
    numReviews = int(st.number_input('How many reviews should I analyze? '))
    if selected_column == data_columns[0] or numReviews <= 0 or numReviews > totalReviews  : 
        st.stop()
    else:
        data = data.iloc[:numReviews]
        data['score'] = data[selected_column].apply(lambda x: sentiment_score(x[-512:])) 
        return data
    


st.title('Jim Bot :robot_face:')
st.text('')
st.text('')

with st.sidebar.header('**Upload your CSV or Sample Data**'):
    uploaded_file = st.sidebar.file_uploader("Please Upload a CSV file", type=["csv"])
   
        
   



            
with st.sidebar.header('Upload reviews via website link: '):
    uploaded_link = st.text_input('Want to scrape reviews instead? Paste your website link here: ', placeholder = '')


with st.sidebar.header('Upload the number of pages to grab reviews from'):
    num_pages = int(st.number_input('Enter the number of pages to scrape:', step = 1, min_value = 1))


with st.sidebar.header('Connect to SQL: '): 
    st.header('Connect to SQL')
with st.sidebar.header('Connect to SQL: '):
   server =  st.text_input('Please enter the name of the server: ')

with st.sidebar.header('Please enter the name of the database: '):
    database =  st.text_input('Please enter the name of the database: ')

st.header('Please Upload A CSV, Website Link, Or Connect To SQL')
try:
     mock_data = pd.read_csv('SentimentApp/AmazonProductReviews.csv')
     data_csv = mock_data.to_csv(index = False).encode('utf-8')
     downloaded_mock = st.download_button("Don't have any test data? Click here to download sample product review data!", data_csv, 'AmazonReviews.csv')
except: 
     st.text('')
        
st.text('')
st.text('')
st.text('')
test_text = st.text_area("""**Write a mock-review here and I'll return a score from 1 (negative emotion) to 5 (positive emotion):**""")
st.text('')
st.text('')
st.text('')
st.text('')



if test_text != '': 
    try: 
        sentScore = sentiment_score(test_text)
        sentCategory = sentimentDict[str(sentScore)]
        st.markdown(f'**Sentiment Score:  ' + str(sentScore) + f' ({sentCategory})' + '**')
    except: 
        st.text('')

if uploaded_link == '' and uploaded_file is not None:
    st.markdown('**Successfully uploaded csv!**')
    review_df = reviews_csv(uploaded_file)
    st.dataframe(review_df, use_container_width= True)
    fig = px.histogram(review_df, x = 'score', title = 'Histogram of Scores')
    
    fig.update_traces(marker_line_color = 'white', marker_line_width = 1.0)
    
    st.plotly_chart(fig)
    review_csv = review_df.to_csv(index = False).encode('utf-8')
    download = st.download_button('Click below to download your sentiment report: ', 
                       review_csv, 'sentiment.csv')
    
    

elif uploaded_link != '' and num_pages is not None and uploaded_file is None: 
    st.markdown('**Successfully received link!**') 
    review_data = reviews_scrape(uploaded_link, num_pages)
    review_data.reset_index(inplace= True, drop = True)
    st.dataframe(review_data, use_container_width= True)
    fig = px.histogram(review_data,  x= 'score', title = 'Histogram of Scores')
    
    fig.update_traces(marker_line_color = 'white', marker_line_width = 1.0)
    
    st.plotly_chart(fig)
    review_csv = review_data.to_csv(index = False).encode('utf-8')
    st.download_button('Click here to download your sentiment report', 
                       review_csv, 'sentiment.csv')
    
elif  server != '' and database != '' and uploaded_file is None and uploaded_link == '':
    try: 
        server_login = f'Server={server};'
        database_login = f'Database={database};'
        connection_login = f'Trusted_Connection=yes;'
    
        login = 'Driver={SQL Server};' + server_login + database_login + connection_login
        conn = pyodbc.connect(login)
        cursor = conn.cursor()
        
        query = st.text_area('Please enter your query here')
        if query == '': 
            st.stop()
        else: 
            try:
                review_data = SQL_scrape(query, conn)
                review_data.reset_index(inplace= True, drop = True)
                st.dataframe(review_data, use_container_width= True)
                fig = px.histogram(review_data,  x= 'score', title = 'Histogram of Scores')
                
                fig.update_traces(marker_line_color = 'white', marker_line_width = 1.0)
                
                st.plotly_chart(fig)
                review_csv = review_data.to_csv(index = False).encode('utf-8')
                st.download_button('Click here to download your sentiment report', 
                                   review_csv, 'sentiment.csv')
                
            except: 
                st.text('')
             
    except: 
         st.text('Encountered an error')

    

