

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
#%%

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

#%% 



def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits))+1 

def reviews_scrape(reviews: str): 
    r = requests.get(reviews)
    soup = BeautifulSoup(r.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews = [result.text for result in results] #make sure to learn this
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    df['score'] = df['review'].apply(lambda x: sentiment_score(x[-512:])) 
    return df

def reviews_csv(reviews): 
    data = pd.read_csv(reviews)
    data_columns = data.columns 
    data_columns = data_columns.insert(0, 'None Selected')
    selected_column = st.selectbox('Which column contains the reviews?', data_columns)
    #df = pd.DataFrame(data, columns=['review'])
    if selected_column == data_columns[0]: 
        st.stop()
    else:
        data['score'] = data[selected_column].apply(lambda x: sentiment_score(x[-512:])) 
        return data
    

#review_df = reviews('https://www.yelp.com/biz/moonstar-chinese-restaurant-east-providence')
#%% 

#image = Image.open('Downloads/sentiment.jpg')
#st.image(image, caption = "Python Sentiment Analysis", use_column_width = True)


st.title('Jim Bot :robot_face:')
with st.sidebar.header('**Upload your CSV data.**'):

    uploaded_file = st.sidebar.file_uploader("Please Upload a CSV file", type=["csv"])
    

with st.sidebar.header('Upload reviews via website link: '):
    uploaded_link = st.text_input('Paste your website link here: ', placeholder = '')



if uploaded_link == '' and uploaded_file is not None:
    st.markdown('**Successfully uploaded csv!**')
    review_df = reviews_csv(uploaded_file)
    st.dataframe(review_df, use_container_width= True)
    fig = px.histogram(review_df, x = 'score', title = 'Histogram of Scores')
    st.plotly_chart(fig)
    review_csv = review_df.to_csv(index = False).encode('utf-8')
    st.download_button('Click below to download your sentiment report: ', 
                       review_csv, 'sentiment.csv')
    
    

elif uploaded_link != '' and uploaded_file is None: 
    st.markdown('**Successfully received link!**') 
    review_data = reviews_scrape(uploaded_link)
    st.dataframe(review_data, use_container_width= True)
    review_csv = review_data.to_csv(index = False).encode('utf-8')
    fig = px.histogram(review_csv, 'score', x= 'score', title = 'Histogram of Stores')
    st.plotly_chart(fig)
    st.download_button('Click here to download your sentiment report', 
                       review_csv, 'sentiment.csv')
    
    
    
    
else: 
    st.stop()





