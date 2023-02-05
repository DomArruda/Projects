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
    selected_column = st.selectbox('Which column contains the reviews?', data_columns)
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
    


st.title('Jim Bot :robot_face:')
with st.sidebar.header('**Upload your CSV data.**'):

    uploaded_file = st.sidebar.file_uploader("Please Upload a CSV file", type=["csv"])


with st.sidebar.header('Upload reviews via website link: '):
    uploaded_link = st.text_input('Paste your website link here: ', placeholder = '')

with st.sidebar.header('Upload the number of pages to grab reviews from'):
    num_pages = int(st.number_input('Enter the number of pages to scrape:', step = 1))




if uploaded_link == '' and uploaded_file is not None:
    st.markdown('**Successfully uploaded csv!**')
    review_df = reviews_csv(uploaded_file)
    st.dataframe(review_df, use_container_width= True)
    fig = px.histogram(review_df, x = 'score', title = 'Histogram of Scores')
    
    fig.update_traces(marker_line_color = 'white', marker_line_width = 1.0)
    
    st.plotly_chart(fig)
    review_csv = review_df.to_csv(index = False).encode('utf-8')
    st.download_button('Click below to download your sentiment report: ', 
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
    
    
    
    
else: 
    st.stop()

