#Load libraries
from newsapi import NewsApiClient
import numpy as np
import nltk
import re
from nltk.corpus import stopwords #corpus is collection of text
from nltk.stem.porter import PorterStemmer
import pandas
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from datetime import datetime
import json
import requests


# set up vars
newsapi = NewsApiClient(api_key='')
model = tf.keras.models.load_model('model_multi.h5')
avg_stream_dataset = ''
author_counter = ''
src_counter = ''
metric = ''

# Data preparation function
def text_preparation(data):
    
    # Input vars
    nltk.download('stopwords')
    embedding_vector_feature_title = 10
    embedding_vector_feature_text = 100
    sent_length_title = 20
    sent_length_text = 1000
    vo_size=500
    ps_title = PorterStemmer()
    ps_text = PorterStemmer()
    corpus_title = []
    corpus_text = []

    #Copy df with title and description columns
    X_test=data[['title','description']]
    messages=X_test.copy()
    messages.reset_index(inplace=True)

    #Preproc text
    for i in range(0, len(messages)):
        print("Status: %s / %s" %(i, len(messages)), end="\r")
    
        #preproc title
        review = re.sub('[^a-zA-Z]', ' ',messages['title'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps_title.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus_title.append(review)
        
        #preproc text
        review = re.sub('[^a-zA-Z]', ' ',messages['description'][i])
        review = review.lower()
        review = review.split()
        
        review = [ps_text.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus_text.append(review)

    #Data frame representation for NN
    onehot_rep_title = [one_hot(words, vo_size) for words in corpus_title]
    onehot_rep_text = [one_hot(words, vo_size) for words in corpus_text]
    embedded_doc_title=pad_sequences(onehot_rep_title, padding='pre', maxlen=sent_length_title)
    embedded_doc_text=pad_sequences(onehot_rep_text, padding='pre', maxlen=sent_length_text)
    X_final_title=np.array(embedded_doc_title)
    X_final_text=np.array(embedded_doc_text)
    
    return X_final_title,X_final_text

#Set up current datetime for procedure
t = datetime.now()

while True:
    delta = datetime.now()-t
    #get TOPnews for las 15 min
    if delta.seconds >= 900:
        # Get news and prepoc it
        top_headlines = newsapi.get_top_headlines(q='covid', language='en')

        #Convert JSON to DataFrame
        data = pandas.DataFrame.from_dict(top_headlines)
        data = pandas.concat([data.drop(['articles'],axis=1), data['articles'].apply(pandas.Series)],axis=1)
        data['srs'] = 1

        for ind in data.index: 
            data['srs'][ind] = data['source'][ind]['name']

        #Preproc text with function
        X_final_title,X_final_text = text_preparation(data)

        # Score model
        data['prob'] = (1-model.predict ([X_final_title,X_final_text]))*100


        # Convert datetime format
        data['publishedAt'] = pandas.to_datetime(data['publishedAt'], unit='M', errors='ignore')


        #AVG by time result
        avg_df = data.groupby(['publishedAt','srs'],as_index=False)['prob'].mean()
        avg_df = pandas.DataFrame(avg_df).to_json(orient='records')

        #Count author
        author_df = pandas.DataFrame(data=data['author'].count(), index=["1"], columns=["author_count"])
        author_df = author_df.to_json(orient='records')

        #Count sources
        src_df = pandas.DataFrame(data=data['srs'].count(), index=["1"], columns=["src_count"])
        src_df = src_df.to_json(orient='records')

        #Calculate min,max and AVG probabi;ity of fake news
        metric_df = pandas.DataFrame(data=data['prob'].mean(), index=["1"], columns=["mean_prob"])
        metric_df['min'] = data['prob'].min()
        metric_df['max'] = data['prob'].max()
        metric_df = metric_df.to_json(orient='records')

        #Post result to Power BI
        r = requests.post(avg_stream_dataset, avg_df, verify = False)

        print(r.status_code)
        print(r.reason)

        r = requests.post(author_counter, author_df, verify = False)

        print(r.status_code)
        print(r.reason)

        r = requests.post(src_counter, src_df, verify = False)

        print(r.status_code)
        print(r.reason)

        r = requests.post(metric, metric_df, verify = False)

        print(r.status_code)
        print(r.reason)

        # Update 't' variable to new time
        t = datetime.now()