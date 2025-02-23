from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model
import googleapiclient.discovery
import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle

class preprocess():
    def __init__(self):
        pass
    def preprocessComment(self,sents):
        with open('res/word_dict.pkl','rb') as f:
            word_dict = pickle.load(f)
        x=[]
        for s in sents:
            result = ''.join([i for i in s if not i.isdigit()])
            x.append(result)

        x =[re.sub('[^a-zA-Z]',' ',words) for words in x]
        voca_len =10000
        x_ohe = [word_dict[word]if word in word_dict.keys()  else one_hot(word,30000) for word in sents]
        sent_len = 100
        x_pad = sequence.pad_sequences(x_ohe,maxlen=sent_len)
        return x_pad
    
    def analysis(self,sents):
        x= self.preprocessComment(sents)
        model = load_model('res/SentimentRNN.h5')
        pred = model.predict(x)
        return [np.argmax(i) for i in pred]

    def fetchComments(self,link):
        id = link.split("=")[1].split("&")[0]
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = st.secrets["api_auth"]


        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=DEVELOPER_KEY)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=id,
            maxResults=100
        )
        response = request.execute()

        comments = []

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['updatedAt'],
                comment['likeCount'],
                comment['textDisplay']
            ])

        df = pd.DataFrame(comments, columns=['author', 'published_at', 'updated_at', 'like_count', 'text'])
        return df['text']
    def analysisReport(self,comments,prediction):
        pred = pd.DataFrame(prediction, columns=['sentiment'])
        pred =pd.concat([comments,pred], axis=1)
        df_pos = pred.loc[pred['sentiment'] ==2]
        df_neg = pred.loc[pred['sentiment'] ==0]
        df_neu = pred.loc[pred['sentiment'] ==1]
        return df_pos,df_neg,df_neu
