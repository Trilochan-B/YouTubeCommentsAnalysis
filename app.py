import streamlit as st
from utils import preprocess
import plotly.express as ex
import pandas as pd

pre = preprocess()
st.title("YouTube comments analysis")
st.write('Disclaimer* This model is trained on English vocabulary so prediction may vary on regional laguage')
st.image("res/vdoid-1.jpg")

vdo_id = st.text_input("Provide video link here ")
bt =st.button('Analyze')
if bt:
    comments = pre.fetchComments(vdo_id)
    pred = pre.analysis(comments)
    df_pos,df_neg,df_neu = pre.analysisReport(comments,pred)
    
    df = pd.DataFrame({
        'Sentiment' : ['Positive','Negative','Neutral'],
        'Count' : [df_pos.shape[0],df_neg.shape[0],df_neu.shape[0]]
    })

    
    fig = ex.pie(df, names='Sentiment', values='Count')
    st.plotly_chart(fig)
   
    st.write("Positive")
    st.dataframe(df_pos)
    st.write("Negative")
    st.dataframe(df_neg)
    st.write("Neutral")
    st.dataframe(df_neu)

