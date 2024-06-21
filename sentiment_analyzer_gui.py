import requests
import streamlit as st
from streamlit_lottie import st_lottie
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

#---------- Streamlit page configuration---------
st.set_page_config(page_title="My Webpage", page_icon=":tada:", layout="wide")

#---------- Load Lottie files------------
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_file1 = load_lottieurl("https://lottie.host/50d1ea50-48cf-49b9-82f4-be81517edc24/U6mGbiJefP.json")
lottie_file2 = load_lottieurl("https://lottie.host/e3692a71-8b53-4220-8ed9-2d85dc3bc386/wfjNSN0gT4.json")
lottie_file3 = load_lottieurl("https://lottie.host/cb7d0dd0-d6ea-4800-af47-8585fb7c2239/rJJGnWaYOI.json")
lottie_file4 = load_lottieurl("https://lottie.host/8faf3e07-261d-4236-a598-dd257772949e/aXM05hXn9a.json")
lottie_file5 = load_lottieurl("https://lottie.host/2cad90f0-97a4-4e81-b1e1-01433a6905b4/otqRRKYzGH.json")
lottie_file6 = load_lottieurl("https://lottie.host/88e0526f-ac78-464c-9c03-f095b8c0fe3e/j61KYKBcxO.json")

#----------- Load models previously built------------
model1 = load_model('CNN_sentiment_analysis_model.h5')
model2 = load_model('LSTM_sentiment_analysis_model.h5')

#--------------Analyzing sentiment of user review--------------
def analyze_review1(review):

    #Load tokenizer to pre-process input review 
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=500)

    #Predicting user review label
    prediction = model1.predict(padded_sequences)[0][0]
    
    label = 'positive' if prediction >= 0.5 else 'negative'
    return label

#--------------Analyzing sentiment of user review--------------
def analyze_review2(review):
   
    #Load tokenizer to pre-process input review 
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
        
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequences = pad_sequences(sequences, maxlen=500)

    #Predict user review label
    prediction = model2.predict(padded_sequences)[0][0]
    
    label = 'positive' if prediction >= 0.5 else 'negative'
    return label

#Custom CSS
st.markdown("""
    <style>
    .stTextArea textarea {
        border: 2px solid white;
        background-color: black;
        color: white;
    }
    .result-text {
        font-size: 30px; 
    }
    .about-project{
        text-align: center;       
    }
    .about-me{
        text-align: center;
        color: grey;
    }
    .text{
        color: grey;
        font-size: 15px;
        margin-bottom: 30px;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        margin-bottom: 45px;
    }
    </style>
    """, unsafe_allow_html=True)


def select_model(model):
    st.session_state.selected_model = model

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'CNN'

#--------- Streamlit gui------
with st.container():
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown("<div class='title'>SentimentWave</div>",unsafe_allow_html=True)
        

        #Buttons for selecting model
        with left_column:
            col1, col2,col3,col4, col5 = st.columns(5)
            with col1:
                if st.button('CNN', key='cnn'):
                    st.session_state.selected_model = 'CNN'
               
            with col2:
                if st.button('LSTM', key='lstm'):
                    st.session_state.selected_model = 'LSTM'
                
        if st.session_state.selected_model == 'CNN':
            st.markdown("<div class= 'text'>Accuracy: 86%</div>",unsafe_allow_html=True)
        if st.session_state.selected_model == 'LSTM':
            st.write("<div class ='text'>Accuracy: 87%</div>",unsafe_allow_html=True)
        
        #textbox for user to write movie review
        
        user_review = st.text_area("Enter your movie review:")

        if st.button("Analyze"):
            if st.session_state.selected_model == 'CNN':
                sentiment_label = analyze_review1(user_review)

            if st.session_state.selected_model == 'LSTM':
                sentiment_label = analyze_review2(user_review)

            if user_review:
                result_text = "Positive" if sentiment_label == 'positive' else "Negative"
  
                placeholder = st.empty()
                accumulated_text = ""

                #To display result in iterations
                for char in result_text:
                    accumulated_text += char
    
                    if result_text == "Positive":
                        placeholder.markdown(f"<span class='result-text'>{accumulated_text} :thumbsup: </span>", unsafe_allow_html=True)
                        time.sleep(0.2)
                    else:
                        placeholder.markdown(f"<span class='result-text'>{accumulated_text} :thumbsdown: </span>", unsafe_allow_html=True)
                        time.sleep(0.2)
    
            else:
                st.write("Please enter a review to analyze.")

    #------------Emoji Animations----------
    with right_column:
        rows = [st.columns(2) for _ in range(3)]
        files = [lottie_file1, lottie_file2, lottie_file3, lottie_file4, lottie_file5, lottie_file6]

        for row, (file1, file2) in zip(rows, zip(files[::2], files[1::2])):
            with row[0]:
                st_lottie(file1, height=130)
            with row[1]:
                st_lottie(file2, height=130)

    #-----------About section-------------
    with st.container():
        
        st.write("---")
        

        st.markdown("<div class='about-me'>By Maryam Shahid</div>", unsafe_allow_html=True)
        st.markdown("<div class='about-me'>June, 2024</div>", unsafe_allow_html=True)
        st.markdown("<div class='about-me'>For Information Retrieval Course - FAST NUCES</div>", unsafe_allow_html=True)
        st.write("#")
        st.markdown("""<div class='about-project'>Sentiment Analysis model trained using both 
                 Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks, 
                 leveraging the comprehensive Large Movie Review Dataset v1.0. 
                 This approach is grounded in the methodologies presented in the research paper, 
                 “Performance Analysis of Different Neural Networks for Sentiment Analysis on IMDb 
                 Movie Reviews.”
                 By integrating CNNs and LSTMs, the model effectively captures and analyzes the 
                 intricate patterns and contextual dependencies within movie reviews.</div>""", unsafe_allow_html=True )






        

