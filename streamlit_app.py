import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile
import google.generativeai as genai
import os
from config import API_KEY  # Assuming API key is stored in config.py


# Page title
st.set_page_config(page_title='ML Model Building', page_icon='🤖')
st.title('🤖 ML Model Building')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users create study plans for themselves to increase test scores')

  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, go to the sidebar and 1. Select a data set and 2. Adjust the model parameters by adjusting the various slider widgets. As a result, this would initiate the ML model building process, display the model results as well as allowing users to download the generated models and accompanying data.')

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- Drug solubility data set
  ''', language='markdown')
  
  st.markdown('Libraries used:')
  st.code('''- Pandas for data wrangling
- Scikit-learn for building a machine learning model
- Altair for chart creation
- Streamlit for user interface
  ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('What study plan can we make for you?')

    st.subheader('Number of days till test/quiz')
    sleep_time = st.slider('Days', 0, 14, 0)
    st.header('Choose Subject')
    with st.expander('Subject'):
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)', 0, 1000, 100, 100)
        parameter_max_features = st.select_slider('Max features (max_features)', options=['all', 'sqrt', 'log2'])
        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 2, 10, 2, 1)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)


# Initialize Google Generative AI client
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("Google Generative AI Chat")

# Initialize session state for messages if not already initialized
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input in real-time
prompt = st.text_input("You:", value="")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using Google Generative AI model
    response = model.generate_content(prompt)
    
    st.session_state["messages"].append({"role": "assistant", "content": response.text})
    with st.chat_message("assistant"):
        st.markdown(response.text)