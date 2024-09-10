## Importing the libraries 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


## getting the word index and reversing it 
words_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in words_index.items()}


## Loading the model 
model = load_model('simple_rnn_imdb.h5')


## Creating helper functions 
# 1. To convert the input string to the format of required input 
# 2. To decode the reviews 

# Function to decode reviews 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,"?") for i in encoded_review])

# Function to preprocess user input 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [words_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


## Streamlit app part 
import streamlit as st 

st.title('IMDb Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify it as positive or negative")

user_input = st.text_area('Movie Review')


if st.button('Classify'):

    preprocessed_input = preprocess_text(user_input)
    

    # make prediction 

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'


    ## Display the result 

    st.write(f'Sentiment is {sentiment}')
    st.write(f'Prediction Score is {prediction[0][0]}')

else:
    st.write('Please enter a movie review')