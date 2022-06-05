from tkinter import TRUE
import tensorflow as tf
from keras.models import load_model
import keras
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import BertTokenizer
from transformers import TFBertModel
import streamlit as st
from st_btn_select import st_btn_select
import nltk
nltk.download('punkt')

model = tf.keras.models.load_model('model.h5')
inputText = 'very cool'

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

input_ids = tf.keras.layers.Input(shape = (256,), name = 'input_ids', dtype = 'int32')
attentionMasks = tf.keras.layers.Input(shape = (256,), name = 'attention_mask', dtype = 'int32')

def prepareData(inputText, tokenizer):
    token = tokenizer.encode_plus (
        inputText,
        max_length = 256,
        truncation = True,
        padding = 'max_length',
        add_special_tokens = True,
        return_tensors = 'tf'

    )

    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64) 
    }

tokenizedTextInput = prepareData(inputText, tokenizer)
probs = model.predict(tokenizedTextInput)
# print(probs)
# print(probs[0][0])
# print(probs[0][1])

selection = st_btn_select(('SCANNER', 'ABOUT', 'HOW IT\'S MADE'))

userTextInput = ""

if selection == 'SCANNER':

    st.title('OK2Say')
    st.subheader('Evaluating sentences for controversy with NLP')
    st.text('Created by Grant Hough')
    
    st.subheader('Select your controversy threshold percentage')
    tolerance = st.slider(label = "For example, a value of 73 means that anything with a controversy percentage above 73% will be flagged and shown.\nEverything else will not be shown.", min_value = 0, max_value = 100, value =  50)

    userTextInput = st.text_area("Provide the text you would like to be scanned:", userTextInput)
    sentences = nltk.sent_tokenize(userTextInput)

    sentenceOutputList = []
    valueOutputList = []

    for sentence in sentences: 
        number = round(model.predict(prepareData(sentence, tokenizer))[0][1] * 100, 2)
        if (number > tolerance): 
            sentenceOutputList.append('"' + sentence + '"')
            valueOutputList.append(str(number) + "%")

    d = {'Sentence': sentenceOutputList, 'Chance to be Controversial': valueOutputList}
    df = pd.DataFrame(d)
   
    if(len(sentences) > 0):
        st.subheader('Potentially Controversial Sentences')
        st.table(df)

if selection == 'ABOUT':

    st.title('OK2Say')
    st.subheader('Times are Changing')
    st.image('socialmedia.jpeg')
    st.text('As society continues to fight for social justice, what people can and can\'t say is \never changing. Some phrases that used to be considered "normal" just a few years ago\nare now off-limits, for better or worse. One unintentional slip-up can result in a \nlost job, a ruined relationship, or a slew of hateful comments.')
    st.subheader('People Don\'t Know What\'s OK to Say')
    st.image('graphic.png')
    st.text('Because of the constantly changing standards of speech, many are left unsure of what\nis OK to say. OK2Say hopes to remedy this by using machine-learning models to scan \nuser-inputted text for potentially controversial phrases and help people get their\nreal message across to everyone.')

if selection == 'HOW IT\'S MADE':

    st.title('OK2Say')
    st.subheader('Technologies Used')
    st.image('REALTFKERAS.png')
    st.text('The website for OK2Say was built with Streamlit, a framework used to create web apps\nin Python. The model for determining the controversy of sentences was created with \nTensorFlow and Keras and was trained on a dataset of toxic Tweets.')
