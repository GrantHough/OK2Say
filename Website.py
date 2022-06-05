    
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

selection = st_btn_select(('HOME', 'ABOUT', 'HOW IT\'S MADE'))

userTextInput = ""

if selection == 'HOME':

    st.title('OK2Say')
    st.subheader('Flagging potentially controversial phrases with NLP')
    st.text('Created by Grant Hough')
    
    userTextInput = st.text_area("Provide the text you would like scanned:", userTextInput)
    sentences = nltk.sent_tokenize(userTextInput)

    outputList = [2][len(sentences)]

    counter = 0

    for sentence in sentences: 
        counter = counter + 1
        number = round(model.predict(prepareData(sentence, tokenizer))[0][1] * 100, 2)
        outputList[0][counter - 1].append('"' + sentence + '"')
        outputList[1][counter - 1].append(number)

    df = pd.DataFrame()
    df["Sentence"], df["Chance of being Controversial"] = outputList.T
    
   
    if(len(sentences) > 0):

        st.table(df)

if selection == 'ABOUT':

    st.title('OK2Say')
    st.subheader('Times are Changing')
    st.image('socialmedia.jpeg')
    st.text('As society continues to fight for social justice, what people can and can\'t say is \neverchanging. Some phrases that used to be considered "normal" just a few years ago\nare now off-limits, for better or worse. One unintentional slip-up can result in a \nlost job, a ruined relationship, or a slew of hateful comments.')
    st.subheader('People Don\'t Know What\'s OK to Say')
    st.image('rightwrong.webp')
    st.text('Because of the constantly changing standards of speech, many are left unsure of what\n is OK to say. OK2Say hopes to remedy this by using machine-learning to scan \nuser-inputted text for potentially controversial phrases.')
    
if selection == 'HOW IT\'S MADE':

    st.title('OK2Say')
