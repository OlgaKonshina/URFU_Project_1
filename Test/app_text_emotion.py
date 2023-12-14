#Модель определения эмоционального окраса сообщений
# 0-нейтрально
# 1-позитивно
# 2-негативно
#It was a wonderful journey. We have visited many beautiful places and seen many sightings! I am happy!

import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from fastapi import FastAPI

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)

res_text = ['Текст нейтральный', 'Текст позитивный','Текст негативный']

app = FastAPI()
@app.get("/")
def root():
    return {"message": "Hello World"}
#Model
@app.post("/predict/")
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted

# Streamlit Header
st.title('Модель определения эмоционального окраса сообщений')
st.write('Это модель определения эмоционального окраса текста. Результат: нейтрально/позитивно/негативно')
# Input text
text_in = st.text_input('Введите текст:')
# кнопка вывод результата
start = st.button('Start:')
# Click the button
if start:
    text_out = float(predict(text_in))
    #st.write("Результат:", text_out)
    st.write("Результат:", res_text[text_out])
