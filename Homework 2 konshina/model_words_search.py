
# импортируем библиотеки
import io
import streamlit as st
from itertools import groupby
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Загружаем модель
model_name = "0x7194633/keyt5-large" # or 0x7194633/keyt5-base
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# обучение модели
@st.cache_resource
def generate(text, **kwargs):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        hypotheses = model.generate(**inputs, num_beams=5, **kwargs)
    s = tokenizer.decode(hypotheses[0], skip_special_tokens=True)
    s = s.replace('; ', ';').replace(' ;', ';').lower().split(';')[:-1]
    s = [el for el, _ in groupby(s)]
    return s
st.title('Поиск ключевых слов в тексте')# Заголовок в streamlit
# описание приложения
st.write('Это приложение можно использовать для поиска ключевых слов в тексте.')
# ввод текста в приложении
context = st.text_input('Введите текст:')
# кнопка вывод результата
result = st.button('найти ключевые слова')
# Eсли кнопка нажата запуск модели и вывод результата.
if result:
    preds = (generate(context, top_p=1.0, max_length=64))
    st.write("Ответ:", *preds)


