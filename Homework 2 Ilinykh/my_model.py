import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
#загружаем модель
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy = False)
model.cuda();
model.eval();
#обучение модели 
@st.cache_resource
def paraphrase(text, beams=5, grams=4, do_sample=False):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size, do_sample=do_sample)
    return tokenizer.decode(out[0], skip_special_tokens=True)
st.title('Перефраз')# Заголовок в streamlit
# описание приложения
st.write('Это приложение можно использовать для перефразирования текста')
# ввод текста в приложении
context = st.text_input('Введите текст:')
# кнопка вывод результата
result = st.button('Перефразировать')
# Eсли кнопка нажата запуск модели и вывод результата.
if result:
    sor = (paraphrase(context, top_p=1.0, max_length=256))
    st.write("Новый текст:", *sor)

model_inputs = tokenizer(src_text, return_tensors="pt")
with tokenizer.as_target_tokenizer():
    labels = tokenizer(tgt_text, return_tensors="pt").input_ids

model(**model_inputs, labels=labels) # forward pass
