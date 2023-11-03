#Модель определения эмоционального окраса сообщений
# 0-нейтрально
# 1-позитивно
# 2-негативно


import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment', return_dict=True)

@torch.no_grad()

def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted

print(predict("Ужасное обслуживание"))
print(predict("Люблю море"))
print(predict("Здание школы расположено по адресу г.Москва ул.Садовая 6"))