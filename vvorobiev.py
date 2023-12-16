# модель ищет ответ на вопрос исходя из заданного текста

# !pip install transformers

from transformers import pipeline

question = 'Когда будем убирать лёд?'
context = ('Вчера выпало много снега, а сегодня из-за'
         ' тёплой погоды он подтаял и заледенел.'
         ' Придётся на выходных взять лом и немного разбить лёд.')

model_pipeline = pipeline(
   task='question-answering',
   model='timpal0l/mdeberta-v3-base-squad2'
)

model_pipeline(question=question, context=context)
