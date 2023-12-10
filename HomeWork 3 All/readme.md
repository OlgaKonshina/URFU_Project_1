Домашнее задание №3

Модель определения эмоционального окраса текста

Используемые модули: (см. requirements.txt)

Скрипт для создания приложения с моделью:
 - Api.py

Команда для запуска сервера Uvicorn: uvicorn Api:app

Команда для формирования запроса с текстом для определения эмоционального окраса:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "It was a wonderful journey. We have visited many beautiful places and seen many sightings! I am happy!"
}'

Пример ответа:
{"Исходный текст":"It was a wonderful journey. We have visited many beautiful places and seen many sightings! I am happy!","Результат:":"позитивный :)"}

Участники команды:
 - Коньшина Ольга
 - Ильиных Виктория
 - Шабанов Дмитрий
 - Воробьев Василий
