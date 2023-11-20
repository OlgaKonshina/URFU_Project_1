# импортируем библиотеки
import io
import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel
tokenizer = AutoTokenizer.from_pretrained("dumperize/movie-picture-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("dumperize/movie-picture-captioning")
model = VisionEncoderDecoderModel.from_pretrained("dumperize/movie-picture-captioning")
def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите изображение')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None
# обучение модели
@st.cache_data
max_length = 128
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_path = 'path/to/image.jpg';
image = Image.open(image_path)
image = image.resize([224,224])
if image.mode != "RGB":
  image = image.convert(mode="RGB")

pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

output_ids = model.generate(pixel_values, **gen_kwargs)

preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# Загружаем предварительно обученную модель
model = load_model()
# Выводим заголовок страницы
st.title('Создание описания к изображению')
# Выводим форму загрузки изображения и получаем изображение
img = load_image()
# Показывам кнопку для запуска распознавания изображения
result = st.button('Распознать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result:
    # Предварительная обработка изображения
    x = preprocess_image(img)
    # Распознавание изображения
    preds = model.predict(x)
    # Выводим заголовок результатов распознавания жирным шрифтом
    # используя форматирование Markdown
    st.write('**Описание к изображению:**')
    # Выводим результаты распознавания
    print_predictions(preds)