from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Загрузка изображения и изменение размера
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Измените размер на тот, который использовался при обучении
    img_array = np.array(img) / 255.0  # Нормализация значений пикселей
    img_array = np.expand_dims(img_array, axis=0)  # Добавление дополнительной размерности для батча
    return img_array

def predict_kremlin(image_path, model):
    # Предобработка изображения
    processed_image = preprocess_image(image_path)

    # Предсказание
    prediction = model.predict(processed_image)

    # Преобразование вероятности в бинарную метку (0 или 1)
    predicted_label = 1 if prediction > 0.5 else 0

    return predicted_label

if __name__ == "__main__":
    # Путь к вашей сохраненной модели
    model_path = "kremlin_model.h5"

    # Путь к изображению для тестирования
    image_path = "D:\datasettest\data.jpg"

    # Загрузка модели
    model = load_model(model_path)

    # Предсказание
    predicted_label = predict_kremlin(image_path, model)

    # Вывод результата
    if predicted_label == 1:
        print("На фотографии есть Кремль!")
    else:
        print("На фотографии нет Кремля.")
