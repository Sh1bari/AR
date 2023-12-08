from PIL import Image
import numpy as np
import os
from datetime import datetime
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

model_path = "adrenaline_model.h5"
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Получение изображения из запроса
    image = request.files.get('image')

    # Предсказание
    prediction = predict_kremlin(image)

    # Возвращение результата в формате JSON
    return jsonify({"prediction": prediction})

@app.route('/save', methods=['POST'])
def save():
    # Получение изображения из запроса
    image = request.files.get('image')

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # Создание пути для сохранения изображения
    save_path = os.path.join("dataset", f"adrenaline_{unique_id}.jpg")

    # Сохранение изображения
    image.save(save_path)

    prediction = "saved"
    # Возвращение результата в формате JSON
    return jsonify({"prediction": prediction})

@app.route('/saveFake', methods=['POST'])
def savefake():
    # Получение изображения из запроса
    image = request.files.get('image')

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # Создание пути для сохранения изображения
    save_path = os.path.join("datasetfake", f"fake_{unique_id}.jpg")

    # Сохранение изображения
    image.save(save_path)

    prediction = "saved"
    # Возвращение результата в формате JSON
    return jsonify({"prediction": prediction})

@app.route('/test', methods=['GET'])
def test():

    # Возвращение результата в формате JSON
    return "hi"

def preprocess_image(image):
    # Изменение размера изображения
    img = Image.open(image)
    img = img.resize((224, 224))  # Измените размер на тот, который использовался при обучении
    img_array = np.array(img) / 255.0  # Нормализация значений пикселей
    img_array = np.expand_dims(img_array, axis=0)  # Добавление дополнительной размерности для батча
    return img_array

def predict_kremlin(image):
    # Предобработка изображения
    processed_image = preprocess_image(image)

    # Предсказание
    prediction = model.predict(processed_image)

    # Преобразование вероятности в бинарную метку (0 или 1)
    predicted_label = 1 if prediction > 0.99 else 0

    if predicted_label == 1:
        return True
    else:
        return False

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
