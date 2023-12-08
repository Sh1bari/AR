from PIL import Image
import os
import numpy as np

def load_and_preprocess_data(data_dir):
    images = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            img_path = os.path.join(data_dir, filename)
            img = Image.open(img_path)
            img = img.resize((224, 224))  # Измените размер изображения на ваш выбор
            img_array = np.array(img) / 255.0  # Нормализация значений пикселей
            label = 1 if "adrenaline" in filename else 0  # 1 для изображений с Кремлем, 0 для других

            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels)

if __name__ == "__main__":
    train_data_dir = "D:\dataset\dataset"
    train_images, train_labels = load_and_preprocess_data(train_data_dir)

    # Сохраните данные в файлы или используйте их в train_model.py
