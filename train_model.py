import tensorflow as tf
from tensorflow.keras import layers, models
from prepare_data import load_and_preprocess_data  # Импорт функции из предыдущего файла

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Выходной слой с сигмоидальной функцией активации для бинарной классификации

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    model = build_model()

    train_data_dir = "D:\dataset\dataset"
    train_images, train_labels = load_and_preprocess_data(train_data_dir)

    model.fit(train_images, train_labels, epochs=12, validation_split=0.2)
    model.save("adrenaline_model.h5")
