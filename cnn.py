import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import pickle
import csv
import cv2

import matplotlib.pyplot as plt

tf.config.list_physical_devices('GPU')


img_size = 100
MODEL_NAME = 'asl_cnn'
CATEGORIES = []


with open('./asl_alphabet_train/class.csv') as file:
    csv_reader = csv.reader(file, delimiter=',')
    for row in csv_reader:
        CATEGORIES.append(row[0])
print(CATEGORIES)


def process_img(img):
    kernel_size = (5, 5)

    # kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, kernel_size)
    # processed = cv2.dilate(img, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    _open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    _open = cv2.Canny(_open, 0, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    close = cv2.Canny(close, 0, 100)

    processed = cv2.bitwise_and(_open, close)
    processed = cv2.resize(processed, (img_size, img_size), interpolation=cv2.INTER_AREA)
    #
    # plt.imshow(cv2.resize(img, (img_size, img_size)), cmap='gray')
    # plt.show()

    return processed


def prepare(x):

    try:
        img_reader = cv2.imread('./asl_alphabet_train/test_04.jpg')
        processed = process_img(img_reader)
        processed = processed / 255.0
        plt.imshow(processed, cmap='gray')
        plt.show()
        test_img = np.array(processed).reshape((-1, img_size, img_size, 1))

        return test_img
    except Exception as e:
        print(e)
        return []


def create_model():
    pickle_in = open('./asl_alphabet_train/X.pickle', 'rb')
    X = pickle.load(pickle_in)

    pickle_in = open('./asl_alphabet_train/y.pickle', 'rb')
    y = pickle.load(pickle_in)

    X = np.array(X)
    y = np.array(y)
    X = X / 255.0

    model = Sequential()
    # Conv layer
    model.add(Conv2D(32, (5, 5), input_shape=X.shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Dense Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten Layer
    model.add(Flatten())
    model.add(Dropout(0.3))

    # Dense Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))

    # Output Layer
    model.add(Dense(len(CATEGORIES), activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=3, validation_split=0.3)
    model.summary()
    model.save(MODEL_NAME)
    return model


def get_model():
    model_found = False
    try:
        model = tf.keras.models.load_model(MODEL_NAME)
        return model
    except IOError as e:
        print(e)
    if model_found is False:
        return create_model()


# prepare('N/A')

model = get_model()
predictions = model.predict(prepare('N/A'))
np.set_printoptions(suppress=True)
print(predictions[0])
print(CATEGORIES[int(np.argmax(predictions[0]))])
