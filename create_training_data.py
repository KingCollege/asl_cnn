import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
import cv2
import random
import time

DATA_DIR = './asl_alphabet_train/asl_alphabet_train/'
CLASS_DIR = './asl_alphabet_train/class.csv'


def get_categories():
    categories = []
    with open(CLASS_DIR) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            categories.append(row[0])

    print(categories)
    return categories


categories = get_categories()


# Simplify image further to reduce noise
def process_img(img):
    kernel_size = (5, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    _open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    _open = cv2.Canny(_open, 0, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    close = cv2.Canny(close, 0, 100)

    processed = cv2.bitwise_and(_open, close)

    # ret, thresh = cv2.threshold(processed, 0, 255, 0)
    #
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
    #                                        cv2.CHAIN_APPROX_SIMPLE)
    #
    # cv2.drawContours(img, contours, -1, (0, 255, 0))
    # large_contour = max(contours, key=cv2.contourArea)
    # x,y,w,h = cv2.boundingRect(large_contour)
    # cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0))

    # plt.imshow(img)
    # plt.show()

    return processed


def create_training_data():
    TRAINING_DATA = []
    IMG_SIZE = 100
    for category in categories[:]:
        path = os.path.join(DATA_DIR, category)
        classification = categories.index(category)
        for img in tqdm(os.listdir(path)[:]):
            try:
                img_reader = cv2.imread(os.path.join(path, img))
                processed = process_img(img_reader)
                processed = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))

                # fig = plt.figure(2)
                # fig.add_subplot(1, 2, 1)
                # plt.imshow(img_reader)
                # fig.add_subplot(1, 2, 2)
                # plt.imshow(processed, cmap='gray')
                # plt.show()
                # plt.pause(0.1)
                # plt.close()
                TRAINING_DATA.append([processed, classification])
            except Exception as e:
                print(e)
        #     break
        # break

    X = []
    y = []
    random.shuffle(TRAINING_DATA)
    for features, label in TRAINING_DATA:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape((-1, IMG_SIZE, IMG_SIZE, 1))
    return X, y


X, y = create_training_data()

# np.save("./asl_alphabet_train/X.npy", X)
# np.save("./asl_alphabet_train/y.npy", y)

pickle_out = open('./asl_alphabet_train/X.pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('./asl_alphabet_train/y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()
