import os
import numpy as np
import glob
import cv2
import pickle


def get(path):
    x = []
    y = []
    for folder in os.listdir(path):
        print('Processing ' + folder)
        for image in os.listdir(os.path.join(path, folder)):
            x.append(os.path.join(path, folder, image))
            y.append(int(folder))
    return np.array(x), np.array(y)


def get_test(path):
    x = []
    for image in os.listdir(path):
        x.append(os.path.join(path, image))
    return np.array(x)


def dump(data, name):
    pickle.dump(data, open(name + '.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


x_train, y_train = get('TrainVal')
x_test = get_test('Public')
dump((x_train, y_train, x_test), 'data')
print(x_train[:10])
print(y_train[:10])
print(x_test[:10])
