import os

import numpy as np
import pickle

from tensorflow.keras.datasets import cifar10
from Utils.perturb_utils import brightness

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

data_save_path = "./Data/"

if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

save_path_train = data_save_path + "train.p"
save_path_test = data_save_path + "test.p"

pickle.dump((X_train, y_train), open(save_path_train, "wb"))
pickle.dump((X_test, y_test), open(save_path_test, "wb"))

X_test_bright = []
for im in X_test:
    X_test_bright.append(brightness(im, 0))
X_test_bright = np.array(X_test_bright)

save_path_perturb = "Data/test_bright.p"
pickle.dump((X_test_bright, y_test), open(save_path_perturb, "wb"))
