import os
import pickle

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

trainData_path = "Data/train.p"
testData_path = "Data/test.p"

model_save_path = "./Models/"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

(X_train, y_train) = pickle.load(open(trainData_path, "rb"))
(X_test, y_test) = pickle.load(open(testData_path, "rb"))

X_train = X_train / 255.0
X_test = X_test / 255.0

classifier = Sequential()
classifier.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1",
                      input_shape=(32, 32, 3)))
classifier.add(MaxPooling2D(pool_size=(2, 2), name="pool_1"))
classifier.add(BatchNormalization(name="bn_1"))
classifier.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2"))
classifier.add(MaxPooling2D(pool_size=(2, 2), name="pool_2"))
classifier.add(BatchNormalization(name="bn_2"))
classifier.add(Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same", name="conv_3"))
classifier.add(Dropout(0.5, name="drop_3"))
classifier.add(Flatten(name="flat"))
classifier.add(Dense(64, activation='relu', name="fc_1"))
classifier.add(Dense(10, name="fc_f"))

classifier.summary()

lr_schedule = schedules.ExponentialDecay(
              initial_learning_rate=1e-3,
              decay_steps=10000,
              decay_rate=0.9)


classifier.compile(optimizer=Adam(learning_rate=lr_schedule),
                   loss=SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

history = classifier.fit(X_train, y_train, epochs=2,
                         validation_data=(X_test, y_test))

test_loss, test_acc = classifier.evaluate(X_test,  y_test, verbose=2)

print("Test loss:     ", test_loss)
print("Test accuracy: ", test_acc)

classifier_path = "./Models/classifier"
classifier.save(classifier_path)

classifier_encoder = Model(classifier.input, classifier.get_layer("conv_3").output)
classifier_encoder_path = "./Models/classifier_encoder"
classifier_encoder.save(classifier_encoder_path)
