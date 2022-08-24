import os
import pickle

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.losses import MeanSquaredError

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

encoder = Sequential(name="encoder")
encoder.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv_1", input_shape=(32, 32, 3)))
encoder.add(MaxPooling2D(pool_size=(2, 2), name="pool_1"))
encoder.add(BatchNormalization(name="bn_1"))
encoder.add(Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", name="conv_2"))
encoder.add(MaxPooling2D(pool_size=(2, 2), name="pool_2"))
encoder.add(BatchNormalization(name="bn_2"))
encoder.add(Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same", name="conv_3"))

decoder = Sequential(name="decoder")
decoder.add(Conv2DTranspose(16, kernel_size=(3, 3), activation="relu", padding="same",
                            name="deconv_1", input_shape=(8, 8, 16)))
decoder.add(Conv2DTranspose(16, kernel_size=(3, 3), padding="same", activation="relu", strides=(2, 2),
                            name="deconv_2"))
decoder.add(Conv2DTranspose(32, kernel_size=(3, 3), padding="same", activation="relu", strides=(2, 2),
                            name="deconv_3"))
decoder.add(Conv2D(3, kernel_size=(3, 3), padding="same", activation="sigmoid",
                   name="out"))

AE = Sequential(name="AE")
AE.add(encoder)
AE.add(decoder)

AE.summary()

lr_schedule = schedules.ExponentialDecay(
              initial_learning_rate=1e-3,
              decay_steps=10000,
              decay_rate=0.9)

AE.compile(optimizer=Adam(learning_rate=lr_schedule),
           loss=MeanSquaredError())

history = AE.fit(X_train, X_train, epochs=2, shuffle=True,
                 validation_data=(X_test, X_test))

AE_path = "./Models/auto_encoder"
AE.save(AE_path)
AE_encoder_path = "./Models/AE_encoder"
encoder.save(AE_encoder_path)
AE_decoder_path = "./Models/AE_decoder"
decoder.save(AE_decoder_path)
