from keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from image.auto_encoder.plain.model import create_model
import numpy as np


def train_model():

    (x_train, _), (x_test, _) = mnist.load_data()
    (len, height, width) = x_train.shape
    x_train = np.reshape(x_train, (len, height, width, 1))
    x_train = x_train/255.0

    (len, height, width) = x_test.shape
    x_test = np.reshape(x_test, (len, height, width, 1))
    x_test = x_test/255.0

    (encoder, decoder, auto_encoder) = create_model(height, width, 1, (16, 32), 16)
    auto_encoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    H = auto_encoder.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=2, batch_size=256, callbacks=[TensorBoard(log_dir='./tmp/autoencoder')])

    auto_encoder.save('autoencoder.keras')

train_model()  
