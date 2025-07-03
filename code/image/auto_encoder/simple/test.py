from keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from model import create_model
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

def test():

    (x_train, _), (x_test, _) = mnist.load_data()
    (len, height, width) = x_train.shape
    x_train = np.reshape(x_train, (len, height, width, 1))
    x_train = x_train/255.0

    (len, height, width) = x_test.shape
    x_test = np.reshape(x_test, (len, height, width, 1))
    x_test = x_test/255.0

    model = load_model('autoencoder.keras')
    predicted = model.predict(x_test)

    n = 10

    fig, ax = plt.subplots(2, 10, figsize=(10, 4))
    ax = ax.flatten()
    for i in range(0, n):
        ax[i].imshow(x_test[i])
        ax[i].axis('off')
        
        ax[i+n].imshow(predicted[i])
        ax[i+n].axis('off')
    plt.show()
    
test()