from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Input, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
import numpy as np


def create_model(height, width, depth, filters, latent_size):
    
    inputs = Input(shape=(height, width, depth))
    x = inputs
    for f in filters:
        x = Conv2D(f, (3,3), padding='same', activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2,2))(x)
    
    volume = int_shape(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    
    # Latent Space
    x = Dense(latent_size, activation= 'relu')(x)
    
    # Decoder
    x = Dense(64, activation='relu')(x)
    x = Dense(np.product(volume[1:]), activation='relu')(x)
    
    x = Reshape((volume[1:]))(x)
    for f in filters[::-1]:
        x = Conv2DTranspose(f, (3,3), padding='same', activation='relu')(x)
    
    outputs = Conv2DTranspose(depth, (3,3), padding='same', activation='sigmoid')(x)
    

    model = Model(inputs=inputs, outputs=outputs, name='autoencoder')
    
    return model



model = create_model(128, 128, 3, (16, 32), 16)
model.summary()