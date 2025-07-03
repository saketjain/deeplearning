from tensorflow.keras.layers import Input, Reshape, Dense, Flatten, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
import numpy as np

def create_model(height, width, depth, filters, latent_size):
    
    inputs = Input(shape=(height, width, depth))
    x = inputs
    for f in filters:
        x = Conv2D(f, (3,3), padding='same', activation='relu')(x)
    
    flatten=int_shape(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(latent_size, activation='relu')(x)
    
    encoder = Model(inputs=inputs, outputs=encoded, name='encoder')
    
    decoded_inputs = Input(shape=(latent_size,))
    x = Dense(64, activation='relu')(decoded_inputs)
    x = Dense(np.product(flatten[1:]))(x)
    x = Reshape(flatten[1:])(x)
    
    for f in filters[::-1]:
        x = Conv2DTranspose(f, (3,3), padding='same', activation='relu')(x)
    
    decoded = Conv2DTranspose(depth, (3,3), padding='same', activation='sigmoid')(x)
    
    decoder = Model(inputs=decoded_inputs, outputs=decoded, name='decoder')
    
    autoencoder = Model(inputs=inputs, outputs=decoder(encoder(inputs)), name='autoencoder')
    
    return (encoded, decoder, autoencoder)

(encoder, decoder, autoencoder) = create_model(28, 28, 3, (16, 32), 16)

print(autoencoder.summary())