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
    encoded = Dense(latent_size, activation= 'relu')(x)
    
    encoder = Model(inputs=inputs, outputs=encoded, name='encoder')
    print('Encoder...')
        
    latent_input = Input(shape=(latent_size,))
    x = Dense(64, activation='relu')(latent_input)
    x = Dense(np.product(volume[1:]), activation='relu')(x)
    
    x = Reshape((volume[1:]))(x)
    for f in filters[::-1]:
        x = Conv2DTranspose(f, (3,3), padding='same', activation='relu')(x)
    
    decoded = Conv2DTranspose(depth, (3,3), padding='same', activation='sigmoid')(x)
    
    decoder = Model(inputs=latent_input, outputs=decoded, name='decoder')
    
    print('Decoder...')
    auto_encoder = Model(inputs=inputs, outputs=decoder(encoder(inputs)), name='autoencoder')
    
    return (encoder, decoder, auto_encoder)


'''
(encoder, decoder, auto_encoder) = create_model(128, 128, 3, (16, 32), 16)
encoder.summary()
decoder.summary()
auto_encoder.summary()
'''
