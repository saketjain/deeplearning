from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Input, MaxPooling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np

def create_model(height, width, depth, filters, latent_dim):
    
    inputs = Input(shape=(height, width, depth))
    x = inputs
    for f in filters:
        x = Conv2D(f, (3,3), padding='same', activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2,2))(x)
    
    volume_size = K.int_shape(x)
    x = Flatten()(x)
    latent = Dense(latent_dim)(x)
    
    encoder = Model(inputs=inputs, outputs=latent, name='encoder')
    
    latent_inputs = Input(shape=(latent_dim, ))
    x = Dense(np.product(volume_size[1:]))(latent_inputs)
    x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
    
    for f in filters[::-1]:
        x = Conv2DTranspose(f, (3,3), padding='same', activation='relu')(x)
        #x = MaxPooling2D(pool_size=(2,2))(x)
    
    output = Conv2DTranspose(depth, (3,3), padding='same', activation='sigmoid')(x)
    
    decoder = Model(inputs=latent_inputs, outputs=output, name='decoder')
    
    auto_encoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')
    
    return (encoder, decoder, auto_encoder)

'''
(encoder, decoder, auto_encoder) = create_model(128, 128, 3, (16, 32), 26)
encoder.summary()
decoder.summary()
auto_encoder.summary()
'''