from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Lambda, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow.keras.backend as K

def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def create_conv_net(shape):
    
    inputs = Input(shape=shape)
    x = Conv2D(64, (2,2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(64, (2,2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.3)(x)
        
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(48)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def create_siamese_network(input_shape):
    
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    
    feature_extractor = create_conv_net(input_shape)
    emb_a = feature_extractor(input_a)
    emb_b = feature_extractor(input_b)
    
    distance = Lambda(euclidean_distance)([emb_a, emb_b])
    
    outputs = Dense(1, activation='sigmoid')(distance)
    
    model = Model(inputs=[input_a, input_b], outputs=outputs)
    return model

'''
model = create_siamese_network((28, 28, 1))
print(model.summary())
'''