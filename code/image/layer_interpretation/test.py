import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import math

base_path = './image/layer_interpretation/assets'

img = image.load_img(base_path + '/cat.jpg', target_size=(224, 244))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = VGG16(weights='imagenet', include_top=False)

layer_outputs = [layer.output for layer in model.layers]
layer_names = [layer.name for layer in model.layers]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x)


def plot_layer(layer_activation, size=28):
    
    activation = layer_activation[0]
    feature_maps = activation.shape[-1]
        
    cols = min(8, feature_maps)
    rows = math.ceil(feature_maps / cols)
    
    grid = np.zeros((rows * size, cols * size))
    
    for i in range(feature_maps):        
        row = i // cols
        col = i % cols
        
        x = activation[:, :, i].copy()
        if x.sum() != 0:
            x = resize(x, (size, size), mode='reflect', anti_aliasing=True)
            x -= (x.std() + 1e-5)
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype("uint8")
            
            grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = x
    
    return grid

def plot_all_activations(activations, layer_names):
    
    size=28
    scale = 1./size
    
    for activation in activations:
        grid = plot_layer(activation, size=28)
        rows, cols = grid.shape
        plt.figure(figsize=(cols*scale, rows*scale))
        plt.grid(False)
        plt.imshow(grid, aspect="auto", cmap='viridis')
        plt.axis('off') 
        plt.show()  
    
plot_all_activations(activations, layer_names)