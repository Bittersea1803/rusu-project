import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image as imgur
from keras.utils.image_utils import img_to_array
from keras.models import Model
from keras.applications import imagenet_utils
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

img = Image.Image()
a = ''

def model_VGG16_raw():
    global img
    model = keras.models.load_model('models/vgg_16_raw')
    
    my_image = Image.Image()
    my_image = img

    #preprocess the image
    my_resized_image = my_image.resize((224, 224))
    
    my_resized_image_np = np.array(my_resized_image)
    #my_image = preprocess_input(my_image)

    #make the prediction
    prediction = model.predict(my_image)
    st.write(np.argmax(prediction))
    

def load_image(image_file):
    global img
    img = Image.open(image_file)
    

def clasify():
    global img
    global a
    if a == 'VGG16_raw':
        model_VGG16_raw()
    if a == 'VGG16':
        model_VGG16()
    if a == 'ResNet':
        model_ResNet()

st.title('X-ray classifier')
st.subheader("Image")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
a = st.selectbox('Which classifier you want to use?', ('VGG16_raw', 'VGG16', 'ResNet'))
st.button('Clasify!', on_click=clasify)
