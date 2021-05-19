"""Create an Image Classification Web App using PyTorch and Streamlit."""
#import libraries

import keras
from PIL import Image, ImageOps
import torch
import streamlit as st
import numpy as np
import tensorflow as tf

# set title of app
st.title("Image Classification Application")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")


def predict(img):
    """Return top 5 predictions ranked by highest probability.
    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: top 5 predictions ranked by highest probability
    """
    # create a ResNet model
    model = keras.models.load_model('model_plant.h5')

    # Create the array of the right shape to feed into the keras model
    #data = np.ndarray(shape=(256, 256, 3), dtype=np.float32)
    # image sizing
    size = (256, 256)
    #image1 = tf.keras.preprocessing.image.load_img(img)
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    preds1 = model.predict_classes(input_arr)

    # run the inference
    #prediction = model.predict()
    return preds1  # return position of the highest probability


if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Plant Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(image)
    if label == 0:
        st.write('Pepper_Bacterial')
    if label == 1:
        st.write('Pepper_healthy')
    if label == 2:
        st.write('Potato_Early_blight')
    if label == 3:
        st.write('Potato_healthy')
    if label == 4:
        st.write('Potato_Late_blight')
    if label == 5:
        st.write('Tomato_Bacterial')
    if label == 6:
        st.write('Tomato_Curl_Virus')
    if label == 7:
        st.write('Tomato_Early_blight')
    if label == 8:
        st.write('Tomato_healthy')
    if label == 9:
        st.write('Tomato_Late_blight')
    if label == 10:
        st.write('Tomato_Leaf_Mold')
    if label == 11:
        st.write('Tomato_mosaic')
    if label == 12:
        st.write('Tomato_Septoria_leaf_spot')
    if label == 13:
        st.write('Tomato_spider_mite')
    if label == 14:
        st.write('Tomato_Target_Spot')

