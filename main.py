# import libraries
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

# set title of app
st.title("Facial Expression Regconition")
st.write("")

# enable users to upload images for the model to make predictions
file_up = st.file_uploader("Upload an image", type = "jpg")

from keras.models import load_model
model100 = load_model("model100.h5")
model30 = load_model('model30.h5')

    

if file_up is not None:
    # display image that user uploaded
    #setting image resizing parameters
    image = Image.open(file_up)
    
    st.image(image, caption = 'Uploaded Image.', use_column_width = (300,300))
    st.write("")
    temp = file_up.name.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    gender_dict = {0:'Male', 1:'Female'}
    
    img= image.convert()
    img = img.resize((128, 128), 1)

    img = np.array(img)
    img= img / 255.0
    print("imga shape",img.shape)
    
    
    
    image_gray = np.mean(img, axis=2, keepdims=True)
    
    #model 100 epoch
    pred100 = model100.predict(image_gray.reshape(1,128,128,1))
    pred_gender100 = round(pred100[0][0][0])
    pred_age100 = round(pred100[1][0][0])
    
    #model 30 epoch
    pred30 = model30.predict(image_gray.reshape(1,128,128,1))
    pred_gender30 = round(pred100[0][0][0])
    pred_age30 = round(pred100[1][0][0])
    
    st.write("Model100:")
    st.write("Predicted Gender:", gender_dict[pred_gender100], "Predicted Age:", pred_age100)
    st.write("Origin Gender:", gender_dict[gender], "Predicted Age:", age)
    st.write("Model30:")
    st.write("Predicted Gender:", gender_dict[pred_gender30], "Predicted Age:", pred_age30)
    st.write("Origin Gender:", gender_dict[gender], "Predicted Age:", age)

   
