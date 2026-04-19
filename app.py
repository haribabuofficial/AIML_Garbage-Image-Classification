import numpy as np
import pandas as pd

import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import warnings
warnings.filterwarnings('ignore', category=UserWarning)



st.set_page_config(layout='wide')
# --------------------------------------------------------------------------------------------------------------

                                                # sidebar

with st.sidebar:
    select = option_menu("Main Menu", ['Home', 'Model Comparison', 'Predict'])



# --------------------------------------------------------------------------------------------------------------

                                                # Home

if select == "Home":
    
    st.header("RecycleVision- Garbage Image Classification Using Deep Learning")
    
    images = Image.open('./image/images_1.jfif')
    st.image(images)

    st.header("About")
    st.write("")
    st.write("""This application is an intelligent waste classification system designed to automatically identify and categorize waste materials such as plastic, metal, glass, paper, and organic. It uses deep learning and computer vision to analyze images and provide accurate results in real time. By simplifying the waste sorting process, it reduces the need for manual effort and minimizes human error. The application features a clean and user-friendly interface, making it accessible to a wide range of users. It is built to support both individual use and large-scale waste management systems.""")
    st.write("")
    st.write("""Proper waste segregation is essential for effective recycling, yet it remains a global challenge due to inefficiency and lack of awareness. This application helps address that problem by offering a smart, automated solution that improves accuracy and speed. It can be used in smart recycling bins, municipal systems, and educational platforms to promote better waste practices. Additionally, it enables tracking of waste data, helping organizations make informed environmental decisions. Overall, the application contributes to a cleaner, more sustainable future through the use of artificial intelligence.""")


# --------------------------------------------------------------------------------------------------------------

                                                # Model Comparison

# import joblib

# X_train, y_train = joblib.load('./split_data/train_data.pkl')
# X_val, y_val = joblib.load('./split_data/val_data.pkl')
# X_test, y_test = joblib.load('./split_data/test_data.pkl')

# classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report


# if select == 'Model Comparison':
#     colm, coln = st.columns(2)

#     with colm:
#         model_1 = load_model('base_model.keras')
#         y_pred_probs = model_1.predict(X_test)
#         y_pred = y_pred_probs.argmax(axis=1)

#         st.text(classification_report(y_test, y_pred, target_names=classes))
    
#     with coln:
#         pass

# --------------------------------------------------------------------------------------------------------------

                                                # Predict


import cv2
from tensorflow.keras.models import load_model

model_3 = load_model('mobilenetv2_model.keras')

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


if select == "Predict":

    st.header("Model Prediction using MobileNetV2_model")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    with col2:
        l, c, r = st.columns(3)
        with c:
            if uploaded_file is not None:
                st.image(uploaded_file, caption="Uploaded Image", width=250)

            else:
                pass



    col3, col4 = st.columns(2)

    with col3:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)       # Convert file to OpenCV format
            img = np.array(image)

            # img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))         
            
            # fig, ax = plt.subplots()                  # show image
            # ax.imshow(img)
            # st.pyplot(fig)  

            img = img.astype('float32')             # int -> float
            img = img / 255.0                           # normalize

            # Predict button
            if st.button("Predict"):
                pred = model_3.predict(img.reshape(1, 224, 224, 3))
                pred = pd.DataFrame(pred, columns=classes)

                class_id = np.argmax(pred)

                # Output
                st.write("Prediction probabilities:",)
                st.dataframe(pred)
                st.success(f"Predicted Class: {classes[class_id]}")
                

# --------------------------------------------------------------------------------------------------------------
