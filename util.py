
import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import pandas as pd

import base64

def set_background(main_bg_color="#EBEBEB", sidebar_bg_color="#edffcc"):
    """
    This function sets the background of a Streamlit app to the specified colors.

    Parameters:
        main_bg_color (str): The background color for the main content area.
        sidebar_bg_color (str): The background color for the sidebar.

    Returns:
        None
    """
    # Main background style
    main_style = f"""
        <style>
        body {{
            background-color: {main_bg_color};
        }}
        </style>
    """
    st.markdown(main_style, unsafe_allow_html=True)

    # Sidebar background style
    sidebar_style = f"""
        <style>
        .css-1l02zno {{
            background-color: {sidebar_bg_color};
        }}
        </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert image to (224, 224)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Expand dimensions to match model input shape
    data = np.expand_dims(normalized_image_array, axis=0)

    # Make prediction
    predictions = model.predict(data)
    predicted_class_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_index]
    class_name = class_names[predicted_class_index]

    return class_name, confidence_score

def nat_recommendation(class_name):
    df = pd.read_csv("/Users/freyasteinhagen/neue_fische/AgronomiCare/data/pesticides_dataset.csv")
    if class_name in df['disease\n'].values:
        natural_remedy = df.loc[df['disease\n'] == class_name, 'natural_remedies'].iloc[0]
        return natural_remedy
    
def chem_recommendation(class_name):
    df = pd.read_csv("/Users/freyasteinhagen/neue_fische/AgronomiCare/data/pesticides_dataset.csv")
    if class_name in df['disease\n'].values:
        natural_remedy = df.loc[df['disease\n'] == class_name, 'chemical_control'].iloc[0]
        return natural_remedy
