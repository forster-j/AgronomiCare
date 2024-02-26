import streamlit as st
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import numpy as np
import io
import os
import tempfile
from util import classify, set_background, nat_recommendation, chem_recommendation

# set title
st.markdown("<h1 style='text-align: center; color: black;'>Agronomicare 🍃</h1>", unsafe_allow_html=True)

# set header
st.header('Welcome to Agronomicare, your one-stop crop diagnosis and remedy recommender.')

# set description
multi = '''
Using Agronomicare is really easy. To diagnose your crops, 
please take or upload a photo of a leaf of the plant according to the instructions on the left.
Agronomicare will then use the power of neural networks to tell if the crop is healthy or not.
If the crop is suffering from a disease, it will automatically suggest natural remedies and
chemical pesticides.
'''
st.markdown(multi)

# set sidebar
st.sidebar.title('How to: ')

# Display the first image in the sidebar
image1 = 'images/00a79a7b-8b96-452c-91c8-deb54eaa28e5___CREC_HLB 7231.JPG'
st.sidebar.image(image1, caption= 'uniform background, good focus', width=144)
#Second example image
image2 = 'images/171026079.jpg'
st.sidebar.image(image2, caption= 'Please do not capture the whole forest', width=144)
# Third example image 
image3 = 'images/redrot (164).jpeg'
st.sidebar.image(image3, caption= 'Please have all of the leaf in focus', width=144)

# set some style 

set_background(main_bg_color="#EBEBEB", sidebar_bg_color="#edffcc")

# load class names 

classes = [
    'alternaria_leaf_spot',
    'bacterial_blight',
    'bacterial_spot',
    'bacterial_wilt',
    'black_measles',
    'black_rot',
    'blast',
    'brown_spot',
    'brown_streak_disease',
    'citrus_greening',
    'common_rust',
    'early_blight',
    'gray_leaf_spot',
    'healthy',
    'isariopsis_leaf_spot',
    'late_blight',
    'leaf_curl',
    'leaf_mold',
    'mosaic_disease',
    'northern_leaf_blight',
    'powdery_mildew',
    'red_rot',
    'spider_mites',
    'target_spot',
    'tungro',
    ]

# Disclaimer for predictions

disclaimer = '''
*Please note that Agronomicare can make mistakes. 
It's good practice to take photos from different angles and consult with your local dealer
or botanist to get a reliable prediction.*
'''

#preprocessing function

def preprocess_image(uploaded_file, target_size=(224, 224)):
    # Create a temporary file to save the uploaded image
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())

    # Close the temporary file to ensure it's saved properly
    temp_file.close()

    # Load the image from the temporary file and resize it to the target size
    img = image.load_img(temp_file.name, target_size=target_size)

    # Delete the temporary file
    os.unlink(temp_file.name)


file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# load model
model = load_model('models/model_filtered.h5')

if file is not None:
    # Preprocess the uploaded image
    input_image = Image.open(file).convert('RGB')

    # Classify image
    class_name, conf_score = classify(input_image, model, classes)

    # Display predictions
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))

    # check if class is 'healthy'
    if class_name == 'healthy':
        st.write("Congratulations, your crop seems to be healthy!")
        st.write("There is no need for any remedy or pesticide.")
        st.markdown("Happy farming 🚜")
        st.divider()
        st.markdown(disclaimer)

    else:
        st.write(f"Your crop seems to be suffering from {class_name}")
        # get the recommendation for the natural remmedy
        recommendation_text_nat = nat_recommendation(class_name)

        # Calculate the height of the text area dynamically based on the length of the recommendation text
        num_lines = max(recommendation_text_nat.count('\n') + 1, 3)  # Adjust minimum height to 3 lines
        height = num_lines * 20  # 20 pixels per line

        # Display the recommendation
        #st.text_area("Natural Remedy Recommendation", value=recommendation_text, height=200)
        st.markdown("<h2 style='font-weight: bold;'>Natural Remedy Recommendation:</h2>", unsafe_allow_html=True)
        st.text_area(" ", value=recommendation_text_nat, height=height)

        # get the recommendation for the chemical remmedy
        recommendation_text_chem = chem_recommendation(class_name)

        # Calculate the height of the text area dynamically based on the length of the recommendation text
        num_lines = max(recommendation_text_chem.count('\n') + 1, 3)  # Adjust minimum height to 3 lines
        height = num_lines * 20  # 20 pixels per line

        # Display the recommendation
        #st.text_area("Natural Remedy Recommendation", value=recommendation_text, height=200)
        st.markdown("<h2 style='font-weight: bold;'>Chemical Remedy Recommendation:</h2>", unsafe_allow_html=True)
        st.text_area(" ", value=recommendation_text_chem, height=height)

        # Print disclaimer
        st.divider()
        st.markdown(disclaimer)
 

