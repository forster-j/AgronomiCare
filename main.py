import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify, lookup_value, set_background

# set title
st.markdown("<h1 style='text-align: center; color: black;'>Agronomicare</h1>", unsafe_allow_html=True)

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

# center content of sidebar

st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)

with st.sidebar:
    logo = 'images/logo_bg.png'    
    st.sidebar.image(logo)
    "# How to:"

# Display the first image in the sidebar
image1 = 'images/good_1.png'
st.sidebar.image(image1, caption= 'uniform background, good focus')
#Second example image
image2 = 'images/bad_1.png'
st.sidebar.image(image2, caption= 'Please do not capture the whole forest')
# Third example image 
image3 = 'images/bad_2.png'
st.sidebar.image(image3, caption= 'Please have the whole leaf in focus')

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

st.header("Please upload an image:")
file = st.file_uploader(label="Choose Photo:", type=["jpg", "jpeg", "png"])

# load model
model = load_model('models/model_filtered.h5')

if file is not None:
    # Preprocess the uploaded image
    input_image = Image.open(file).convert('RGB')

    # Classify image
    class_name, conf_score = classify(input_image, model, classes)


    # check if class is 'healthy'
    if class_name == 'healthy':

        st.header("Congratulations, your crop seems to be healthy!")
        st.write("There is no need for any remedy or pesticide.")
        st.markdown("Happy farming 🚜")
        st.divider()
        st.markdown(disclaimer)

    else:
        ### Disease name
        # Get disease name
        disease_name = lookup_value(class_name, 'disease_name')
        # Display diesease name
        st.header(f"This crop suffers from {disease_name}.")

        # Get disease type -- optional since this information is also included in the disease description
        #disease_type = lookup_value(class_name, 'disease_type')
        # Display diesease name and type
        #st.header(f"This crop suffers from {disease_name}, a {disease_type}")

        ### Disease description
        # Get disease description
        disease_description = lookup_value(class_name, 'disease_description')
        # Calculate height of text area based on the length of the text
        num_lines = max(disease_description.count('\n') + 1, 3)  # Adjust minimum height to 3 lines
        height = num_lines * 40  # 40 pixels per line
        # Display text
        st.text_area(" ", value=disease_description, height=height)

        ### Natural remedy
        # Get natural remedy
        natural_remedy = lookup_value(class_name, 'natural_remedies')
        # Calculate height of text area based on the length of the text
        num_lines = max(natural_remedy.count('\n') + 1, 3)  # Adjust minimum height to 3 lines
        height = num_lines * 30  # 30 pixels per line
        # Display text
        st.header("Suggested Natural Remedies:")
        st.text_area(" ", value=natural_remedy, height=height)

        ### Chemical treatment
        # Get chemical treatment
        chemical_treatment = lookup_value(class_name, 'chemical_control')
        # Calculate height of text area based on the length of the text
        num_lines = max(chemical_treatment.count('\n') + 1, 3)  # Adjust minimum height to 3 lines
        height = num_lines * 25  # 30 pixels per line
        # Display text
        st.header("Suggested Chemical Treatments:")
        st.text_area(" ", value=chemical_treatment, height=height)

        # Print disclaimer
        st.divider()
        st.markdown(disclaimer)
 

## Just for debugging
# Display predictions
# st.write("## {}".format(class_name))
# st.write("### score: {}%".format(int(conf_score * 1000) / 10))        
#st.write(f"Disease name: {disease_name}")
#st.write(f"Disease type: {disease_type}")