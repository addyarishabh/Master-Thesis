import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your trained CNN model
model = tf.keras.models.load_model('finalized.h5')

# Function to preprocess image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert("RGB")  # Convert image to RGB format to ensure it has 3 channels
    img = img.resize((256, 256))  # Resize image to match model's expected sizing
    img_array = np.array(img)  # Convert image to numpy array
    # Expand dimensions to match the input shape of the model
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array

# Streamlit app
st.title('Deep Learning based diabetic retinopathy detection model')

# HTML code to set the background color to black and text color to white
html_code = """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .healthy {
        color: green;
    }
    .detected {
        color: red;
    }
    </style>
"""
st.write(html_code, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Probability threshold slider
threshold = st.slider("Probability Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

if uploaded_file is not None:
    # Preprocess uploaded image
    image = preprocess_image(uploaded_file)
    
    # Make prediction
    prediction = model.predict(image)
    prediction_probability = prediction[0][0]  # Extracting the probability value
    
    # Display prediction
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Prediction Probability:", prediction_probability)
    if prediction_probability > threshold:
        #st.write('Diabetic Retinopathy Detected', unsafe_allow_html=True)
        st.markdown('<p class="detected">Diabetic Retinopathy Detected</p>', unsafe_allow_html=True)
    else:
        #st.write('Healthy', unsafe_allow_html=True)
        st.markdown('<p class="healthy">Healthy</p>', unsafe_allow_html=True)

    # Confidence meter
    st.write("Confidence Level:")
    st.progress(float(prediction_probability))
