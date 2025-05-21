import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load the model
model = joblib.load('model.pkl')

# Define the class names
class_names = {
    0: "Normal Person",
    1: "Patient with History of MI",
    2: "Patient with abnormal heartbeat",
    3: "Myocardial Infarction Patient"
}

# Streamlit app
st.title('ECG Analysis')

# Upload image
uploaded_file = st.file_uploader("Choose an ECG image...", type="jpg")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded ECG Image.', use_container_width=True)

    # Resize and flatten the image
    image = image.resize((64, 64))
    image_array = np.array(image).flatten()
    image_array = image_array.reshape(1, -1)  # Reshape for the model

    # Make prediction
    prediction = model.predict(image_array)[0]
    predicted_class = class_names[prediction]

    st.write(f'Prediction: {predicted_class}')
