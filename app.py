from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained CNN model
model = tf.keras.models.load_model('finalized.h5')

# Function to preprocess image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Preprocess uploaded image
        image = preprocess_image(uploaded_file)
        # Make prediction
        prediction = model.predict(image)
        prediction_probability = prediction[0][0]
        return jsonify({'prediction_probability': float(prediction_probability)})
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
