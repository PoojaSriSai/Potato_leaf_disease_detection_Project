import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template, send_from_directory
import uuid
import shutil

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model_path = 'potato.h5'  # Update with your model's path
model = load_model(model_path,compile=False)

# Set up the image upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define a function to make predictions
def make_prediction(image_path):
    target_size = (256, 256)  # Ensure this matches the model's expected input shape
    img = image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Predict and get class name and confidence
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = round(100 * np.max(predictions[0]), 2)

    return predicted_class, confidence

# Route for the home page to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')  # This file will contain your HTML form

# Route for handling the image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Save the uploaded image to the upload folder
    file = request.files['file']
    filename = str(uuid.uuid4()) + '.jpg'  # Unique filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    # Make the prediction
    predicted_class, confidence = make_prediction(image_path)

    # Class names (adjust these based on your model)
    class_names = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']

    # Display the prediction result
    result = {
        'class_name': class_names[predicted_class],
        'confidence': confidence
    }

    return render_template('result.html', result=result, filename=filename)

# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)
