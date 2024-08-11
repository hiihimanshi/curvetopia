import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify, send_file
import os

app = Flask(__name__)

# Directory to save processed images temporarily
TEMP_DIR = 'temp_images'
os.makedirs(TEMP_DIR, exist_ok=True)

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Using adaptive thresholding to handle varying illumination
    binary_image = cv2.adaptiveThreshold(
        blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)
    
    return binary_image

def reflect_image(image, axis='vertical'):
    if axis == 'vertical':
        return cv2.flip(image, 1)
    elif axis == 'horizontal':
        return cv2.flip(image, 0)
    elif axis == 'main_diagonal':
        return cv2.transpose(image)
    elif axis == 'anti_diagonal':
        return cv2.flip(cv2.transpose(image), 1)

def check_symmetry(image, axis='vertical', mse_threshold=0.005):
    reflected_image = reflect_image(image, axis)
    
    if reflected_image is None or image.shape != reflected_image.shape:
        return False
    
    # Calculating the Mean Squared Error (MSE) between the original and reflected images
    mse = mean_squared_error(image.flatten(), reflected_image.flatten())
    return mse < mse_threshold

@app.route('/check_symmetry', methods=['POST'])
def check_symmetry_route():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({"error": "Could not load image"}), 400
    
    processed_image = preprocess_image(image)
    
    axes = ['vertical', 'horizontal', 'main_diagonal', 'anti_diagonal']
    symmetries = {axis: check_symmetry(processed_image, axis) for axis in axes}
    
    result = {axis.replace('_', ' '): 'Yes' if symmetric else 'No' for axis, symmetric in symmetries.items()}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
