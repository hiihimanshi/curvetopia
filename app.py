from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)


# Directory to save processed images temporarily
TEMP_DIR = 'temp_images'
os.makedirs(TEMP_DIR, exist_ok=True)

def preprocess_image(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
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
    mse = mean_squared_error(image.flatten(), reflected_image.flatten())
    return mse < mse_threshold

def detect_edges(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
    return edges

def morphological_closing(edges, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed_image

def find_and_draw_largest_contour(closed_image, original_image):
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(original_image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_image, [largest_contour], -1, 255, thickness=2)
    return contour_image

def smooth_contour(contour_image):
    smoothed_image = cv2.GaussianBlur(contour_image, (3, 3), 0)
    return smoothed_image

@app.route('/process_image', methods=['POST'])
def process_image():
    feature = request.form.get('feature')
    file = request.files['file']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    # Load image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    
    if feature == 'Symmetry':
        processed_image = preprocess_image(image)
        symmetries = {axis: check_symmetry(processed_image, axis) for axis in ['vertical', 'horizontal', 'main_diagonal', 'anti_diagonal']}
        result = {axis: 'Yes' if symmetric else 'No' for axis, symmetric in symmetries.items()}
        return jsonify(result)
    
    elif feature == 'Regularized':
        edges = detect_edges(image)
        closed_image = morphological_closing(edges)
        contour_image = find_and_draw_largest_contour(closed_image, image)
        final_image = smooth_contour(contour_image)
        
        output_image_path = os.path.join(TEMP_DIR, 'regularized.png')
        cv2.imwrite(output_image_path, final_image)
        return send_file(output_image_path, mimetype='image/png')
    
    elif feature == 'Plannerized':
        # Implement Plannerized logic here (not provided in your original code)
        return jsonify({"message": "Plannerized feature is not yet implemented"})
    
    else:
        return jsonify({"error": "Unknown feature"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
