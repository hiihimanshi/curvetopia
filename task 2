import cv2
import numpy as np
from sklearn.metrics import mean_squared_error

def preprocess_image(image):
    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Use adaptive thresholding to handle varying illumination
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
        print(f"Error: Reflection failed or dimensions mismatch for axis {axis}")
        return False
    
    # Calculate the Mean Squared Error (MSE) between the original and reflected images
    mse = mean_squared_error(image.flatten(), reflected_image.flatten())
    return mse < mse_threshold

def main(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Preprocess the image for better symmetry detection
    processed_image = preprocess_image(image)
    
    axes = ['vertical', 'horizontal', 'main_diagonal', 'anti_diagonal']
    symmetries = {axis: check_symmetry(processed_image, axis) for axis in axes}
    
    for axis, symmetric in symmetries.items():
        print(f"Symmetry along {axis.replace('_', ' ')} axis: {'Yes' if symmetric else 'No'}")
    
    # Display the processed image
    cv2.imshow("Processed Image", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'symmetric\\8.png'  # Replace with your image path
main(image_path)
