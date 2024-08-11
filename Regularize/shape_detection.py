import cv2
import numpy as np
import os

print("Current working directory:", os.getcwd())

def preprocess_image(image):
    """Preprocess the image by converting to binary and applying morphological operations."""
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Apply GaussianBlur to smooth the image and reduce noise
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)
    
    # Apply morphological operations to clean up noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_OPEN, kernel)
    
    return cleaned_image

def fit_circle_to_contour(contour):
    """Fit a circle to a contour."""
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return (int(x), int(y), int(radius))

def fit_ellipse_to_contour(contour):
    """Fit an ellipse to a contour."""
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        return ellipse
    return None

def draw_regularized_shape(image, shape_type, params):
    """Draw regularized shapes on the image."""
    output_image = image.copy()  # Copy the original image to draw on
    
    if shape_type == 'circle':
        x, y, radius = params
        cv2.circle(output_image, (x, y), radius, (0, 255, 0), 2)
        cv2.putText(output_image, 'Circle', (x - 50, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    elif shape_type == 'ellipse':
        center, axes, angle = params
        axes = (int(axes[0]), int(axes[1]))
        cv2.ellipse(output_image, (int(center[0]), int(center[1])), axes, angle, 0, 360, (0, 255, 0), 2)
        cv2.putText(output_image, 'Ellipse', (int(center[0]) - 50, int(center[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image

def regularize_shapes(image):
    """Regularize shapes in an image."""
    preprocessed_image = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Initialize output image as a color image
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < 100:  # Adjust this threshold as needed
            continue
        
        # Attempt to fit a circle
        (x, y, radius) = fit_circle_to_contour(contour)
        circle_area = np.pi * radius ** 2
        contour_area = cv2.contourArea(contour)
        
        # Check if the contour is more like a circle or an ellipse
        if abs(circle_area - contour_area) / contour_area < 0.1:  # Adjust this threshold as needed
            shape_type = 'circle'
            params = (x, y, radius)
        else:
            ellipse_params = fit_ellipse_to_contour(contour)
            if ellipse_params:
                shape_type = 'ellipse'
                params = (ellipse_params[0], ellipse_params[1], ellipse_params[2])
            else:
                continue  # Skip if neither circle nor ellipse fits well
        
        # Draw the regularized shape
        output_image = draw_regularized_shape(output_image, shape_type, params)
    
    return output_image

# Main script
if __name__ == "__main__":
    # Load and preprocess the image
    image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image. Check the file path.")
        exit()
    
    # Regularize the shapes
    output_image = regularize_shapes(image)
    
    # Save the output image
    if output_image is not None and np.any(output_image):
        cv2.imwrite('regularized_image.png', output_image)
        print("Output image saved as 'regularized_image.png'")
    else:
        print("No shapes were detected or drawn.")
