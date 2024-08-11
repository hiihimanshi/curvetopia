import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def detect_edges(image):
    # Apply GaussianBlur to reduce noise and then Canny edge detection
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, threshold1=50, threshold2=150)
    return edges

def morphological_closing(edges, kernel_size=(5, 5)):
    # Apply morphological closing to connect broken segments
    kernel = np.ones(kernel_size, np.uint8)
    closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return closed_image

def find_and_draw_largest_contour(closed_image, original_image):
    # Find contours from the closed image
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank image to draw the largest contour
    contour_image = np.zeros_like(original_image)
    
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw only the largest contour
        cv2.drawContours(contour_image, [largest_contour], -1, 255, thickness=2)
    
    return contour_image

def smooth_contour(contour_image):
    # Apply Gaussian blur to smooth the final contour
    smoothed_image = cv2.GaussianBlur(contour_image, (3, 3), 0)
    return smoothed_image

def main(image_path, output_image_path):
    image = load_image(image_path)

    # Step 1: Detect edges using Canny
    edges = detect_edges(image)

    # Step 2: Apply morphological closing to close gaps and connect segments
    closed_image = morphological_closing(edges)

    # Step 3: Find and draw only the largest contour
    contour_image = find_and_draw_largest_contour(closed_image, image)

    # Step 4: Smooth the final contour to ensure it is clean and continuous
    final_image = smooth_contour(contour_image)

    # Save and display the output image
    cv2.imwrite(output_image_path, final_image)

    # Display the final result without any title
    plt.imshow(final_image, cmap='gray')
    plt.axis('off')  # Remove axes for a cleaner display
    plt.show()

# Example usage
image_path = 'occ3.png'  # Replace with your input image path
output_image_path = 'combined_apple_outline.png'  # Desired output image path
main(image_path, output_image_path)
