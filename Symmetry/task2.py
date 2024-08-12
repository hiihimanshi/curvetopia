import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd

def load_curve_data(csv_file):
    """
    Load curve data from a CSV file and return a list of curve point sequences.
    """
    data = pd.read_csv(csv_file, header=None).values
    curve_paths = []
    for curve_id in np.unique(data[:, 0]):
        curve_segments = data[data[:, 0] == curve_id][:, 1:]
        segments = []
        for segment_id in np.unique(curve_segments[:, 0]):
            segment_points = curve_segments[curve_segments[:, 0] == segment_id][:, 1:]
            segments.append(segment_points)
        curve_paths.append(segments)
    return curve_paths

def generate_image_from_curves(curves, img_size=(500, 500), line_color=(255, 255, 255)):
    """
    Generate an image with the curves drawn on it using the provided list of curve data.
    """
    img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

    for curve in curves:
        for points in curve:
            if len(points) < 2:
                continue

            if len(points) == 2:
                cv2.line(img, tuple(map(int, points[0])), tuple(map(int, points[1])), line_color, 2)
                continue

            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])

            tck, u = splprep([x_coords, y_coords], s=0, k=2)
            u_new = np.linspace(u.min(), u.max(), 1000)
            x_new, y_new = splev(u_new, tck, der=0)

            for i in range(len(x_new) - 1):
                cv2.line(img, (int(x_new[i]), int(y_new[i])), (int(x_new[i+1]), int(y_new[i+1])), line_color, 2)

    return img

def find_harris_corners(img, max_corners=2, threshold_ratio=0.1):
    """
    Detect Harris corners in the image.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    corner_response = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    corner_response = cv2.dilate(corner_response, None)
    threshold = threshold_ratio * corner_response.max()

    detected_corners = np.argwhere(corner_response > threshold)
    detected_corners = [(int(c[1]), int(c[0])) for c in detected_corners]

    sorted_corners = sorted(detected_corners, key=lambda pt: corner_response[pt[1], pt[0]], reverse=True)

    return sorted_corners[:max_corners], img

def calculate_opposite_point(corner, symmetry_line_coords):
    """
    Calculate the point symmetric to the given corner across the provided symmetry line.
    """
    x1, y1, x2, y2 = symmetry_line_coords
    x, y = corner

    line_vec = np.array([x2 - x1, y2 - y1])
    pt_vec = np.array([x - x1, y - y1])
    line_length_sq = np.dot(line_vec, line_vec)
    if line_length_sq == 0:
        raise ValueError("Symmetry line has zero length!")

    proj_length = np.dot(pt_vec, line_vec) / line_length_sq
    proj_point = proj_length * line_vec
    closest_pt = np.array([x1, y1]) + proj_point

    symmetric_pt = 2 * closest_pt - np.array([x, y])

    return int(symmetric_pt[0]), int(symmetric_pt[1])

def draw_spline_on_image(img, pts, spline_color=(0, 255, 0)):
    """
    Draw a B-spline curve connecting the given points on the image.
    """
    if len(pts) < 2:
        return img

    if len(pts) == 2:
        cv2.line(img, pts[0], pts[1], spline_color, 2)
        return img

    x_coords = np.array([p[0] for p in pts])
    y_coords = np.array([p[1] for p in pts])

    tck, u = splprep([x_coords, y_coords], s=0, k=2)
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    for i in range(len(x_new) - 1):
        cv2.line(img, (int(x_new[i]), int(y_new[i])), (int(x_new[i+1]), int(y_new[i+1])), spline_color, 2)

    return img

def detect_and_draw_symmetry_line(csv_file, image_title):
    """
    Detect the symmetry line in the image and find the Harris corners and their corresponding symmetric points.
    """
    curves = load_curve_data(csv_file)
    curve_img = generate_image_from_curves(curves, img_size=(500, 500), line_color=(255, 255, 255))
    
    # Save and reload the image to ensure it gets processed correctly
    temp_img_path = 'temp_curve_image.png'
    cv2.imwrite(temp_img_path, curve_img)
    img = cv2.imread(temp_img_path)
    
    # Detect the top 15 Harris corners
    detected_corners, img_with_corners = find_harris_corners(img, max_corners=15)

    symmetry_detector = Mirror_Symmetry_detection(temp_img_path)
    matched_points = symmetry_detector.find_matchpoints()
    r_vals, theta_vals = symmetry_detector.find_points_r_theta(matched_points)

    hexbin_image = plt.hexbin(r_vals, theta_vals, bins=200, cmap=plt.cm.Spectral_r)
    sorted_votes = symmetry_detector.sort_hexbin_by_votes(hexbin_image)
    r, theta = symmetry_detector.find_coordinate_maxhexbin(hexbin_image, sorted_votes, vertical=False)

    symmetry_line_coords = symmetry_detector.draw_mirrorLine(r, theta, image_title)

    if symmetry_line_coords:
        x1, y1, x2, y2 = symmetry_line_coords

        # Draw the symmetry line on the image
        cv2.line(img_with_corners, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for the symmetry line

        for corner in detected_corners:
            symmetric_point = calculate_opposite_point(corner, (x1, y1, x2, y2))

            # Draw the Harris corner and its corresponding symmetric point
            cv2.circle(img_with_corners, corner, 5, (0, 255, 0), 2)  # Green for the Harris corner
            cv2.circle(img_with_corners, symmetric_point, 5, (255, 0, 0), 2)  # Red for the symmetric point

            # Draw the B-spline connecting the points
            img_with_corners = draw_spline_on_image(img_with_corners, [corner, symmetric_point], spline_color=(255, 165, 0))

        # Display the final image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"{image_title} - Harris Corners, Symmetric Points, and Symmetry Line")
        plt.show()
        
def run_test():
    csv_file = 'occlusion1.csv'  # Update this if you need CSV data for other purposes
    detect_and_draw_symmetry_line(csv_file, "Symmetry Detection - Test Case 1")
    
run_test()
