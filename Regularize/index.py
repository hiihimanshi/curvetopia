import pandas as pd
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt

# Load and process data
data = pd.read_csv("regularize\frag2.csv", header=None, names=['CurveID', 'ShapeType', 'XCoord', 'YCoord'])

# Function to smooth curve points
def smooth_curve(x_vals, y_vals, smooth_factor=0):
    spline_x = UnivariateSpline(range(len(x_vals)), x_vals, s=smooth_factor)
    spline_y = UnivariateSpline(range(len(y_vals)), y_vals, s=smooth_factor)
    return spline_x(range(len(x_vals))), spline_y(range(len(y_vals)))

# Function to interpolate curve points
def interpolate_curve(x_vals, y_vals, num_points):
    t_vals = np.linspace(0, 1, len(x_vals))
    interp_x = interp1d(t_vals, x_vals, kind='linear')
    interp_y = interp1d(t_vals, y_vals, kind='linear')
    t_new = np.linspace(0, 1, num_points)
    return interp_x(t_new), interp_y(t_new)

# Function to create an image from points
def create_image_from_points(points, img_width=1000, img_height=1000):
    image = np.zeros((img_height, img_width), dtype=np.uint8)
    for x, y in points:
        if 0 <= int(y) < img_height and 0 <= int(x) < img_width:
            image[int(y), int(x)] = 255
    return image

# Function to identify shapes within an image
def identify_shapes(image):
    detected_shapes = []
    edge_image = cv2.Canny(image.copy(), 0, 50)
    blurred_edges = cv2.GaussianBlur(edge_image.copy(), (15, 15), 0)

    # Detect lines
    detected_lines = cv2.HoughLinesP(blurred_edges, 1, np.pi / 2, threshold=200, minLineLength=0, maxLineGap=100)
    if detected_lines is not None:
        for line in detected_lines:
            for x1, y1, x2, y2 in line:
                detected_shapes.append(("Line", np.array([[x1, y1], [x2, y2]])))

    # Detect other shapes
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx_curve = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx_curve) == 3:
            detected_shapes.append(("Triangle", approx_curve))
        elif len(approx_curve) == 4:
            x, y, w, h = cv2.boundingRect(approx_curve)
            aspect_ratio = w / float(h)
            shape = "Square" if 0.85 <= aspect_ratio <= 1.15 else "Rectangle"
            detected_shapes.append((shape, approx_curve))
        elif len(approx_curve) > 4:
            contour_area = cv2.contourArea(contour)
            enclosing_circle_center, enclosing_circle_radius = cv2.minEnclosingCircle(contour)
            circularity = contour_area / (np.pi * enclosing_circle_radius ** 2)
            if 0.70 <= circularity <= 1.3:
                detected_shapes.append(("Circle", (enclosing_circle_center, enclosing_circle_radius)))
            else:
                detected_shapes.append(("Polygon", approx_curve))

            if len(approx_curve) >= 6:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                axes = (int(axes[0] / 2), int(axes[1] / 2))
                ellipse_contour = cv2.ellipse2Poly(
                    center=(int(center[0]), int(center[1])),
                    axes=axes,
                    angle=int(angle),
                    arcStart=0,
                    arcEnd=360,
                    delta=5
                )
                detected_shapes.append(("Ellipse", ellipse_contour))

            if len(approx_curve) == 10:
                detected_shapes.append(("Star", approx_curve))

    shape_priority = {"Circle": 1, "Square": 2, "Rectangle": 3, "Triangle": 4, "Ellipse": 5, "Star": 6, "Polygon": 7, "Line": 8}

    if detected_shapes:
        detected_shapes = sorted(detected_shapes, key=lambda s: shape_priority.get(s[0], 9))
        return [detected_shapes[0]]

    return detected_shapes

# Function to draw identified shapes on an image
def render_shapes_on_image(image, shapes, curve_points=None):
    if len(image.shape) == 2:
        colored_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        colored_image = image.copy()

    blank_image = np.zeros_like(colored_image)
    shape_coordinates = []

    for shape, contour in shapes:
        color = (255, 255, 255)  # White for all shapes

        if shape == "Circle":
            center, radius = contour
            num_points = 100
            angles = np.linspace(0, 2 * np.pi, num_points)
            circle_points = np.array([
                (int(center[0] + radius * np.cos(a)), int(center[1] + radius * np.sin(a)))
                for a in angles
            ])
            cv2.polylines(blank_image, [circle_points], isClosed=True, color=color, thickness=1)
            shape_coordinates.append(("Circle", circle_points))
        else:
            cv2.drawContours(blank_image, [contour], -1, color, 1)
            shape_coordinates.append((shape, contour.squeeze()))

    if curve_points is not None:
        color = (255, 255, 255)
        cv2.polylines(blank_image, [curve_points], isClosed=False, color=color, thickness=1)
        shape_coordinates.append(("Curve", curve_points))

    return blank_image, shape_coordinates

# Function to combine multiple images into one
def merge_images(image_list, positions, canvas_width=1000, canvas_height=1000):
    merged_image = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for img, (x_pos, y_pos) in zip(image_list, positions):
        h, w = img.shape[:2]
        x_pos = max(0, min(x_pos, canvas_width - w))
        y_pos = max(0, min(y_pos, canvas_height - h))
        mask = img != 0
        merged_image[y_pos:y_pos + h, x_pos:x_pos + w][mask] = img[mask]
    return merged_image

# Main processing loop
all_curves = data.groupby(['CurveID', 'ShapeType'])
processed_images = []
image_positions = []
output_coordinates = []

for curve_id, curve_data in all_curves:
    x_vals, y_vals = curve_data['XCoord'].values, curve_data['YCoord'].values
    smoothed_x, smoothed_y = smooth_curve(x_vals, y_vals, smooth_factor=0)
    interpolated_x, interpolated_y = interpolate_curve(smoothed_x, smoothed_y, num_points=1000)

    curve_position = (int(x_vals.min()), int(y_vals.min()))
    curve_points = np.vstack((interpolated_x, interpolated_y)).T
    image_positions.append(curve_position)

    curve_image = create_image_from_points(curve_points, img_width=1000, img_height=1000)
    identified_shapes = identify_shapes(curve_image)

    final_image, shape_coords = render_shapes_on_image(curve_image, identified_shapes, curve_points=np.int32(curve_points))
    processed_images.append(final_image)

    for shape_type, coordinates in shape_coords:
        if shape_type == "Curve":
            for x, y in coordinates:
                output_coordinates.append([curve_id[0], curve_id[1], x, y])
        else:
            for coord in coordinates:
                output_coordinates.append([curve_id[0], curve_id[1], coord[0], coord[1]])

# Combine all processed images into a final output image
final_combined_image = merge_images(processed_images, image_positions, canvas_width=1000, canvas_height=1000)

# Enlarge the final image
scale_factor = 2.0  # Factor by which to enlarge the image
final_resized_image = cv2.resize(final_combined_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Save and display the final enlarged image
cv2.imwrite("final_combined_image_enlarged.png", final_resized_image)

# Set up a larger figure size for better visualization
plt.figure(figsize=(10, 10))  # Adjust figure size to be larger
plt.imshow(final_resized_image, cmap='gray')
plt.axis('off')
plt.show()
