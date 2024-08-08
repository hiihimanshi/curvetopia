import cv2
import numpy as np
import os

print("Current working directory:", os.getcwd())

# Define the shape detection functions
def is_straight_line(points, threshold=1.0):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    fit_line = m * x + c
    mse = np.mean((y - fit_line) ** 2)
    return mse < threshold, (m, c)

def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, y, np.ones(len(x))]).T
    B = x**2 + y**2
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = C[0]/2, C[1]/2
    radius = np.sqrt(C[2] + xc**2 + yc**2)
    return xc, yc, radius

def is_circle(points, threshold=1.0):
    xc, yc, radius = fit_circle(points)
    distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
    return np.std(distances) < threshold, (xc, yc, radius)

def fit_ellipse(points):
    x = points[:, 0]
    y = points[:, 1]
    D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    _, _, V = np.linalg.svd(np.dot(np.linalg.inv(S), C))
    a = V[0, :]
    return a

def is_ellipse(points, threshold=1.0):
    a = fit_ellipse(points)
    # Check the geometric properties of the ellipse
    return True, a  # Implement specific checks based on ellipse parameters

def detect_corners(points):
    # Implement corner detection (e.g., Harris corner detection)
    corners = cv2.goodFeaturesToTrack(np.float32(points), maxCorners=4, qualityLevel=0.01, minDistance=10)
    return corners

def calculate_angles(corners):
    # Calculate angles between corners
    def angle(pt1, pt2, pt0):
        dx1 = pt1[0] - pt0[0]
        dy1 = pt1[1] - pt0[1]
        dx2 = pt2[0] - pt0[0]
        dy2 = pt2[1] - pt0[1]
        return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2))

    angles = []
    for i in range(4):
        angles.append(angle(corners[i], corners[(i+1) % 4], corners[(i+2) % 4]))
    return np.degrees(np.arccos(angles))

def check_rounded_corners(points):
    # Check for rounded corners by analyzing curvature
    pass  # Placeholder for actual implementation

def is_rectangle(points, threshold=0.1):
    corners = detect_corners(points)
    if len(corners) != 4:
        return False, corners
    angles = calculate_angles(corners)
    return np.allclose(angles, 90, atol=threshold), corners

def is_rounded_rectangle(points, threshold=0.1):
    is_rect, corners = is_rectangle(points, threshold)
    if is_rect:
        return check_rounded_corners(points), corners
    return False, corners

def calculate_side_lengths(points):
    # Calculate side lengths of the polygon
    pass

def calculate_internal_angles(points):
    # Calculate internal angles of the polygon
    pass

def is_regular_polygon(points, threshold=0.1):
    side_lengths = calculate_side_lengths(points)
    angles = calculate_internal_angles(points)
    return np.std(side_lengths) < threshold and np.std(angles) < threshold, (side_lengths, angles)

def find_center(points):
    # Find the center point of the star shape
    pass

def calculate_radial_distances(points, center):
    # Calculate radial distances from the center
    pass

def check_radial_symmetry(distances, threshold=0.1):
    # Check for radial symmetry
    pass

def is_star_shape(points, threshold=0.1):
    center = find_center(points)
    radial_distances = calculate_radial_distances(points, center)
    return check_radial_symmetry(radial_distances, threshold), center

# Define the function to draw detected shapes
def draw_detected_shapes(image, shapes):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for shape in shapes:
        if shape['type'] == 'line':
            m, c = shape['params']
            x = np.array([0, image.shape[1]])
            y = m * x + c
            cv2.line(output_image, (int(x[0]), int(y[0])), (int(x[1]), int(y[1])), (0, 255, 0), 2)
        elif shape['type'] == 'circle':
            xc, yc, radius = shape['params']
            cv2.circle(output_image, (int(xc), int(yc)), int(radius), (255, 0, 0), 2)
        elif shape['type'] == 'ellipse':
            # Implement drawing ellipse based on its parameters
            pass
        elif shape['type'] == 'rectangle':
            corners = shape['params']
            for i in range(len(corners)):
                cv2.line(output_image, tuple(corners[i][0]), tuple(corners[(i+1) % len(corners)][0]), (0, 255, 255), 2)
        elif shape['type'] == 'rounded_rectangle':
            # Implement drawing rounded rectangle based on its corners
            pass
        elif shape['type'] == 'polygon':
            # Implement drawing regular polygon
            pass
        elif shape['type'] == 'star':
            # Implement drawing star shape
            pass
    return output_image

# Main script
if __name__ == "__main__":
    # Step 1: Load and preprocess the image
    image = cv2.imread('circle.png', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load image. Check the file path.")
        exit()

    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Step 2: Process contours to detect shapes
    shapes = []
    for contour in contours:
        contour = contour.squeeze()
        if len(contour.shape) == 1:  # Skip single point contours
            continue

        if len(contour) < 5:  # Minimum points to fit ellipse
            if is_straight_line(contour)[0]:
                shapes.append({'type': 'line', 'params': is_straight_line(contour)[1]})
            elif is_circle(contour)[0]:
                shapes.append({'type': 'circle', 'params': is_circle(contour)[1]})
            elif is_rectangle(contour)[0]:
                shapes.append({'type': 'rectangle', 'params': is_rectangle(contour)[1]})

    # Step 3: Draw detected shapes
    output_image = draw_detected_shapes(image, shapes)
    cv2.imwrite('output_image.png', output_image)
