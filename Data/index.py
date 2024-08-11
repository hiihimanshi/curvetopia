import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import svgwrite

# Function to read CSV file
def read_csv(csv_path):
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
        path_XYs = []
        
        unique_paths = np.unique(np_path_XYs[:, 0])
        for i in unique_paths:
            npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
            XYs = []
            unique_points = np.unique(npXYs[:, 0])
            for j in unique_points:
                XY = npXYs[npXYs[:, 0] == j][:, 1:]
                XYs.append(XY)
            path_XYs.append(np.vstack(XYs))  # Combine all points in the path
            
        return path_XYs
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return []

# Function to detect if a path is a straight line
def is_straight_line(points, threshold=10.0):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    fit_line = m * x + c
    mse = np.mean((y - fit_line) ** 2)
    return mse < threshold, (m, c)

# Function to fit a circle to a path
def fit_circle(points):
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, y, np.ones(len(x))]).T
    B = x**2 + y**2
    C, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    xc, yc = C[0]/2, C[1]/2
    radius = np.sqrt(C[2] + xc**2 + yc**2)
    return xc, yc, radius

# Function to detect if a path is a circle
def is_circle(points, threshold=10.0):
    xc, yc, radius = fit_circle(points)
    distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
    return np.std(distances) < threshold, (xc, yc, radius)

# Function to fit an ellipse to a path
def fit_ellipse(points):
    x = points[:, 0]
    y = points[:, 1]
    
    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = np.dot(D1.T, D1)
    S2 = np.dot(D1.T, D2)
    S3 = np.dot(D2.T, D2)
    T = -np.linalg.inv(S3).dot(S2.T)
    M = S1 + S2.dot(T)
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    eigval, eigvec = np.linalg.eig(M)
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
    a1 = eigvec[:, cond > 0]
    
    a = np.concatenate([a1, T.dot(a1)])
    a = a.ravel()
    
    # Ellipse parameters
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    
    F = 1 + (d**2) / (4 * a) + (f**2) / (4 * c)
    a = np.sqrt(F / a)
    b = np.sqrt(F / c)
    
    return x0, y0, a, b

# Function to detect if a path is an ellipse
def is_ellipse(points, threshold=0.5):
    try:
        x0, y0, a, b = fit_ellipse(points)
        if np.isreal(x0) and np.isreal(y0) and a > 0 and b > 0:
            x0, y0, a, b = float(x0), float(y0), float(a), float(b)
        else:
            return False, None

        theta = np.arctan2(points[:, 1] - y0, points[:, 0] - x0)
        distances = ((points[:, 0] - x0) / a) ** 2 + ((points[:, 1] - y0) / b) ** 2
        return np.std(distances) < threshold, (x0, y0, a, b)
    except Exception as e:
        print(f"Ellipse fitting failed: {e}")
        return False, None

# Function to detect if a path is a regular polygon
def is_regular_polygon(points, threshold=0.5):
    num_points = len(points)
    if num_points < 3:
        return False, None  # A polygon must have at least 3 points

    side_lengths = np.linalg.norm(np.diff(points, axis=0, append=[points[0]]), axis=1)
    if np.std(side_lengths) > threshold * np.mean(side_lengths):
        return False, None

    vectors = np.diff(points, axis=0, append=[points[0]])
    angles = []
    for i in range(num_points):
        vec1 = vectors[i]
        vec2 = vectors[(i + 1) % num_points]
        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angles.append(np.degrees(angle))

    if np.std(angles) > threshold * np.mean(angles):
        return False, None

    return True, points

# Function to detect shapes in a given path
def detect_shapes_in_path(path):
    detected_shapes = []
    
    # Check if the path is a straight line
    is_line, line_params = is_straight_line(path)
    if is_line:
        detected_shapes.append({'type': 'line', 'params': line_params})
    
    # Check if the path is a circle
    is_circle_shape, circle_params = is_circle(path)
    if is_circle_shape:
        detected_shapes.append({'type': 'circle', 'params': circle_params})
    
    # Check if the path is an ellipse
    is_ellipse_shape, ellipse_params = is_ellipse(path)
    if is_ellipse_shape:
        detected_shapes.append({'type': 'ellipse', 'params': ellipse_params})
    
    # Check if the path is a regular polygon
    is_polygon, polygon_params = is_regular_polygon(path)
    if is_polygon:
        detected_shapes.append({'type': 'polygon', 'params': polygon_params})
    
    return detected_shapes

# Function to plot and save detected shapes as an image
def plot_and_save_detected_shapes(paths, shapes, output_image_path):
    fig, ax = plt.subplots()
    
    # Plot original paths
    for path in paths:
        plt.plot(path[:, 0], path[:, 1], 'o-', color='gray')
    
    # Plot detected shapes
    for shape in shapes:
        if shape['type'] == 'line':
            m, c = shape['params']
            x = np.array([0, 512])
            y = m * x + c
            plt.plot(x, y, 'g--', label='Line')
        elif shape['type'] == 'circle':
            xc, yc, radius = shape['params']
            circle = plt.Circle((xc, yc), radius, color='b', fill=False, label='Circle')
            ax.add_patch(circle)
        elif shape['type'] == 'ellipse':
            x0, y0, a, b = shape['params']
            ellipse = patches.Ellipse((x0, y0), 2*a, 2*b, angle=0, edgecolor='y', fill=False, label='Ellipse')
            ax.add_patch(ellipse)
        elif shape['type'] == 'polygon':
            points = shape['params']
            polygon = patches.Polygon(points, closed=True, edgecolor='m', fill=False, label='Polygon')
            ax.add_patch(polygon)

    ax.set_aspect('equal', 'box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig(output_image_path)
    plt.show()

# Example usage
csv_path = '../Data/frag1.csv'  # Ensure the path is correct
output_image_path = '../Data/detected_shapes.png'

# Read path data from CSV
path_XYs = read_csv(csv_path)
if path_XYs:
    # Detect shapes
    all_detected_shapes = []
    for path in path_XYs:
        detected_shapes = detect_shapes_in_path(path)
        all_detected_shapes.extend(detected_shapes)

    # Plot and save detected shapes as an image
    plot_and_save_detected_shapes(path_XYs, all_detected_shapes, output_image_path)
else:
    print("No paths found in CSV.")
