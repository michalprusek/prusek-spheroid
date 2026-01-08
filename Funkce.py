import cv2 as cv
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.validation import make_valid
import math
import rasterio.features
import warnings
import numpy as np
import json as js
# Optional tkinter import (for headless environments)
try:
    import tkinter as tk
    from tkinter import messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    messagebox = None

warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely.set_operations")


def show_error(title, message):
    """Display error using messagebox or print."""
    if HAS_TKINTER:
        messagebox.showerror(title, message)
    else:
        print(f"ERROR: {message}")


def watershed_convergence(img_bgr, img_binary, contour):
    img = img_bgr.copy()

    # Create a black image for drawing contours
    drawing = np.zeros_like(img_binary, dtype=np.uint8)
    cv.drawContours(drawing, contour, -1, 255, thickness=cv.FILLED)

    # Extract coordinates of all contour points as seed points
    contour_points = np.squeeze(contour)
    seed_points = [tuple(point) for point in contour_points]

    # Create a marker image for watershed
    marker_image = np.zeros_like(img_binary, dtype=np.int32)

    # Dilate the filled contour to get sure background
    dilation_size = 10
    element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    sure_bg = cv.dilate(drawing, element)

    # Set sure background in marker image
    marker_image[sure_bg == 255] = 1

    # Set sure foreground for the region inside the contour
    cv.drawContours(marker_image, contour, -1, 2, thickness=cv.FILLED)

    # Set every pixel of the contour as a seed point
    for i, seed_point in enumerate(seed_points, start=3):  # Start from label 3 (2 is used for sure foreground)
        cv.circle(marker_image, seed_point, 1, i, -1)

    # Apply watershed algorithm
    cv.watershed(img, marker_image)

    # Set watershed boundaries in the original image
    img[marker_image == -1] = [0, 0, 255]  # Red color for watershed boundaries

    return img


def find_contour_centroid(contour):
    M = cv.moments(contour)

    # Calculate the centroid coordinates
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])

    return centroid_x, centroid_y


def convert_contours_to_coco(outer_contours, inner_contours, height, width, img_name, start_id, coco_data):
    new_id = start_id  # Starting ID for annotations
    outer_category_id = 1  # ID for label "spheroids"
    inner_category_id = 2  # ID for label "holes"

    # Add image information
    image_info = {
        "id": len(coco_data["images"]) + 1,
        "file_name": img_name,
        "width": width,
        "height": height,
        "license": 0,
        "flickr_url": "",
        "coco_url": "",
        "date_captured": 0
    }

    coco_data["images"].append(image_info)

    def process_contours(contours, category_id):
        nonlocal new_id
        for contour in contours:
            # Ensure each contour has enough points
            if len(contour) < 3:
                continue  # Skip contours with less than 3 points

            segmentation = contour.flatten().tolist()

            annotation_data = {
                "id": new_id,
                "image_id": image_info["id"],
                "category_id": category_id,
                "segmentation": [segmentation],
                "area": float(cv.contourArea(np.array(contour))),
                "bbox": cv.boundingRect(np.array(contour))[:4],
                "iscrowd": 0
            }

            coco_data["annotations"].append(annotation_data)
            new_id += 1

    # Process outer contours
    process_contours(outer_contours, outer_category_id)

    # Process inner contours if available
    if inner_contours:
        process_contours(inner_contours, inner_category_id)

    return coco_data


def Erosion(img, erosion_size=3, iterations=1, erosion_shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    img_final = cv.erode(img, element, iterations=iterations)

    return img_final


def find_bounds(contour):
    upper_bound = np.max(contour[:, 0, 0])
    lower_bound = np.min(contour[:, 0, 0])
    right_bound = np.max(contour[:, 0, 1])
    left_bound = np.min(contour[:, 0, 1])

    return lower_bound, upper_bound, left_bound, right_bound


def Dilation(img, dilation_size=3, iterations=1, dilation_shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    img_final = cv.dilate(img, element, iterations=iterations)

    return img_final


def initialize_coco_data():
    coco_data = {
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "spheroids", "supercategory": ""},
                       {"id": 2, "name": "microtubules", "supercategory": ""}],
        # Assuming there is one category named "object"
        "licenses": [{"name": "", "id": 0, "url": ""}]
    }
    return coco_data


def threshold_img(img, threshold):
    img_thresh = img.copy()
    img_thresh[img >= threshold] = 255
    img_thresh[img < threshold] = 0
    return img_thresh


def mean_squared_deviation_area(contours):
    total_lenght = 0

    for contour in contours:
        lenght = len(contour)
        total_lenght += lenght

    mean_lenght = total_lenght / len(contours)

    msd = 0
    for contour in contours:
        lenght = len(contour)
        msd += (lenght - mean_lenght) ** 2

    msd /= len(contours)

    msd = math.sqrt(msd)

    return mean_lenght, msd


def subprocess_img(img_gray, ot, erosion_element, closing_element, show_img):
    img_otsu = threshold_img(img_gray, ot)

    img_otsu = Dilation(Erosion(img_otsu, closing_element), closing_element)

    img_otsu = Erosion(img_otsu, erosion_element)

    img_otsu = np.invert(img_otsu)

    if show_img:
        cv.imshow("img", img_otsu)
        cv.waitKey(0)

    contours, hierarchy = cv.findContours(img_otsu, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    return contours, img_otsu


def process_img_V2(img_gray, threshold_multiplier=1.0, show_img=False, erosion_element=3, closing_element=9,
                   radius_moments=None):
    ot, img_otsu = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, img_otsu = subprocess_img(img_gray, ot * threshold_multiplier, erosion_element, closing_element, show_img)
    contour = max(contours, key=cv.contourArea)

    if radius_moments is not None:
        print("not none")

    return contour, img_otsu


def plot_polygons(polygon1, polygon2, title):
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")
    if polygon1.geom_type != "Point":
        xs, ys = polygon1.exterior.xy
        plt.plot(xs, ys, "r-", linewidth=0.4)
    if polygon2.geom_type != "Point":
        xs, ys = polygon2.exterior.xy
        plt.plot(xs, ys, "b-", linewidth=0.4)
    ax.invert_yaxis()
    plt.title(title)
    plt.show()
    plt.close()


def save_polygons(polygons_ref, polygons_pred, title, name, projekt, algorithm, address=None):
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")

    # Process list of polygons for ground truth
    for polygon in polygons_ref:
        if polygon.geom_type != "Point":
            xs, ys = polygon.exterior.xy
            plt.plot(xs, ys, "r-", linewidth=0.4)
            plt.text(xs[0], ys[0], "Truth", color="red", fontsize=10, ha="right", va="bottom")

    # Process list of polygons for prediction
    for polygon in polygons_pred:
        if polygon.geom_type != "Point":
            xs, ys = polygon.exterior.xy
            plt.plot(xs, ys, "b-", linewidth=0.4)
            plt.text(xs[0], ys[0], "Prediction", color="blue", fontsize=10, ha="right", va="bottom")

    ax.invert_yaxis()
    plt.title(title)

    # Save image
    if address:
        plt.savefig(f"{address}/{algorithm}_{name.replace('bmp', 'png')}")
    else:
        plt.savefig(f"Results/{projekt}/IoU output/plots/{algorithm}_{name.replace('bmp', 'png')}")

    plt.close()


def load_annotations(annotations_address, images_address):
    with open(annotations_address) as f:
        data = js.load(f)

    masks_per_image = {}
    annotations_data = []

    for annotation in data["annotations"]:
        img_id = annotation["image_id"]
        img_info = next((item for item in data["images"] if item["id"] == img_id), None)

        if img_info is not None:
            img_name = img_info["file_name"]
            img_width = img_info["width"]
            img_height = img_info["height"]

            img_path = os.path.join(images_address, img_name)
            img = cv.imread(img_path)

            # Initialize mask for each image
            if img_name not in masks_per_image:
                masks_per_image[img_name] = np.zeros((img_height, img_width), dtype=np.uint8)

            # Create contours for each segment
            for segmentation in annotation["segmentation"]:
                points = np.array([[segmentation[i], segmentation[i + 1]] for i in range(0, len(segmentation), 2)],
                                  dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv.fillPoly(masks_per_image[img_name], [points], 255)

            annotations_data.append((masks_per_image[img_name], img, img_name))

    return annotations_data


def checkPolygon(geom):
    if geom.geom_type == "GeometryCollection":
        new_poly = geom.geoms[0]
    elif geom.geom_type == "Multipolygon":
        new_poly = max(geom, key=lambda a: a.area)
    else:
        new_poly = geom

    return new_poly


def IoU(projekt, algorithm, mask_ref, mask_pred, name, plot=False, save=True, lock=None, address=None):
    # Calculate IoU
    intersection = np.logical_and(mask_ref, mask_pred)
    union = np.logical_or(mask_ref, mask_pred)
    IoU = np.sum(intersection) / np.sum(union)

    true_positive = np.sum(np.logical_and(mask_ref, mask_pred))
    false_negative = np.sum(np.logical_and(mask_ref, np.logical_not(mask_pred)))
    false_positive = np.sum(np.logical_and(np.logical_not(mask_ref), mask_pred))

    # Calculate True Positive Rate (TPR) and Positive Predictive Value (PPV)
    tpr = true_positive / (true_positive + false_negative)
    ppv = true_positive / (true_positive + false_positive)

    # Get contours from masks
    contours_ref, _ = cv.findContours(mask_ref.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv.findContours(mask_pred.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Convert contours directly to polygon representations
    polygon_ref = [Polygon(cnt.reshape(-1, 2)) for cnt in contours_ref if len(cnt) >= 3]
    polygon_pred = [Polygon(cnt.reshape(-1, 2)) for cnt in contours_pred if len(cnt) >= 3]

    # Remove invalid polygons
    polygon_ref = [p for p in polygon_ref if p.is_valid and p.area > 0]
    polygon_pred = [p for p in polygon_pred if p.is_valid and p.area > 0]

    # Plot and save polygons if requested
    if plot:
        plot_polygons(polygon_ref, polygon_pred, "IoU: " + str(IoU * 100) + "%")
    if save:
        with lock:
            if address:
                save_polygons(polygon_ref, polygon_pred, "IoU: " + str(IoU * 100) + "%", name, projekt, algorithm,
                              address=address)
            else:
                save_polygons(polygon_ref, polygon_pred, "IoU: " + str(IoU * 100) + "%", name, projekt, algorithm)

    return IoU, tpr, ppv


def f1_score(polygon1, polygon2):
    if not polygon1.is_valid:
        polygon1_valid = checkPolygon(make_valid(polygon1))
    else:
        polygon1_valid = polygon1
    if not polygon2.is_valid:
        polygon2_valid = checkPolygon(make_valid(polygon2))
    else:
        polygon2_valid = polygon2

    poly_inter = checkPolygon(polygon1_valid.intersection(polygon2_valid))
    area_polygon1 = checkPolygon(polygon1_valid).area
    area_polygon2 = checkPolygon(polygon2_valid).area

    f1 = round(2 * poly_inter.area / (area_polygon1 + area_polygon2), 4)

    return f1


def create_polygon_mask(polygon, image_shape):
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Create a generator of geometry and value pairs
    shapes = [(polygon, 1)]

    # Rasterize the geometry onto the image
    rasterized = rasterio.features.geometry_mask(
        shapes,
        out_shape=image_shape,
        transform=rasterio.transform.from_origin(0, image_shape[0], 1, 1),  # Adjust origin and resolution as needed
        invert=True,  # Invert mask to get True inside the polygon
        all_touched=True
    )

    mask[rasterized] = 1

    return mask


def pair_images(adresaDatasetuHand, adresaDatasetuFace, filename):
    adresaFace = os.path.join(adresaDatasetuFace, filename)
    adresaHand = os.path.join(adresaDatasetuHand, filename)
    imgHand = cv.imread(adresaHand)
    imgFace = cv.imread(adresaFace)

    return imgHand, imgFace


def is_circle_inside_image(contour, image_shape):
    # Fit the minimum enclosing circle around the contour
    (x, y), radius = cv.minEnclosingCircle(contour)

    # Check if the circle lies completely inside the image
    if (0 <= x - radius < image_shape[1] and
            0 <= y - radius < image_shape[0] and
            x + radius < image_shape[1] and
            y + radius < image_shape[0]):
        return True
    else:
        return False


def bernsen_thresholding(image, window_size=15):
    half_size = window_size // 2
    rows, cols = image.shape
    thresholded_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            local_window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            threshold = (np.max(local_window) + np.min(local_window)) / 2
            thresholded_image[i, j] = 255 if image[i, j] > threshold else 0

    return thresholded_image


def nick_thresholding(image, window_size=15, k=0.15):
    half_size = window_size // 2
    rows, cols = image.shape
    thresholded_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            local_window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            threshold = np.mean(local_window) + k * np.sqrt(np.var(local_window))
            thresholded_image[i, j] = 255 if image[i, j] > threshold else 0

    return thresholded_image


def bradley_rota_thresholding(image, window_size=15, r=0.25):
    half_size = window_size // 2
    rows, cols = image.shape
    thresholded_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            local_window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            threshold = np.median(local_window) * (1 - r)
            thresholded_image[i, j] = 255 if image[i, j] > threshold else 0

    return thresholded_image


def otsu_adaptive_thresholding(image, window_size=15):
    half_size = window_size // 2
    rows, cols = image.shape
    thresholded_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            local_window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            _, threshold = cv.threshold(local_window, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            thresholded_image[i, j] = 255 if image[i, j] > threshold else 0

    return thresholded_image


def wolf_thresholding(image, window_size=15, k=0.5):
    half_size = window_size // 2
    rows, cols = image.shape
    thresholded_image = np.zeros_like(image, dtype=np.uint8)

    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            local_window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            threshold = np.mean(local_window) + k * np.sqrt(np.var(local_window) - (np.mean(local_window) ** 2))
            thresholded_image[i, j] = 255 if image[i, j] > threshold else 0

    return thresholded_image


def is_point_inside_contour(contour, point):
    # Get distance from point to nearest point on contour
    distance = cv.pointPolygonTest(contour, point, True)

    # If distance is negative, point is inside contour
    return distance > 0


def check_contour_in_corners(contour, image_width, image_height):
    margin = 2
    # Define image corner coordinates
    top_left_corner = (margin, margin)
    top_right_corner = (image_width - 1 - margin, margin)
    bottom_left_corner = (margin, image_height - 1 - margin)
    bottom_right_corner = (image_width - 1 - margin, image_height - 1 - margin)

    # Iterate through corners
    corners = [top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner]

    # Check if point lies inside contour for each corner
    for corner in corners:
        if is_point_inside_contour(contour, corner):
            return False  # Contour extends into corner

    return True


def clahe_correction(img_gray, clip_limit=2.0, tile_size=(8, 8)):
    # Initialize CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    # Apply CLAHE to grayscale image
    clahe_image = clahe.apply(img_gray)

    return clahe_image


def compute_optimal_radii(img_color, k):
    # Convert color image to LAB color space
    lab_img = cv.cvtColor(img_color, cv.COLOR_BGR2Lab)

    # Calculate the standard deviation of the L channel
    std_dev = np.std(lab_img[:, :, 0])

    # Set spatial and color radius based on the standard deviation
    spatial_radius = int(k * std_dev)
    color_radius = int(k * std_dev)

    return spatial_radius, color_radius

def load_masks_from_images(mask_address, images_address):
    mask_files = [f for f in os.listdir(mask_address) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    data = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_address, mask_file)
        img_path = os.path.join(images_address, mask_file)

        if not os.path.exists(img_path):
            show_error("Error", f"Image not found for mask '{mask_file}'")
            continue

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        img = cv.imread(img_path)

        if mask is None:
            show_error("Error", f"Failed to load mask {mask_file}.")
            continue

        # Check if the mask pixel values are only 0 and 255
        unique_values = np.unique(mask)
        if not np.array_equal(unique_values, [0, 255]) and len(unique_values) != 2:
            show_error("Error", f"File {mask_file} does not contain a binary mask. It contains unique values: {unique_values}")
            continue

        data.append((mask, img, mask_file))

    return data

def load_masks(mask_address):
    mask_files = [f for f in os.listdir(mask_address) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    data = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_address, mask_file)

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if mask is None:
            show_error("Error", f"Failed to load mask {mask_file}.")
            continue

        # Check if the mask pixel values are only 0 and 255
        unique_values = np.unique(mask)
        if not np.array_equal(unique_values, [0, 255]) and len(unique_values) != 2:
            show_error("Error", f"File {mask_file} does not contain a binary mask. It contains unique values: {unique_values}")
            continue

        data.append((mask, mask_file))

    return data
