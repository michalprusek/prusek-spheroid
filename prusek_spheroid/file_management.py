import os
import shutil
import zipfile
import cv2 as cv
import numpy as np
import json as js
from tkinter import messagebox


def create_directory(directory_path, delete=False):
    if delete and os.path.exists(directory_path):
        # Remove all files and subdirectories in the specified directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Error occurred while deleting {file_path}. Error: {e}')
    elif not os.path.exists(directory_path):
        os.makedirs(directory_path)


def zip_folder(folder_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=arcname)


def initialize_coco_data():
    coco_data = {
        "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "spheroids", "supercategory": ""},
                       {"id": 2, "name": "holes", "supercategory": ""}],
        # Assuming there is one category named "object"
        "licenses": [{"name": "", "id": 0, "url": ""}]
    }
    return coco_data


def convert_contours_to_coco(outer_contours, inner_contours, height, width, img_name, start_id, coco_data):
    new_id = start_id  # Starting ID for annotations
    outer_category_id = 1  # ID pro štítek "spheroids"
    inner_category_id = 2  # ID pro štítek "holes"

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

    # Zpracování vnějších kontur
    process_contours(outer_contours, outer_category_id)

    # Zpracování vnitřních kontur, pokud jsou k dispozici
    if inner_contours:
        process_contours(inner_contours, inner_category_id)

    return coco_data


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

            # Inicializace masky pro každý obrázek
            if img_name not in masks_per_image:
                masks_per_image[img_name] = np.zeros((img_height, img_width), dtype=np.uint8)

            # Pro každý segment vytvořte kontury
            for segmentation in annotation["segmentation"]:
                points = np.array([[segmentation[i], segmentation[i + 1]] for i in range(0, len(segmentation), 2)],
                                  dtype=np.int32)
                points = points.reshape((-1, 1, 2))
                cv.fillPoly(masks_per_image[img_name], [points], 255)

            annotations_data.append((masks_per_image[img_name], img, img_name))

    return annotations_data


def load_masks(mask_address):
    mask_files = [f for f in os.listdir(mask_address) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    data = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_address, mask_file)

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        if mask is None:
            messagebox.showerror("Error", f"Failed to load mask {mask_file}.")
            continue

        # Check if the mask pixel values are only 0 and 255
        unique_values = np.unique(mask)
        if not np.array_equal(unique_values, [0, 255]) and len(unique_values) != 2:
            messagebox.showerror("Error",
                                 f"File {mask_file} does not contain a binary mask. It contains unique values: {unique_values}")
            continue

        data.append((mask, mask_file))

    return data


def load_masks_from_images(mask_address, images_address):
    mask_files = [f for f in os.listdir(mask_address) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
    data = []

    for mask_file in mask_files:
        mask_path = os.path.join(mask_address, mask_file)
        img_path = os.path.join(images_address, mask_file)

        if not os.path.exists(img_path):
            messagebox.showerror("Error", f"Image not found for mask '{mask_file}'")
            continue

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        img = cv.imread(img_path)

        if mask is None:
            messagebox.showerror("Error", f"Failed to load mask {mask_file}.")
            continue

        # Check if the mask pixel values are only 0 and 255
        unique_values = np.unique(mask)
        if not np.array_equal(unique_values, [0, 255]) and len(unique_values) != 2:
            messagebox.showerror("Error",
                                 f"File {mask_file} does not contain a binary mask. It contains unique values: {unique_values}")
            continue

        data.append((mask, img, mask_file))

    return data
