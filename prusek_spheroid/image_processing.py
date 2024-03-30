import numpy as np
import cv2 as cv
from skimage import img_as_ubyte
from skimage import feature


def find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours):
    result_mask = np.zeros_like(img_binary, dtype=np.uint8)

    for contour in filtered_contours:
        filled_contour = cv.fillPoly(np.zeros_like(img_binary), [contour], 1)

        intersection = filled_contour & edges

        if np.any(intersection):
            result_mask = result_mask | filled_contour

    if inner_contours:
        inner_contours_mask = np.zeros_like(img_binary, dtype=np.uint8)
        for i in range(len(contours)):
            if hierarchy[0, i, 3] != -1:
                inner_filled_contour = cv.fillPoly(np.zeros_like(img_binary), [contours[i]], 1)

                inner_contours_mask = cv.bitwise_or(inner_filled_contour, inner_contours_mask)

        # inner_contours_mask = f.Erosion(inner_contours_mask, 3, 1)

        return np.clip(result_mask, 0, 1), np.clip(inner_contours_mask, 0, 1)

    return np.clip(result_mask, 0, 1), None


def create_binary_mask(img_gray, threshold, dilation_size, erosion_size=None):
    img_binary = img_as_ubyte(img_gray > threshold)
    img_binary = np.invert(img_binary)
    if erosion_size is not None:
        img_binary = Erosion(img_binary, erosion_size, 1)
    img_binary = Dilation(img_binary, dilation_size, 1)
    return img_binary


def calculate_canny_edges(img_gray, std_k, sigma):
    mean = np.mean(img_gray)
    std = np.std(img_gray)
    low_threshold = mean - std_k * std / 2
    high_threshold = mean + std_k * std / 2
    edges = feature.canny(img_gray, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    return edges


def filter_contours(contours, img_shape, min_area, detect_corrupted=True):
    height, width = img_shape
    filtered_contours = []

    for contour in contours:
        if detect_corrupted:
            if not (np.any(contour[:, :, 0] == 0) or np.any(contour[:, :, 1] == 0) or
                    np.any(contour[:, :, 0] == width - 1) or np.any(contour[:, :, 1] == height - 1)):
                if cv.contourArea(contour) >= min_area:
                    filtered_contours.append(contour)
        else:
            if cv.contourArea(contour) >= min_area:
                filtered_contours.append(contour)

    return filtered_contours


def check_window_size(window_size):
    return window_size + 1 if window_size % 2 == 0 else window_size


def findContours(img_binary, inner_contours):
    if inner_contours:
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


def Dilation(img, dilation_size=3, iterations=1, dilation_shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
                                       (dilation_size, dilation_size))
    img_final = cv.dilate(img, element, iterations=iterations)

    return img_final


def Erosion(img, erosion_size=3, iterations=1, erosion_shape=cv.MORPH_ELLIPSE):
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    img_final = cv.erode(img, element, iterations=iterations)

    return img_final
