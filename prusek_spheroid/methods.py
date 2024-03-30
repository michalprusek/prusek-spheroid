import sys
from prusek_spheroid import image_processing as ip
from skimage.filters import threshold_sauvola, threshold_niblack
import cv2 as cv


class BaseImageProcessing:
    def apply_segmentation_algorithm(self, algorithm, parameters, img, inner_contours,
                                     detect_corrupted):
        if algorithm == "Sauvola":
            return self.sauvola(parameters, img, inner_contours, detect_corrupted)
        elif algorithm == "Niblack":
            return self.niblack(parameters, img, inner_contours, detect_corrupted)
        elif algorithm == "Gaussian":
            return self.gaussian_adaptive(parameters, img, inner_contours, detect_corrupted)
        else:
            print(f"Algorithm with name {algorithm} not found.")
            sys.exit(1)

    @staticmethod
    def sauvola(parameters, img_gray, inner_contours, detect_corrupted):
        window_size = ip.check_window_size(int(parameters["window_size"]))
        std_k = parameters["std_k"]
        min_area = parameters["min_area"]
        dilation_size = int(parameters["dilation_size"])
        sigma = parameters["sigma"]

        thresh_sauvola = threshold_sauvola(img_gray, window_size=window_size)
        img_binary = ip.create_binary_mask(img_gray, thresh_sauvola, dilation_size)
        edges = ip.calculate_canny_edges(img_gray, std_k, sigma)

        contours, hierarchy = ip.findContours(img_binary, inner_contours)
        filtered_contours = ip.filter_contours(contours, img_gray.shape, min_area,
                                               detect_corrupted)

        return ip.find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours)

    @staticmethod
    def niblack(parameters, img_gray, inner_contours, detect_corrupted):
        window_size = ip.check_window_size(int(parameters["window_size"]))
        k = parameters["k"]
        min_area = parameters["min_area"]
        std_k = parameters["std_k"]
        dilation_size = int(parameters["dilation_size"])
        sigma = parameters["sigma"]

        thresh_niblack = threshold_niblack(img_gray, window_size=window_size, k=k)
        img_binary = ip.create_binary_mask(img_gray, thresh_niblack, dilation_size, 1)

        edges = ip.calculate_canny_edges(img_gray, std_k, sigma)

        contours, hierarchy = ip.findContours(img_binary, inner_contours)
        filtered_contours = ip.filter_contours(contours, img_gray.shape, min_area,
                                               detect_corrupted)

        return ip.find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours)

    @staticmethod
    def gaussian_adaptive(parameters, img_gray, inner_contours, detect_corrupted):
        window_size = ip.check_window_size(int(parameters["window_size"]))
        min_area = parameters["min_area"]
        std_k = parameters["std_k"]
        dilation_size = int(parameters["dilation_size"])
        sigma = parameters["sigma"]

        img_binary = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,
                                          window_size, 0)

        img_binary = ip.Erosion(img_binary, 1, 1)

        img_binary = ip.Dilation(img_binary, dilation_size, 1)

        edges = ip.calculate_canny_edges(img_gray, std_k, sigma)

        contours, hierarchy = ip.findContours(img_binary, inner_contours)

        filtered_contours = ip.filter_contours(contours, img_gray.shape, min_area,
                                               detect_corrupted)

        result = ip.find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours)

        return result
