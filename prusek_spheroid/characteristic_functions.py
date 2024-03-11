import cv2
import numpy as np
def calculate_area_from_contour(contour):
    return cv2.contourArea(contour)
def calculate_perimeter_from_contour(contour):
    return cv2.arcLength(contour, True)

def calculate_equivalent_diameter_from_contour(contour):
    area = calculate_area_from_contour(contour)
    return np.sqrt(4 * area / np.pi)

def calculate_convex_perimeter_from_contour(contour):
    convex_hull = cv2.convexHull(contour)
    return cv2.arcLength(convex_hull, True)

def calculate_circularity_from_contour(contour):
    area = calculate_area_from_contour(contour)
    convex_perimeter = calculate_convex_perimeter_from_contour(contour)
    return (4 * np.pi * area) / (convex_perimeter ** 2) if convex_perimeter else 0

def calculate_compactness_from_contour(contour):
    area = calculate_area_from_contour(contour)
    perimeter = calculate_perimeter_from_contour(contour)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter else 0

def calculate_convexity_from_contour(contour):
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    contour_perimeter = calculate_perimeter_from_contour(contour)
    return hull_perimeter / contour_perimeter if contour_perimeter else 0

def calculate_solidity_from_contour(contour):
    area = calculate_area_from_contour(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area else 0

def calculate_sphericity_from_contour(contour):
    area = calculate_area_from_contour(contour)
    perimeter = calculate_perimeter_from_contour(contour)
    return np.pi * np.sqrt(4 * area / np.pi) / perimeter if perimeter else 0


def calculate_feret_properties_from_contour(contour):
    # Nalezení minimálního obdélníku, který obaluje konturu
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]

    # Určení Maximálního Feretova průměru a Minimálního Feretova průměru
    feret_diameter_max = max(width, height)
    feret_diameter_min = min(width, height)

    # Výpočet Feretova poměru
    feret_aspect_ratio = feret_diameter_max / feret_diameter_min if feret_diameter_min else 0

    return feret_diameter_max, feret_diameter_min, feret_aspect_ratio


def calculate_diameters_from_contour(contour):
    # Nalezení elipsy, která nejlépe aproximuje konturu
    ellipse = cv2.fitEllipse(contour)
    (major_axis_length, minor_axis_length) = ellipse[1]

    return major_axis_length, minor_axis_length


def calculate_orthogonal_diameter(contour):
    if contour is None or len(contour) < 2:
        return 0

    # Nalezení minimálního obdélníku, který obaluje konturu
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Výpočet vzdáleností mezi páry bodů v rotovaném obdélníku
    distances = [np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)]

    # Určení ortogonálního průměru jako menší z dvou párů stran
    orthogonal_diameter = min(distances[0::2] + distances[1::2])

    return orthogonal_diameter

def calculate_all(contour):
    area = calculate_area_from_contour(contour)
    perimeter = calculate_perimeter_from_contour(contour)
    eq_diam = calculate_equivalent_diameter_from_contour(contour)
    circularity = calculate_circularity_from_contour(contour)
    feret_diameter_max, feret_diameter_min, feret_aspect_ratio = calculate_feret_properties_from_contour(contour)
    feret_max_orthogonal_distance = calculate_orthogonal_diameter(contour)
    major_axis_length, minor_axis_length = calculate_diameters_from_contour(contour)
    compactness = calculate_compactness_from_contour(contour)
    convexity = calculate_convexity_from_contour(contour)
    solidity = calculate_solidity_from_contour(contour)
    sphericity = calculate_sphericity_from_contour(contour)

    data = {
        "Area": area,
        "Perimeter": perimeter,
        "EquivalentDiameter": eq_diam,
        "Circularity": circularity,
        "FeretDiameterMax": feret_diameter_max,
        "FeretDiameterMaxOrthogonalDistance": feret_max_orthogonal_distance,
        "FeretDiameterMin": feret_diameter_min,
        "FeretAspectRatio": feret_aspect_ratio,
        "LengthMajorDiameterThroughCentroid": major_axis_length,
        "LengthMinorDiameterThroughCentroid": minor_axis_length,
        "Compactness": compactness,
        "Convexity": convexity,
        "Solidity": solidity,
        "Sphericity": sphericity
    }

    return data