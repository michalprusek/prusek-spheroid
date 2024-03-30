import numpy as np
import cv2 as cv
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def IoU(projekt, algorithm, mask_ref, mask_pred, name, plot=False, save=True, lock=None, address=None):
    intersection = np.logical_and(mask_ref, mask_pred)
    union = np.logical_or(mask_ref, mask_pred)
    iou = np.sum(intersection) / np.sum(union)

    true_positive = np.sum(np.logical_and(mask_ref, mask_pred))
    false_negative = np.sum(np.logical_and(mask_ref, np.logical_not(mask_pred)))
    false_positive = np.sum(np.logical_and(np.logical_not(mask_ref), mask_pred))

    tpr = true_positive / (true_positive + false_negative)
    ppv = true_positive / (true_positive + false_positive)

    contours_ref, _ = cv.findContours(mask_ref.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv.findContours(mask_pred.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    polygon_ref = [Polygon(cnt.reshape(-1, 2)) for cnt in contours_ref if len(cnt) >= 3]
    polygon_pred = [Polygon(cnt.reshape(-1, 2)) for cnt in contours_pred if len(cnt) >= 3]

    polygon_ref = [p for p in polygon_ref if p.is_valid and p.area > 0]
    polygon_pred = [p for p in polygon_pred if p.is_valid and p.area > 0]

    # Modified to combine plotting and saving functionalities
    handle_polygons(polygon_ref, polygon_pred, iou, name, projekt, algorithm, plot, save, lock, address)

    return iou, tpr, ppv


def handle_polygons(polygons_ref, polygons_pred, iou, name, projekt, algorithm, plot, save, lock, address):
    plt.figure()
    ax = plt.gca()
    ax.set_aspect("equal")

    for polygon in polygons_ref + polygons_pred:
        color, label = ("r-", "Truth") if polygon in polygons_ref else ("b-", "Prediction")
        if polygon.geom_type != "Point":
            xs, ys = polygon.exterior.xy
            plt.plot(xs, ys, color, linewidth=0.4)
            plt.text(xs[0], ys[0], label, color=color[0], fontsize=10, ha="right", va="bottom")

    ax.invert_yaxis()
    title = f"IoU: {iou * 100:.2f}%"
    plt.title(title)

    if plot:
        plt.show()

    if save and lock:
        with lock:
            save_path = f"{address}/{algorithm}_{name.replace('bmp', 'png')}" if address else f"Results/{projekt}/IoU output/plots/{algorithm}_{name.replace('bmp', 'png')}"
            plt.savefig(save_path)

    plt.close()
