#!/usr/bin/env python3
"""
5-Fold Cross-Validation for Bayesian Optimizer spheroid segmentation.
METHODOLOGY: Per cell line (A549, BxPC-3, MIA PaCa-2, PC-3)

For each cell line separately:
1. Optimize parameters on train data
2. Evaluate on test data
3. Save segmentation masks and images
"""

import os
import sys
import numpy as np
import cv2 as cv
import pandas as pd
from openpyxl import load_workbook

# Add path to module
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# Imports relative to current directory
from BayesianOptimizerGUI import BayesianOptimizer
from ContoursClassGUI import IoU
import Funkce as f

# Configuration
CV_BASE_PATH = "/Users/michalprusek/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Vyzkumak CVAT/CrossValidation"
EXCEL_PATH = "/Users/michalprusek/Desktop/CROSS VALIDATION RESULTS.xlsx"
OUTPUT_ROOT = "/Users/michalprusek/Desktop/CV_BO_Results"  # Output folder in root

NUM_ITERATIONS = 100
BATCH_SIZE = 8
INNER_CONTOURS = False
DETECT_CORRUPTED = True
ALGORITHMS = ["Gaussian", "Sauvola", "Niblack"]
CELL_LINES = ["A549", "BxPC-3", "MIA PaCa-2", "PC-3"]


def load_data_from_folder(folder_path):
    """
    Load data from folder (images + masks).

    Returns:
        List of triplets (mask, image, filename)
    """
    masks_path = os.path.join(folder_path, "masks")
    images_path = os.path.join(folder_path, "images")

    if not os.path.exists(masks_path) or not os.path.exists(images_path):
        raise FileNotFoundError(f"Folder {masks_path} or {images_path} does not exist")

    mask_files = sorted([f for f in os.listdir(masks_path)
                         if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    data = []

    for mask_file in mask_files:
        mask_path_full = os.path.join(masks_path, mask_file)
        img_path = os.path.join(images_path, mask_file)

        if not os.path.exists(img_path):
            print(f"WARNING: Image not found for mask '{mask_file}'")
            continue

        mask = cv.imread(mask_path_full, cv.IMREAD_GRAYSCALE)
        img = cv.imread(img_path)

        if mask is None or img is None:
            print(f"WARNING: Cannot load {mask_file}")
            continue

        data.append((mask, img, mask_file))

    return data


def evaluate_and_save(iou_instance, test_data, parameters, output_dir, cell_line, algorithm):
    """
    Evaluate test data and save segmentation masks and images.

    Returns:
        dict: {filename: iou_value}
    """
    results = {}

    # Create output folders
    masks_output = os.path.join(output_dir, "masks")
    images_output = os.path.join(output_dir, "segmented_images")
    os.makedirs(masks_output, exist_ok=True)
    os.makedirs(images_output, exist_ok=True)

    for mask, img, filename in test_data:
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Apply segmentation algorithm
        img_binary, inner_contours_mask = iou_instance.apply_segmentation_algorithm(
            algorithm, parameters, img_gray, img, filename,
            inner_contours=INNER_CONTOURS,
            detect_corrupted=DETECT_CORRUPTED
        )

        if INNER_CONTOURS and inner_contours_mask is not None:
            intersection = inner_contours_mask & img_binary
            img_binary = img_binary - intersection

        # Find contours and create predicted mask
        contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        pred_mask = np.zeros_like(img_binary, dtype=np.uint8)
        if contours:
            for contour in contours:
                cv.drawContours(pred_mask, [contour], 0, color=255, thickness=-1)

        # Calculate IoU
        intersection = np.logical_and(mask, pred_mask).sum()
        union = np.logical_or(mask, pred_mask).sum()
        iou_value = intersection / union if union > 0 else 0.0

        results[filename] = iou_value

        # Save predicted mask
        cv.imwrite(os.path.join(masks_output, filename), pred_mask)

        # Save segmented image with contours
        img_with_contours = img.copy()
        cv.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)
        cv.imwrite(os.path.join(images_output, filename), img_with_contours)

        print(f"    {filename}: IoU = {iou_value*100:.2f}%")

    return results


def run_cv_per_cell_line():
    """
    Run 5-fold CV per cell line.
    """
    print("="*60)
    print("5-Fold Cross-Validation (Per Cell Line)")
    print("="*60)
    print(f"Cell lines: {CELL_LINES}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Iterations: {NUM_ITERATIONS}")
    print(f"Output: {OUTPUT_ROOT}")
    print("="*60)

    # Create output folder
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Results for Excel: {(cell_line, filename): {algorithm: iou}}
    all_results = {}

    for cell_line in CELL_LINES:
        print(f"\n{'='*60}")
        print(f"Cell Line: {cell_line}")
        print(f"{'='*60}")

        cell_line_path = os.path.join(CV_BASE_PATH, cell_line)

        if not os.path.exists(cell_line_path):
            print(f"  WARNING: Folder {cell_line_path} does not exist, skipping")
            continue

        for algorithm in ALGORITHMS:
            print(f"\n--- Algorithm: {algorithm} ---")

            for fold_idx in range(1, 6):
                fold_path = os.path.join(cell_line_path, f"fold_{fold_idx}")

                if not os.path.exists(fold_path):
                    print(f"  WARNING: Fold {fold_idx} does not exist for {cell_line}")
                    continue

                print(f"\n  Fold {fold_idx}/5:")

                # 1. Load train data
                train_path = os.path.join(fold_path, "train")
                print(f"    Loading train data...")
                train_data = load_data_from_folder(train_path)
                print(f"    Loaded {len(train_data)} train images")

                # 2. Optimization on train data
                print(f"    Starting Bayesian optimization ({NUM_ITERATIONS} iterations)...")

                output_dir = os.path.join(OUTPUT_ROOT, cell_line, f"fold_{fold_idx}", algorithm)
                os.makedirs(output_dir, exist_ok=True)

                def iou_factory(output_addr, projekt, alg, inner_contours=False, detect_corrupted=True):
                    return IoU(output_addr, projekt, alg, inner_contours, detect_corrupted)

                optimizer = BayesianOptimizer(
                    annotation_data=train_data,
                    outputAddress=output_dir,
                    projekt=f"{cell_line}_fold_{fold_idx}",
                    algorithm=algorithm,
                    learning_rate=0.01,
                    num_iterations=NUM_ITERATIONS,
                    delta=0,
                    batch_size=BATCH_SIZE,
                    f=iou_factory,
                    progress_window=None,
                    inner_contours=INNER_CONTOURS,
                    detect_corrupted=DETECT_CORRUPTED
                )

                best_params, best_iou = optimizer.run()
                print(f"    Best IoU on train: {best_iou*100:.2f}%")
                print(f"    Parameters: {best_params}")

                # 3. Evaluation on test data
                print(f"    Evaluating on test data...")
                test_path = os.path.join(fold_path, "test")
                test_data = load_data_from_folder(test_path)
                print(f"    Loaded {len(test_data)} test images")

                # Create IoU instance for evaluation
                iou_evaluator = IoU(
                    output_dir,
                    f"{cell_line}_fold_{fold_idx}_test",
                    algorithm,
                    inner_contours=INNER_CONTOURS,
                    detect_corrupted=DETECT_CORRUPTED
                )

                # Evaluate and save
                fold_results = evaluate_and_save(
                    iou_evaluator, test_data, best_params,
                    output_dir, cell_line, algorithm
                )

                # Save results
                for filename, iou_value in fold_results.items():
                    key = (cell_line, filename)
                    if key not in all_results:
                        all_results[key] = {}
                    all_results[key][f"{algorithm}_BO"] = iou_value

                # Average for fold
                avg_iou = np.mean(list(fold_results.values()))
                print(f"    Average IoU on test: {avg_iou*100:.2f}%")

                # Save parameters to JSON
                import json
                params_file = os.path.join(output_dir, f"best_params_{algorithm}.json")
                with open(params_file, 'w') as f:
                    json.dump({
                        'algorithm': algorithm,
                        'cell_line': cell_line,
                        'fold': fold_idx,
                        'best_iou_train': best_iou,
                        'avg_iou_test': avg_iou,
                        'parameters': {k: float(v) if hasattr(v, 'item') else v
                                      for k, v in best_params.items()}
                    }, f, indent=2)

    # Update Excel
    update_excel(all_results)

    print("\n" + "="*60)
    print("DONE!")
    print(f"Results saved in: {OUTPUT_ROOT}")
    print("="*60)


def update_excel(all_results):
    """
    Update Excel file with new columns for BO results.
    """
    print(f"\n{'='*60}")
    print("Updating Excel file...")
    print(f"{'='*60}")

    # Load existing Excel
    df = pd.read_excel(EXCEL_PATH, sheet_name="original")

    # Add new columns if they don't exist
    for alg in ALGORITHMS:
        col_name = f"{alg}_BO"
        if col_name not in df.columns:
            df[col_name] = None

    # Map results
    matched = 0
    for idx, row in df.iterrows():
        project = row["project"]
        name = row["name"]
        key = (project, name)

        if key in all_results:
            for col_name, iou_value in all_results[key].items():
                df.at[idx, col_name] = iou_value
            matched += 1

    print(f"Mapped {matched}/{len(df)} rows")

    # Save
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name="original", index=False)

    # Print statistics
    print(f"\nAverage IoU (BO):")
    for alg in ALGORITHMS:
        col = f"{alg}_BO"
        if col in df.columns:
            mean_val = df[col].mean()
            median_val = df[col].median()
            if pd.notna(mean_val):
                print(f"  {col}: mean={mean_val*100:.2f}%, median={median_val*100:.2f}%")


if __name__ == "__main__":
    run_cv_per_cell_line()
