#!/usr/bin/env python3
"""
5-Fold Cross-Validation pro Bayesian Optimizer segmentace sferoidů.

Tento skript:
1. Pro každý fold (1-5) optimalizuje parametry na train datech
2. Evaluuje nalezené parametry na test datech
3. Zapisuje výsledky IoU do existujícího Excel souboru
"""

import os
import sys
import numpy as np
import cv2 as cv
import pandas as pd
from openpyxl import load_workbook
import tempfile

# Přidat cestu k modulu
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

# Importy relativní k aktuálnímu adresáři
from BayesianOptimizerGUI import BayesianOptimizer
from ContoursClassGUI import IoU
import Funkce as f

# Konfigurace
CV_DATA_PATH = "/Users/michalprusek/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Vyzkumak CVAT/CrossValidation/ALL"
EXCEL_PATH = "/Users/michalprusek/Desktop/CROSS VALIDATION RESULTS.xlsx"
OUTPUT_DIR = tempfile.mkdtemp(prefix="bayesian_cv_")

NUM_ITERATIONS = 100  # Stejně jako benchmark
BATCH_SIZE = 8
INNER_CONTOURS = False
DETECT_CORRUPTED = True
ALGORITHMS = ["Gaussian", "Sauvola", "Niblack"]


def parse_filename(filename):
    """
    Parsuje název souboru na project a name pro mapování na Excel.

    Příklady:
        A549_A7.png → ("A549", "A7.png")
        BxPC-3_A 8.png → ("BxPC-3", "A 8.png")
        MIA_PaCa-2_B 3.png → ("MIA PaCa-2", "B 3.png")
        PC-3_A10.png → ("PC-3", "A10.png")
    """
    if filename.startswith("MIA_PaCa-2_"):
        return "MIA PaCa-2", filename[11:]
    elif filename.startswith("BxPC-3_"):
        return "BxPC-3", filename[7:]
    elif filename.startswith("A549_"):
        return "A549", filename[5:]
    elif filename.startswith("PC-3_"):
        return "PC-3", filename[5:]
    else:
        # Fallback: první část před _ je project
        parts = filename.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "Unknown", filename


def load_data_from_fold(fold_path, subset="train"):
    """
    Načte data z dané složky foldu (train nebo test).

    Args:
        fold_path: Cesta k fold složce (např. fold_1)
        subset: "train" nebo "test"

    Returns:
        Seznam trojic (mask, image, filename)
    """
    masks_path = os.path.join(fold_path, subset, "masks")
    images_path = os.path.join(fold_path, subset, "images")

    if not os.path.exists(masks_path) or not os.path.exists(images_path):
        raise FileNotFoundError(f"Složka {masks_path} nebo {images_path} neexistuje")

    mask_files = sorted([f for f in os.listdir(masks_path)
                         if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    data = []

    for mask_file in mask_files:
        mask_path = os.path.join(masks_path, mask_file)
        img_path = os.path.join(images_path, mask_file)

        if not os.path.exists(img_path):
            print(f"VAROVÁNÍ: Obrázek nenalezen pro masku '{mask_file}'")
            continue

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        img = cv.imread(img_path)

        if mask is None or img is None:
            print(f"VAROVÁNÍ: Nelze načíst {mask_file}")
            continue

        data.append((mask, img, mask_file))

    return data


def evaluate_single_image(iou_instance, mask, img, img_name, parameters):
    """
    Vyhodnotí IoU pro jeden obrázek s danými parametry.

    Returns:
        float: IoU hodnota
    """
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Aplikace segmentačního algoritmu
    img_binary, inner_contours_mask = iou_instance.apply_segmentation_algorithm(
        iou_instance.algorithm, parameters, img_gray, img, img_name,
        inner_contours=iou_instance.inner_contours,
        detect_corrupted=iou_instance.detect_corrupted
    )

    if iou_instance.inner_contours and inner_contours_mask is not None:
        intersection = inner_contours_mask & img_binary
        img_binary = img_binary - intersection

    # Najít kontury a vytvořit masku
    contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    pred_mask = np.zeros_like(img_binary, dtype=np.uint8)
    if contours:
        for contour in contours:
            cv.drawContours(pred_mask, [contour], 0, color=255, thickness=-1)

    # Výpočet IoU - použít stejnou logiku jako f.IoU() v Funkce.py
    # np.logical_and interpretuje nenulové hodnoty jako True
    intersection = np.logical_and(mask, pred_mask).sum()
    union = np.logical_or(mask, pred_mask).sum()

    if union == 0:
        return 0.0

    return intersection / union


def run_cv_for_algorithm(algorithm, results_dict):
    """
    Spustí 5-fold cross-validaci pro daný algoritmus.

    Args:
        algorithm: Název algoritmu ("Gaussian", "Sauvola", "Niblack")
        results_dict: Dictionary pro ukládání výsledků {(project, name): iou}
    """
    print(f"\n{'='*60}")
    print(f"Algoritmus: {algorithm}")
    print(f"{'='*60}")

    for fold_idx in range(1, 6):
        fold_path = os.path.join(CV_DATA_PATH, f"fold_{fold_idx}")
        print(f"\n--- Fold {fold_idx}/5 ---")

        # 1. Načíst train data
        print("Načítám train data...")
        train_data = load_data_from_fold(fold_path, "train")
        print(f"  Načteno {len(train_data)} train obrázků")

        # 2. Optimalizace na train datech
        print(f"Spouštím Bayesian optimalizaci ({NUM_ITERATIONS} iterací)...")

        # Vytvořit IoU factory funkci
        def iou_factory(output_addr, projekt, alg, inner_contours=False, detect_corrupted=True):
            return IoU(output_addr, projekt, alg, inner_contours, detect_corrupted)

        optimizer = BayesianOptimizer(
            annotation_data=train_data,
            outputAddress=OUTPUT_DIR,
            projekt=f"cv_fold_{fold_idx}",
            algorithm=algorithm,
            learning_rate=0.01,  # Nepoužívá se v BO
            num_iterations=NUM_ITERATIONS,
            delta=0,  # Disabled early stopping (stejně jako benchmark)
            batch_size=BATCH_SIZE,
            f=iou_factory,
            progress_window=None,
            inner_contours=INNER_CONTOURS,
            detect_corrupted=DETECT_CORRUPTED
        )

        best_params, best_iou = optimizer.run()
        print(f"  Nejlepší IoU na train: {best_iou*100:.2f}%")
        print(f"  Parametry: {best_params}")

        # 3. Evaluace na test datech
        print("Evaluuji na test datech...")
        test_data = load_data_from_fold(fold_path, "test")
        print(f"  Načteno {len(test_data)} test obrázků")

        # Vytvořit IoU instance pro evaluaci
        iou_evaluator = IoU(
            OUTPUT_DIR,
            f"cv_fold_{fold_idx}_test",
            algorithm,
            inner_contours=INNER_CONTOURS,
            detect_corrupted=DETECT_CORRUPTED
        )

        for mask, img, filename in test_data:
            iou_value = evaluate_single_image(iou_evaluator, mask, img, filename, best_params)
            project, name = parse_filename(filename)
            results_dict[(project, name)] = iou_value
            print(f"    {filename}: IoU = {iou_value*100:.2f}%")

        # Průměr pro fold
        fold_ious = [evaluate_single_image(iou_evaluator, m, i, n, best_params)
                     for m, i, n in test_data]
        print(f"  Průměrné IoU na test fold {fold_idx}: {np.mean(fold_ious)*100:.2f}%")


def update_excel(results_gaussian, results_sauvola, results_niblack):
    """
    Aktualizuje Excel soubor s novými sloupci pro Bayesian optimization výsledky.
    """
    print(f"\n{'='*60}")
    print("Aktualizuji Excel soubor...")
    print(f"{'='*60}")

    # Načíst existující Excel
    df = pd.read_excel(EXCEL_PATH, sheet_name="original")

    # Vytvořit nové sloupce
    df["Gaussian_BO"] = None
    df["Sauvola_BO"] = None
    df["Niblack_BO"] = None

    # Mapovat výsledky
    matched = 0
    for idx, row in df.iterrows():
        project = row["project"]
        name = row["name"]
        key = (project, name)

        if key in results_gaussian:
            df.at[idx, "Gaussian_BO"] = results_gaussian[key]
            matched += 1
        if key in results_sauvola:
            df.at[idx, "Sauvola_BO"] = results_sauvola[key]
        if key in results_niblack:
            df.at[idx, "Niblack_BO"] = results_niblack[key]

    print(f"Mapováno {matched}/{len(df)} řádků")

    # Uložit - zachovat sheet "synthetic"
    with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name="original", index=False)

    # Výpis průměrů
    print(f"\nPrůměrné IoU:")
    print(f"  Gaussian_BO: {df['Gaussian_BO'].mean()*100:.2f}%")
    print(f"  Sauvola_BO:  {df['Sauvola_BO'].mean()*100:.2f}%")
    print(f"  Niblack_BO:  {df['Niblack_BO'].mean()*100:.2f}%")
    print(f"\nPorovnání s existujícími metodami:")
    print(f"  Gaussian (původní): {df['Gaussian'].mean()*100:.2f}%")
    print(f"  Sauvola (původní):  {df['Sauvola'].mean()*100:.2f}%")
    print(f"  Niblack (původní):  {df['Niblack'].mean()*100:.2f}%")


def main():
    print("="*60)
    print("5-Fold Cross-Validation pro Bayesian Optimizer")
    print("="*60)
    print(f"CV data: {CV_DATA_PATH}")
    print(f"Excel: {EXCEL_PATH}")
    print(f"Temp output: {OUTPUT_DIR}")
    print(f"Iterace: {NUM_ITERATIONS}")
    print(f"Algoritmy: {ALGORITHMS}")

    # Výsledky pro každý algoritmus
    results_gaussian = {}
    results_sauvola = {}
    results_niblack = {}

    # Spustit CV pro každý algoritmus
    run_cv_for_algorithm("Gaussian", results_gaussian)
    run_cv_for_algorithm("Sauvola", results_sauvola)
    run_cv_for_algorithm("Niblack", results_niblack)

    # Aktualizovat Excel
    update_excel(results_gaussian, results_sauvola, results_niblack)

    print("\n" + "="*60)
    print("HOTOVO!")
    print("="*60)


if __name__ == "__main__":
    main()
