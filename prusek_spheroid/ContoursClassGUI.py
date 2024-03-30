import os
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
import pandas as pd
from prusek_spheroid import file_management as fm
from prusek_spheroid import selection_dialog as sd
from prusek_spheroid import calculate_iou as metric
from prusek_spheroid.methods import BaseImageProcessing
from prusek_spheroid import characteristic_functions as cf


class Contours(BaseImageProcessing):

    def __init__(self, master, adresaDatasetu, adresa_output, projekt, algorithm, parameters, show_img, function,
                 contours_state, detect_corrupted, create_json, calculate_properties,
                 progress_window=None):
        super().__init__()
        self.master = master
        self.user_decision_lock = Lock()
        self.adresaDatasetu = adresaDatasetu
        self.output_json_path = f"{adresa_output}/{projekt}/CVAT/{algorithm}/annotations/instances_default.json"
        self.output_images_path = f"{adresa_output}/{projekt}/CVAT/{algorithm}/images"
        self.output_segmented_path = f"{adresa_output}/{projekt}/segmented_images/{algorithm}"
        self.zipfile_address = f"{adresa_output}/{projekt}/CVAT/{algorithm}"
        self.excel_address = f"{adresa_output}/{projekt}"
        self.coco_data = fm.initialize_coco_data()
        self.show_img = show_img
        self.projekt = projekt
        self.algorithm = algorithm
        self.parameters = parameters
        self.contours_state = contours_state
        self.detect_corrupted = detect_corrupted
        self.create_json = create_json
        self.calculate_properties = calculate_properties
        self.f = function
        self.progress_window = progress_window

        fm.create_directory(os.path.dirname(self.output_json_path), delete=True)
        fm.create_directory(self.output_images_path, delete=True)
        fm.create_directory(f"{self.output_segmented_path}/masks", delete=True)
        fm.create_directory(f"{self.output_segmented_path}/results", delete=True)

    def run(self):
        all_contour_data = []
        filenames = os.listdir(self.adresaDatasetu)
        total_files = len(filenames)
        print(f"loaded {total_files} dataset images")
        self.counter = 1

        if self.contours_state == "select":
            dialog = sd.SelectionDialog(self.master, self.counter, total_files, self.user_decision_lock)

        for filename in os.listdir(self.adresaDatasetu):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                self.user_decision_lock.acquire()

                img = cv.imread(os.path.join(self.adresaDatasetu, filename))

                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                img_binary, inner_contours_mask = self.apply_segmentation_algorithm(self.algorithm, self.parameters,
                                                                                    img_gray,
                                                                                    self.contours_state,
                                                                                    self.detect_corrupted)

                if self.contours_state == "no" or self.contours_state == "select":
                    mask_without = img_binary * 255
                else:
                    mask_without = None
                if self.contours_state == "all" or self.contours_state == "select":
                    intersection = inner_contours_mask & img_binary
                    img_binary = img_binary - intersection
                    mask_with = img_binary * 255
                else:
                    mask_with = None

                contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                height, width = np.shape(img_binary)

                if self.create_json:
                    if not cv.imwrite(f"{self.output_images_path}/{filename}", img):
                        print(f"FAILED to save image: {self.output_images_path}/{filename}")

                if not contours:
                    if self.create_json:
                        self.coco_data = fm.convert_contours_to_coco([], [], height, width,
                                                                     filename,
                                                                     self.counter,
                                                                     self.coco_data)
                    cv.line(img, (0, 0), (width - 1, height - 1), (0, 0, 255), 5)
                    cv.line(img, (0, height - 1), (width - 1, 0), (0, 0, 255), 5)
                else:

                    inner_contours = []
                    outer_contours = []
                    img_without = img.copy()
                    if self.contours_state == "all" or self.contours_state == "select":
                        img_with = img.copy()

                        for i, contour in enumerate(contours):
                            # Získání indexu rodiče pro aktuální konturu
                            parent_index = hierarchy[0][i][3]

                            if parent_index != -1:
                                # Kontrola, zda má rodič této kontury také rodiče (kontury druhého řádu)
                                grandparent_index = hierarchy[0][parent_index][3]
                                if grandparent_index == -1:
                                    # Kontura je prvního řádu (má rodiče, ale nemá dědečka)
                                    cv.drawContours(img_with, [contour], -1, [255, 0, 0], 2)
                                    inner_contours.append(contour)
                            else:
                                cv.drawContours(img_with, [contour], -1, [0, 0, 255], 2)
                                outer_contours.append(contour)

                    if len(outer_contours) == 0:
                        outer_contours = contours

                    for index, contour in enumerate(outer_contours):
                        if not self.contours_state == "all":
                            cv.drawContours(img_without, [contour], -1, [0, 0, 255], 2)

                        if self.calculate_properties:
                            contour_data = {
                                'MaskName': os.path.basename(filename),
                                'ContourOrder': index + 1
                            }

                            additional_data = cf.calculate_all(contour)
                            contour_data.update(additional_data)

                            all_contour_data.append(contour_data)

                    if self.create_json:
                        self.coco_data = fm.convert_contours_to_coco(outer_contours, inner_contours, height, width,
                                                                     filename,
                                                                     self.counter,
                                                                     self.coco_data)

                if self.contours_state == "select":
                    dialog.update_selection_dialog(img_without, img_with, mask_without, mask_with,
                                                   f"{self.output_segmented_path}/results/{filename.replace('.bmp', '.png')}",
                                                   f"{self.output_segmented_path}/masks/{filename.replace('.bmp', '.png')}", self.counter)
                elif self.contours_state == "all":
                    if not cv.imwrite(f"{self.output_segmented_path}/masks/{filename.replace('.bmp', '.png')}",
                                      mask_with):
                        print(
                            f"FAILED to save mask: {self.output_segmented_path}/masks/{filename.replace('.bmp', '.png')}")
                    if not cv.imwrite(f"{self.output_segmented_path}/results/{filename.replace('.bmp', '.png')}",
                                      img_with):
                        print(
                            f"FAILED to save image: {self.output_segmented_path}/results/{filename.replace('.bmp', '.png')}")
                else:
                    if not cv.imwrite(f"{self.output_segmented_path}/masks/{filename.replace('.bmp', '.png')}",
                                      mask_without):
                        print(
                            f"FAILED to save mask: {self.output_segmented_path}/masks/{filename.replace('.bmp', '.png')}")
                    if not cv.imwrite(f"{self.output_segmented_path}/results/{filename.replace('.bmp', '.png')}",
                                      img_without):
                        print(
                            f"FAILED to save image: {self.output_segmented_path}/results/{filename.replace('.bmp', '.png')}")

                if self.progress_window:
                    progress_text = f"{self.counter}/{total_files}"
                    self.progress_window.update_progress(progress_text)

                self.counter += 1

        if dialog:
            dialog.destroy_dialog()

        if self.progress_window:
            self.progress_window.update_progress("dumping...")

        if self.calculate_properties:
            all_contour_data.sort(key=lambda x: x['MaskName'])
            df = pd.DataFrame(all_contour_data, columns=[
                'MaskName', 'ContourOrder', 'Area', 'Circularity', 'Compactness', 'Convexity',
                'EquivalentDiameter', 'FeretAspectRatio', 'FeretDiameterMax',
                'FeretDiameterMaxOrthogonalDistance', 'FeretDiameterMin',
                'LengthMajorDiameterThroughCentroid', 'LengthMinorDiameterThroughCentroid',
                'Perimeter', 'Solidity', 'Sphericity'
            ])
            df.to_excel(f"{self.excel_address}/contour_properties.xlsx")

        if self.create_json:
            with open(self.output_json_path, "w") as json_file:
                json.dump(self.coco_data, json_file)
            if self.progress_window:
                self.progress_window.update_progress("zipping folder...")
            fm.zip_folder(self.zipfile_address, f"{self.zipfile_address}.zip")

        if self.progress_window:
            self.progress_window.update_progress("FINISHED")


class IoU(BaseImageProcessing):
    def __init__(self, adresa_output, project, algorithm, contours_state,
                 detect_corrupted):
        super().__init__()
        self.project = project
        self.algorithm = algorithm
        self.contours_state = contours_state
        self.detect_corrupted = detect_corrupted
        self.adresa_output = f"{adresa_output}/{project}/IoU"
        self.adresa_plots = f"{adresa_output}/{project}/IoU/plots/{self.algorithm}"

        fm.create_directory(self.adresa_output)
        fm.create_directory(self.adresa_plots, delete=True)

        self.plot_lock = Lock()

        # for mask, name in zip(self.masks, self.img_names):
        #    cv.imwrite(f"img_print/{name}",mask)

    def process_and_compute_iou(self, ref_mask, img, img_name, parameters, save, lock):
        # Převod tensoru masky a obrázku na numpy pole
        ref_mask = ref_mask.numpy()
        img = img.numpy()

        # Zpracování obrazu (předpokládá se, že img je ve formátu BGR)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Aplikace algoritmu segmentace
        img_binary, inner_contours_mask = self.apply_segmentation_algorithm(
            self.algorithm, parameters, img_gray, inner_contours=self.contours_state,
            detect_corrupted=self.detect_corrupted)

        # Další zpracování a výpočet IoU, TPR, PPV
        if self.contours_state == "all" or self.contours_state == "select":
            intersection = inner_contours_mask & img_binary
            img_binary = img_binary - intersection

        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        mask = np.zeros_like(img_binary, dtype=np.uint8)
        if not contours:
            contour = np.array([[0, 0]], dtype=np.int32)
            cv.drawContours(mask, [contour], 0, color=255, thickness=-1)
        else:
            for contour in contours:
                cv.drawContours(mask, [contour], 0, color=255, thickness=-1)

        iou, tpr, ppv = metric.IoU(self.project, self.algorithm, ref_mask, mask, img_name, save=save,
                                   lock=lock, address=self.adresa_plots)

        return iou, tpr, ppv

    def run(self, batch, parameters, save_txt):
        IoUbuffer = []
        ratesBuffer = []

        lock = Lock()  # Create a Lock for thread-safe IoU calculations
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.process_and_compute_iou, ref_mask, img, img_name, parameters, save_txt, lock)
                for ref_mask, img, img_name in zip(*batch)]
            for future in futures:
                iou, tpr, ppv = future.result()
                IoUbuffer.append(iou)
                ratesBuffer.append([tpr, ppv])

        averageIoU = np.average(IoUbuffer)

        if save_txt:
            rounded_parameters = {key: round(value, 2) for key, value in parameters.items()}
            TPRs = [entry[0] for entry in ratesBuffer]
            PPVs = [entry[1] for entry in ratesBuffer]
            averageTPR = np.average(TPRs)
            averagePPV = np.average(PPVs)

            # Uložení do JSON souboru
            json_data = {
                "method": self.algorithm,
                "parameters": rounded_parameters,
                "averageIoU": round(averageIoU * 100, 2),
                "averageTPR": round(averageTPR * 100, 2),
                "averagePPV": round(averagePPV * 100, 2),
                "contours_state": self.contours_state,
                "detect_corrupted": self.detect_corrupted
            }

            return json_data
        return averageIoU

    def save_parameters_json(self, averageIoU, json_data_list):
        json_data = average_json_data(json_data_list)
        json_data.update({
            "method": self.algorithm,
            "contours_state": self.contours_state,
            "detect_corrupted": self.detect_corrupted
        })

        # NumpyEncoder pro správné uložení numpy dat
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(NumpyEncoder, self).default(obj)

        # Název souboru s výsledky
        if self.contours_state == "no":
            contours_state_string = "NO_HOLES"
        elif self.contours_state == "all":
            contours_state_string = "WITH_HOLES"
        else:
            contours_state_string = "SELECT_HOLES"

        detect_corrupted_string = "WITH_detecting_corrupted" if self.detect_corrupted else "WITHOUT_detecting_corrupted"

        with open(
                f"{self.adresa_output}/results_{self.project}_{self.algorithm}_IoU_{round(averageIoU * 100, 2)}_{contours_state_string}_{detect_corrupted_string}.json",
                "w") as json_file:
            json.dump(json_data, json_file, indent=4, cls=NumpyEncoder)


def average_json_data(json_data_list):
    # Inicializujeme prázdné seznamy pro jednotlivé hodnoty
    parameters_list = []

    # Projdeme všechny JSON data a přidáme jejich hodnoty do příslušných seznamů
    for json_data in json_data_list:
        if json_data:
            # Kontrola, zda jsou hodnoty v json_data ve správném formátu
            if isinstance(json_data["parameters"], dict):
                parameters_list.append(json_data["parameters"])

    # Pokud jsou všechna json_data prázdná nebo neobsahují správné hodnoty, vrátíme None
    if not parameters_list:
        return None

    # Zprůměrujeme hodnoty v seznamech
    averaged_parameters = {}
    for key in parameters_list[0].keys():
        averaged_parameters[key] = np.mean([param[key] for param in parameters_list])

    # Vytvoříme nový JSON objekt se zprůměrovanými hodnotami
    averaged_json_data = {
        "parameters": averaged_parameters,
    }

    return averaged_json_data
