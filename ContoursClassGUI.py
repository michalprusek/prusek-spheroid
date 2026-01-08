import os
import numpy as np
import cv2 as cv
from skimage.filters import threshold_sauvola, threshold_niblack
from skimage import img_as_ubyte
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
# Flexibilní importy - podpora jak package, tak lokálního použití
try:
    from prusek_spheroid_bayesian import Funkce as f
    from prusek_spheroid_bayesian import characteristic_functions as cf
    from prusek_spheroid_bayesian import gpu_utils as gpu
except ImportError:
    import Funkce as f
    import characteristic_functions as cf
    import gpu_utils as gpu
import json
import sys
import zipfile
from skimage import feature
import pandas as pd
import shutil

# Initialize GPU at module load (force CPU for headless/non-CUDA environments)
import os
_FORCE_CPU = os.environ.get('FORCE_CPU', 'false').lower() == 'true'
gpu.init_gpu(force_cpu=_FORCE_CPU)


def find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours):
    # Vytvoření výsledné masky pro průnik s Cannyho detektorem
    result_mask = np.zeros_like(img_binary, dtype=np.uint8)

    for contour in filtered_contours:
        # Vyplnění kontury
        filled_contour = cv.fillPoly(np.zeros_like(img_binary), [contour], 1)

        # Průnik s Cannyho detektorem
        intersection = filled_contour & edges

        # Kontrola neprázdného průniku a přidání do výsledné masky
        if np.any(intersection):
            result_mask = result_mask | filled_contour

    if inner_contours:
        inner_contours_mask = np.zeros_like(img_binary, dtype=np.uint8)
        for i in range(len(contours)):
            if hierarchy[0, i, 3] != -1:
                inner_filled_contour = cv.fillPoly(np.zeros_like(img_binary), [contours[i]], 1)

                inner_contours_mask = cv.bitwise_or(inner_filled_contour, inner_contours_mask)

        inner_contours_mask = f.Erosion(inner_contours_mask, 3, 1)

        return np.clip(result_mask, 0, 1), np.clip(inner_contours_mask, 0, 1)

    return np.clip(result_mask, 0, 1), None


def create_binary_mask(img_gray, threshold, dilation_size, erosion_size=None):
    img_binary = img_as_ubyte(img_gray > threshold)
    img_binary = np.invert(img_binary)
    if erosion_size is not None:
        img_binary = f.Erosion(img_binary, erosion_size, 1)
    img_binary = f.Dilation(img_binary, dilation_size, 1)
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
        # Kontrola, zda kontura není na okraji obrazu
        if detect_corrupted:
            if not (np.any(contour[:, :, 0] == 0) or np.any(contour[:, :, 1] == 0) or
                    np.any(contour[:, :, 0] == width - 1) or np.any(contour[:, :, 1] == height - 1)):
                if cv.contourArea(contour) >= min_area:
                    # Přidání kontury do filtrovaných kontur
                    filtered_contours.append(contour)
        else:
            if cv.contourArea(contour) >= min_area:
                # Přidání kontury do filtrovaných kontur
                filtered_contours.append(contour)

    return filtered_contours


def check_window_size(window_size):
    window_size = max(3, window_size)  # OpenCV requires blockSize > 1
    return window_size + 1 if window_size % 2 == 0 else window_size


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


def findContours(img_binary, inner_contours):
    if inner_contours:
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy


class BaseImageProcessing:
    def apply_segmentation_algorithm(self, algorithm, parameters, img, img_rgb, img_name, inner_contours,
                                     detect_corrupted):
        if algorithm == "Sauvola":
            return self.sauvola(parameters, img, img_name, inner_contours, detect_corrupted)
        elif algorithm == "Niblack":
            return self.niblack(parameters, img, img_name, inner_contours, detect_corrupted)
        elif algorithm == "Gaussian":
            return self.gaussian_adaptive(parameters, img, img_name, inner_contours, detect_corrupted)
        else:
            print(f"Algoritmus s názvem {algorithm} nenalezen.")
            sys.exit(1)

    @staticmethod
    def sauvola(parameters, img_gray, img_name, inner_contours, detect_corrupted):
        window_size = check_window_size(int(parameters["window_size"]))
        std_k = parameters["std_k"]
        min_area = parameters["min_area"]
        dilation_size = int(parameters["dilation_size"])
        sigma = parameters["sigma"]

        thresh_sauvola = threshold_sauvola(img_gray, window_size=window_size)
        img_binary = create_binary_mask(img_gray, thresh_sauvola, dilation_size)
        edges = calculate_canny_edges(img_gray, std_k, sigma)

        contours, hierarchy = findContours(img_binary, inner_contours)
        filtered_contours = filter_contours(contours, img_gray.shape, min_area,
                                            detect_corrupted)

        return find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours)

    @staticmethod
    def niblack(parameters, img_gray, img_name, inner_contours, detect_corrupted):
        window_size = check_window_size(int(parameters["window_size"]))
        k = parameters["k"]
        min_area = parameters["min_area"]
        std_k = parameters["std_k"]
        dilation_size = int(parameters["dilation_size"])
        sigma = parameters["sigma"]

        thresh_niblack = threshold_niblack(img_gray, window_size=window_size, k=k)
        img_binary = create_binary_mask(img_gray, thresh_niblack, dilation_size, 1)

        edges = calculate_canny_edges(img_gray, std_k, sigma)

        contours, hierarchy = findContours(img_binary, inner_contours)
        filtered_contours = filter_contours(contours, img_gray.shape, min_area,
                                            detect_corrupted)

        return find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours)

    @staticmethod
    def gaussian_adaptive(parameters, img_gray, img_name, inner_contours, detect_corrupted):
        window_size = check_window_size(int(parameters["window_size"]))
        min_area = parameters["min_area"]
        std_k = parameters["std_k"]
        dilation_size = int(parameters["dilation_size"])
        sigma = parameters["sigma"]

        img_binary = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, window_size, 0)

        img_binary = f.Erosion(img_binary,1,1)

        img_binary = f.Dilation(img_binary, dilation_size, 1)

        edges = calculate_canny_edges(img_gray, std_k, sigma)

        contours, hierarchy = findContours(img_binary, inner_contours)

        filtered_contours = filter_contours(contours, img_gray.shape, min_area,
                                            detect_corrupted)

        result = find_intersection(img_binary, filtered_contours, contours, hierarchy, edges, inner_contours)

        return result

    @staticmethod
    def gaussian(x, mean, amplitude, standard_deviation):
        return amplitude * np.exp(- ((x - mean) ** 2 / (2 * standard_deviation ** 2)))

    @staticmethod
    def find_holes(img_gray, mask_img, parameters, img_name):
        holes_t = parameters["holes_t"]

        background_mask = cv.bitwise_not(mask_img)
        background = cv.bitwise_and(img_gray, img_gray, mask=background_mask)

        # Výpočet histogramu pro referenční oblast pozadí
        hist_background = cv.calcHist([background], [0], background_mask, [256], [0, 256])

        # Normalizace histogramu
        cv.normalize(hist_background, hist_background, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

        hist_background[0] = 0

        # Zpětná projekce histogramu
        back_project = cv.calcBackProject([img_gray], [0], hist_background, [0, 256], 1)

        # Aplikace zpětné projekce pouze na oblast sféroidu
        spheroid_back_project = cv.bitwise_and(back_project, back_project, mask=mask_img)

        # Prahování s použitím Otsuovy metody
        _, img_binary = cv.threshold(spheroid_back_project, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        img_binary = f.Dilation(f.Erosion(img_binary, 3, 1), 3, 1)

        return img_binary, spheroid_back_project


class Contours(BaseImageProcessing):
    def __init__(self, adresaDatasetu, adresa_output, projekt, algorithm, parameters, show_img, function,
                 inner_contours, detect_corrupted, create_json, calculate_properties,
                 progress_window=None):
        super().__init__()
        self.adresaDatasetu = adresaDatasetu
        self.output_json_path = f"{adresa_output}/{projekt}/CVAT/{algorithm}/annotations/instances_default.json"
        self.output_images_path = f"{adresa_output}/{projekt}/CVAT/{algorithm}/images"
        self.output_segmented_path = f"{adresa_output}/{projekt}/segmented_images/{algorithm}"
        self.zipfile_address = f"{adresa_output}/{projekt}/CVAT/{algorithm}"
        self.excel_address = f"{adresa_output}/{projekt}"
        self.coco_data = f.initialize_coco_data()
        self.show_img = show_img
        self.projekt = projekt
        self.algorithm = algorithm
        self.parameters = parameters
        self.inner_contours = inner_contours
        self.detect_corrupted = detect_corrupted
        self.create_json = create_json
        self.calculate_properties = calculate_properties
        self.f = function
        self.counter = 1
        self.progress_window = progress_window

        create_directory(os.path.dirname(self.output_json_path))
        create_directory(self.output_images_path)
        create_directory(f"{self.output_segmented_path}/masks", delete=True)
        create_directory(f"{self.output_segmented_path}/results", delete=True)

    def run(self):
        all_contour_data = []
        filenames = os.listdir(self.adresaDatasetu)
        total_files = len(filenames)
        print(f"loaded {total_files} dataset images")
        for filename in os.listdir(self.adresaDatasetu):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')):
                img = cv.imread(os.path.join(self.adresaDatasetu,filename))

                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                img_binary, inner_contours_mask = self.apply_segmentation_algorithm(self.algorithm, self.parameters,
                                                                                    img_gray, img,
                                                                                    filename.replace(".bmp", ".png"),
                                                                                    self.inner_contours,
                                                                                    self.detect_corrupted)

                if self.inner_contours:
                    intersection = inner_contours_mask & img_binary
                    img_binary = img_binary - intersection

                contours, hierarchy = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                height, width = np.shape(img_binary)

                if not contours:
                    cv.line(img, (0, 0), (width - 1, height - 1), (0, 0, 255), 5)
                    cv.line(img, (0, height - 1), (width - 1, 0), (0, 0, 255), 5)
                else:
                    if self.create_json:
                        cv.imwrite(f"{self.output_images_path}/{filename}", img)

                    inner_contours = []
                    outer_contours = []
                    if self.inner_contours:

                        for i, contour in enumerate(contours):
                            # Získání indexu rodiče pro aktuální konturu
                            parent_index = hierarchy[0][i][3]

                            if parent_index != -1:
                                # Kontrola, zda má rodič této kontury také rodiče (kontury druhého řádu)
                                grandparent_index = hierarchy[0][parent_index][3]
                                if grandparent_index == -1:
                                    # Kontura je prvního řádu (má rodiče, ale nemá dědečka)
                                    cv.drawContours(img, [contour], -1, [255, 0, 0], 2)
                                    inner_contours.append(contour)
                            else:
                                outer_contours.append(contour)

                    if len(outer_contours) == 0:
                        outer_contours = contours

                    for index, contour in enumerate(outer_contours):
                        cv.drawContours(img, [contour], -1, [0, 0, 255], 2)

                        if self.calculate_properties:
                            contour_data = {
                                'MaskName': os.path.basename(filename),
                                'ContourOrder': index + 1
                            }

                            additional_data = cf.calculate_all(contour)
                            contour_data.update(additional_data)

                            all_contour_data.append(contour_data)

                    if self.create_json:
                        self.coco_data = f.convert_contours_to_coco(outer_contours, inner_contours, height, width,
                                                                    filename,
                                                                    self.counter,
                                                                    self.coco_data)

                if cv.imwrite(f"{self.output_segmented_path}/results/{filename.replace('.bmp', '.png')}", img):
                    print(f"image {filename.replace('.bmp', '.png')} saved")
                if cv.imwrite(f"{self.output_segmented_path}/masks/{filename.replace('.bmp', '.png')}",
                           img_binary * 255):
                    print(f"mask {filename.replace('.bmp', '.png')} saved")

                if self.progress_window:
                    progress_text = f"{self.counter}/{total_files}"
                    self.progress_window.update_progress(progress_text)
                    self.counter += 1

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
            zip_folder(self.zipfile_address, f"{self.zipfile_address}.zip")

        if self.progress_window:
            self.progress_window.update_progress("FINISHED")


class IoU(BaseImageProcessing):
    def __init__(self, adresa_output, projekt, algorithm, inner_contours,
                 detect_corrupted):
        super().__init__()
        self.adresa_output = f"{adresa_output}/{projekt}/IoU"
        self.adresa_plots = f"{adresa_output}/{projekt}/IoU/plots"
        self.projekt = projekt
        self.algorithm = algorithm
        self.inner_contours = inner_contours
        self.detect_corrupted = detect_corrupted

        create_directory(self.adresa_output)
        create_directory(self.adresa_plots)

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
            self.algorithm, parameters, img_gray, img, img_name,
            inner_contours=self.inner_contours, detect_corrupted=self.detect_corrupted)

        # Další zpracování a výpočet IoU, TPR, PPV
        if self.inner_contours:
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

        iou, tpr, ppv = f.IoU(self.projekt, self.algorithm, ref_mask, mask, img_name, save=save,
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
                "inner_contours": self.inner_contours,
                "detect_corrupted": self.detect_corrupted
            }

            return json_data
        return averageIoU

    def save_parameters_json(self, averageIoU, json_data_list):
        json_data = average_json_data(json_data_list)
        json_data.update({
                "method": self.algorithm,
                "inner_contours": self.inner_contours,
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
        inner_contours_string = "WITH_inner_contours" if self.inner_contours else "WITHOUT_inner_contours"
        detect_corrupted_string = "WITH_detecting_corrupted" if self.detect_corrupted else "WITHOUT_detecting_corrupted"

        with open(
                f"{self.adresa_output}/results_{self.projekt}_{self.algorithm}_IoU_{round(averageIoU * 100, 2)}_{inner_contours_string}_{detect_corrupted_string}.json",
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