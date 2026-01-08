import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import math
# Flexibilní import gpu_utils
try:
    from prusek_spheroid_bayesian import gpu_utils as gpu
except ImportError:
    import gpu_utils as gpu

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.callbacks import DeltaYStopper

# Volitelný import tkinter (pro headless prostředí)
try:
    import tkinter as tk
    from tkinter import messagebox
    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False

# Set random seeds for reproducibility (same as GD)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # data by měla být seznamem trojic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mask, image, name = self.data[idx]
        return mask, image, name


class BayesianOptimizer:
    """
    Bayesian Optimization pro hyperparametr tuning segmentace sféroidů.
    Používá Gaussian Process jako surrogate model a Expected Improvement
    jako acquisition function.

    Nahrazuje GradientDescent třídu - místo Adam optimizeru s finite differences
    používá globální Bayesian optimization, která je efektivnější pro
    nalezení globálního optima a lépe pracuje s diskrétními parametry.
    """

    def __init__(self, annotation_data, outputAddress, projekt, algorithm, learning_rate,
                 num_iterations, delta, batch_size, f, progress_window=None, inner_contours=False,
                 detect_corrupted=True):
        """
        Inicializace Bayesian optimizeru.

        Args:
            annotation_data: Data pro trénink (masky, obrázky, jména)
            outputAddress: Výstupní adresář
            projekt: Název projektu
            algorithm: Algoritmus segmentace ("Sauvola", "Niblack", "Gaussian")
            learning_rate: Nepoužívá se v BO, zachováno pro kompatibilitu
            num_iterations: Počet iterací (n_calls pro gp_minimize)
            delta: Práh pro early stopping
            batch_size: Velikost batche
            f: Factory funkce pro vytvoření IoU instance
            progress_window: GUI progress window
            inner_contours: Zda detekovat vnitřní kontury
            detect_corrupted: Zda detekovat poškozené kontury
        """
        self.projekt = projekt
        self.algorithm = algorithm
        self.progress_window = progress_window

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(annotation_data) / batch_size)

        dataset = MyDataset(annotation_data)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # shuffle=False pro konzistentní evaluaci v BO

        self.outputAddress = outputAddress

        # Definice parametrů a jejich rozsahů
        self.cont_param_ranges = {}
        if self.algorithm in {"Sauvola", "Niblack", "Gaussian"}:
            self.cont_param_ranges["window_size"] = [1, 2001]

        if self.algorithm == "Niblack":
            self.cont_param_ranges["k"] = [0.0, 0.35]

        self.cont_param_ranges.update({
            "min_area": [0, 50000],
            "sigma": [0.0, 5.0],
            "std_k": [0.0, 3.0]
        })

        self.disc_param_ranges = {"dilation_size": [0, 10]}

        # Výchozí hodnoty pro inicializaci
        self.default_params = {}
        if self.algorithm in {"Sauvola", "Niblack", "Gaussian"}:
            self.default_params["window_size"] = 800
        if self.algorithm == "Niblack":
            self.default_params["k"] = 0.2
        self.default_params.update({
            "min_area": 5000,
            "sigma": 1.0,
            "std_k": 0.5,
            "dilation_size": 2
        })

        self.num_iterations = num_iterations
        self.delta = delta
        self.f = f

        self.instance = self.f(self.outputAddress, self.projekt,
                               self.algorithm, inner_contours=inner_contours,
                               detect_corrupted=detect_corrupted)

        # Pro tracking průběhu optimalizace
        self.iteration_count = 0
        self.best_iou = 0
        self.start_time = None
        self.parameters_history = []
        self.iou_history = []

    @staticmethod
    def show_error_message(message):
        if HAS_TKINTER:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error", message)
            root.destroy()
        else:
            print(f"ERROR: {message}")

    def create_search_space(self):
        """
        Vytvoří search space pro scikit-optimize.
        Kombinuje continuous (Real) a discrete (Integer) parametry.
        """
        dimensions = []
        self.param_names = []

        # Continuous parametry
        if self.algorithm in {"Sauvola", "Niblack", "Gaussian"}:
            dimensions.append(Real(3, 2001, name='window_size'))  # Min 3 for OpenCV
            self.param_names.append('window_size')

        if self.algorithm == "Niblack":
            dimensions.append(Real(0.0, 0.35, name='k'))
            self.param_names.append('k')

        dimensions.extend([
            Real(0, 50000, name='min_area'),
            Real(0.0, 5.0, name='sigma'),
            Real(0.0, 3.0, name='std_k'),
        ])
        self.param_names.extend(['min_area', 'sigma', 'std_k'])

        # Discrete parametr - Integer space umožňuje GP správně modelovat
        dimensions.append(Integer(0, 10, name='dilation_size'))
        self.param_names.append('dilation_size')

        return dimensions

    def params_list_to_dict(self, params_list):
        """Převede seznam parametrů z optimizeru na dictionary."""
        return {name: value for name, value in zip(self.param_names, params_list)}

    def params_dict_to_list(self, params_dict):
        """Převede dictionary parametrů na seznam pro optimizer."""
        return [params_dict[name] for name in self.param_names]

    def objective(self, params_list):
        """
        Objektová funkce pro Bayesian optimization.
        Vrací NEGATIVNÍ IoU (protože gp_minimize minimalizuje).

        Args:
            params_list: Seznam parametrů v pořadí definovaném v create_search_space()

        Returns:
            float: Negativní průměrné IoU přes všechny batche
        """
        parameters = self.params_list_to_dict(params_list)

        # Průměr přes všechny batche
        iou_values = []
        for batch in self.data_loader:
            iou = self.instance.run(batch, parameters, False)
            iou_values.append(iou)

        avg_iou = np.mean(iou_values)

        # Uložit do historie
        self.parameters_history.append(parameters.copy())
        self.iou_history.append(avg_iou)

        return -avg_iou  # Negativní, protože minimalizujeme

    def progress_callback(self, res):
        """
        Callback pro aktualizaci progress baru a logování.

        Args:
            res: OptimizeResult objekt z scikit-optimize
        """
        self.iteration_count += 1
        current_iou = -res.func_vals[-1]  # Poslední hodnota (negativní zpět)
        best_iou_so_far = -res.fun

        elapsed_time = time.time() - self.start_time
        avg_time_per_iter = elapsed_time / self.iteration_count
        remaining_iters = self.num_iterations - self.iteration_count
        estimated_remaining = avg_time_per_iter * remaining_iters

        # Získat aktuální parametry
        current_params = self.params_list_to_dict(res.x_iters[-1])
        rounded_params = {k: round(v, 3) if isinstance(v, float) else v
                         for k, v in current_params.items()}

        if self.progress_window:
            self.progress_window.update_info(
                self.projekt,
                self.algorithm,
                self.iteration_count,
                round(best_iou_so_far * 100, 2),
                round(estimated_remaining),
                f"BO iter {self.iteration_count}/{self.num_iterations}"
            )

        print(f"Project: {self.projekt}, Algorithm: {self.algorithm}, "
              f"Iteration: {self.iteration_count}/{self.num_iterations}, "
              f"Current IoU: {round(current_iou * 100, 2)}%, "
              f"Best IoU: {round(best_iou_so_far * 100, 2)}%, "
              f"Parameters: {rounded_params}")

        # Early stopping disabled for benchmark
        # if len(res.func_vals) > 20:
        #     recent_best = min(res.func_vals[-20:])
        #     older_best = min(res.func_vals[:-20]) if len(res.func_vals) > 20 else res.func_vals[0]
        #     recent_improvement = abs(recent_best - older_best)
        #     if recent_improvement < self.delta:
        #         print(f"Early stopping: improvement {recent_improvement:.6f} < delta {self.delta}")
        #         return True

        return False

    def get_initial_point(self):
        """Vrátí výchozí bod pro optimalizaci (x0)."""
        return self.params_dict_to_list(self.default_params)

    def bayesian_optimize(self):
        """
        Hlavní Bayesian optimization smyčka.

        Používá:
        - Gaussian Process jako surrogate model
        - Expected Improvement (EI) jako acquisition function
        - Integer space pro diskrétní parametry (lepší než hill climbing)

        Returns:
            tuple: (nejlepší_parametry_dict, nejlepší_iou)
        """
        self.start_time = time.time()
        self.iteration_count = 0
        self.parameters_history = []
        self.iou_history = []

        dimensions = self.create_search_space()
        x0 = self.get_initial_point()

        # Počet počátečních náhodných bodů
        n_initial = min(10, self.num_iterations // 3)

        print(f"\nStarting Bayesian Optimization for {self.algorithm}")
        print(f"Search space dimensions: {len(dimensions)}")
        print(f"Parameters: {self.param_names}")
        print(f"Initial point: {x0}")
        print(f"n_calls: {self.num_iterations}, n_initial_points: {n_initial}\n")

        try:
            result = gp_minimize(
                func=self.objective,
                dimensions=dimensions,
                acq_func='EI',  # Expected Improvement
                n_calls=self.num_iterations,
                n_initial_points=n_initial,
                x0=x0,  # Začít od výchozích hodnot
                random_state=42,
                callback=self.progress_callback,
                n_jobs=1,  # Sekvenční evaluace (objective není thread-safe)
                verbose=False
            )
        except StopIteration:
            # Early stopping byl aktivován
            pass

        # Najít nejlepší parametry z historie
        best_idx = np.argmax(self.iou_history)
        best_params = self.parameters_history[best_idx]
        best_iou = self.iou_history[best_idx]

        total_time = time.time() - self.start_time

        print(f"\nBayesian Optimization completed in {round(total_time)} seconds")
        print(f"Best IoU: {round(best_iou * 100, 2)}%")
        print(f"Best parameters: {best_params}")

        # Finální run s nejlepšími parametry a uložení výsledků
        json_data_list = []
        for batch in self.data_loader:
            json_data_list.append(self.instance.run(batch, best_params, True))

        self.instance.save_parameters_json(best_iou, json_data_list)

        if self.progress_window:
            self.progress_window.update_info(
                self.projekt,
                self.algorithm,
                self.iteration_count,
                round(best_iou * 100, 2),
                0,
                "FINISHED"
            )

        print(f"Optimization done. IoU: {round(best_iou * 100, 2)}%")

        return best_params, best_iou

    def run(self):
        """
        Spustí Bayesian optimization.

        Returns:
            tuple: (nejlepší_parametry_dict, nejlepší_iou)
        """
        print(f"Project: {self.projekt}, Algorithm: {self.algorithm}, "
              f"Starting Bayesian Optimization with {self.num_iterations} iterations")

        parameters, iou = self.bayesian_optimize()

        return parameters, iou
