import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import messagebox
import torch
from torch.utils.data import DataLoader
import math


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # data by měla být seznamem trojic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mask, image, name = self.data[idx]
        return mask, image, name


class GradientDescent:
    def __init__(self, annotation_data, outputAddress, projekt, algorithm, learning_rate,
                 num_iterations, delta, batch_size, f, progress_window=None, contours_state=None,
                 detect_corrupted=True):
        self.projekt = projekt
        self.algorithm = algorithm
        self.progress_window = progress_window

        self.batch_size = batch_size
        self.num_batches = math.ceil(len(annotation_data) / batch_size)

        dataset = MyDataset(annotation_data)
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.outputAddress = outputAddress

        self.cont_parameters = {}
        self.cont_param_ranges = {}
        # Define parameter dictionaries
        if self.algorithm in {"Sauvola", "Niblack", "Gaussian"}:
            self.cont_parameters.update({"window_size": 800})
            self.cont_param_ranges.update({"window_size": [1, 2001]})

        if self.algorithm == "Niblack":
            self.cont_parameters.update({"k": 0.2})
            self.cont_param_ranges.update({"k": [0.0, 0.35]})

        self.cont_parameters.update({"min_area": 5000, "sigma": 1.0, "std_k": 0.5})
        self.cont_param_ranges.update({"min_area": [0, 50000], "sigma": [0.0, 5.0], "std_k": [0.0, 3.0]})

        self.disc_parameters = {"dilation_size": 2}
        self.disc_param_ranges = {"dilation_size": [0, 10]}

        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.epsilon = 0.05
        self.delta = delta
        self.f = f

        self.instance = self.f(self.outputAddress, self.projekt,
                               self.algorithm, contours_state=contours_state, detect_corrupted=detect_corrupted)

        # Initialize cont_normalized_parameters
        self.cont_normalized_parameters = self.normalize_parameters(self.cont_parameters)

    @staticmethod
    def show_error_message(message):
        root = tk.Tk()
        root.withdraw()  # Skryje hlavní okno
        messagebox.showerror("Error", message)
        root.destroy()

    def evaluate_function(self, batch, parameters):
        cont_parameters = self.denormalize_parameters(parameters)
        disc_parameters = self.disc_parameters

        return self.instance.run(batch, {**cont_parameters, **disc_parameters}, False)

    def compute_gradient_parallel_batch_disc(self, batch):
        batch_gradient = {param_name: 0 for param_name in self.disc_parameters.keys()}

        f_original = self.instance.run(batch, {**self.cont_parameters, **self.disc_parameters}, False)

        for param_name in self.disc_parameters.keys():
            parameters = self.disc_parameters.copy()
            if parameters[param_name] >= self.disc_param_ranges[param_name][0] + 1:
                parameters[param_name] += -1
                f_minus_one = self.instance.run(batch, {**self.cont_parameters, **parameters}, False)
            else:
                f_minus_one = 0

            parameters = self.disc_parameters.copy()
            if parameters[param_name] <= self.disc_param_ranges[param_name][1] - 1:
                parameters[param_name] += 1
                f_plus_one = self.instance.run(batch, {**self.cont_parameters, **parameters}, False)
            else:
                f_plus_one = 0

            index = np.argmax([f_minus_one, f_original, f_plus_one])
            batch_gradient[param_name] += index - 1

        averaged_batch_gradient = {param_name: grad / self.num_batches for param_name, grad in batch_gradient.items()}
        return averaged_batch_gradient

    def compute_gradient_parallel_batch(self, batch):
        """ Vypočítá gradient paralelně pro daný batch. """
        batch_gradient = {param_name: 0 for param_name in self.cont_parameters.keys()}

        with ThreadPoolExecutor() as executor:
            for param_name in self.cont_parameters.keys():
                cont_normalized_parameters_plus = self.cont_normalized_parameters.copy()
                cont_normalized_parameters_minus = self.cont_normalized_parameters.copy()

                epsilon = self.epsilon
                cont_normalized_parameters_plus[param_name] += min(epsilon,
                                                                   1 - cont_normalized_parameters_plus[param_name])
                cont_normalized_parameters_minus[param_name] -= min(epsilon,
                                                                    cont_normalized_parameters_minus[param_name])

                future_plus = executor.submit(self.evaluate_function, batch,
                                              cont_normalized_parameters_plus)
                future_minus = executor.submit(self.evaluate_function, batch,
                                               cont_normalized_parameters_minus)

                f_plus_epsilon = future_plus.result()
                f_minus_epsilon = future_minus.result()

                batch_gradient[param_name] = (f_plus_epsilon - f_minus_epsilon) / (2 * self.epsilon)

        averaged_batch_gradient = {param_name: grad / self.num_batches for param_name, grad in batch_gradient.items()}
        return averaged_batch_gradient

    def normalize_parameters(self, parameters):
        normalized_parameters = {
            name: (value - self.cont_param_ranges[name][0]) / (
                    self.cont_param_ranges[name][1] - self.cont_param_ranges[name][0])
            for name, value in parameters.items()
        }
        return normalized_parameters

    def denormalize_parameters(self, normalized_parameters):
        parameters = {
            name: value * (self.cont_param_ranges[name][1] - self.cont_param_ranges[name][0]) +
                  self.cont_param_ranges[name][0]
            for name, value in normalized_parameters.items()
        }
        return parameters

    def adam_optimizer_parallel(self):
        average_iou = 0
        iteration_time = 0
        parameters_array = []
        IoU_array = []
        iterations = 1
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        m_t = {param_name: 0 for param_name in self.cont_parameters.keys()}
        v_t = {param_name: 0 for param_name in self.cont_parameters.keys()}

        while iterations < self.num_iterations:
            start_time = time.time()
            iteration_iou = []

            parameters_prev = np.r_[list(self.cont_parameters.values()), list(self.disc_parameters.values())]

            cont_normalized_parameters = self.cont_normalized_parameters.copy()
            disc_parameters = self.disc_parameters.copy()

            for batch_num, batch in enumerate(self.data_loader):
                batch_gradient = self.compute_gradient_parallel_batch(batch)

                m_t = {param_name: beta1 * m_t[param_name] + (1 - beta1) * batch_gradient[param_name] for param_name in
                       self.cont_parameters.keys()}
                v_t = {param_name: beta2 * v_t[param_name] + (1 - beta2) * (batch_gradient[param_name] ** 2) for
                       param_name in self.cont_parameters.keys()}

                m_t_hat = {param_name: m_t[param_name] / (1 - beta1 ** iterations) for param_name in
                           self.cont_parameters.keys()}
                v_t_hat = {param_name: v_t[param_name] / (1 - beta2 ** iterations) for param_name in
                           self.cont_parameters.keys()}

                for param_name in cont_normalized_parameters.keys():
                    cont_normalized_parameters[param_name] += self.lr * m_t_hat[param_name] / (
                            np.sqrt(v_t_hat[param_name]) + epsilon)
                    # Clip normalized parameters to [0, 1]
                    cont_normalized_parameters[param_name] = np.clip(cont_normalized_parameters[param_name],
                                                                     0, 1)

                batch_disc_gradient = self.compute_gradient_parallel_batch_disc(batch)
                for key in disc_parameters.keys():
                    disc_parameters[key] += batch_disc_gradient.get(key, 0)

                iou = self.instance.run(batch, {**self.cont_parameters, **self.disc_parameters}, False)
                iteration_iou.append(iou)

                if self.progress_window:
                    self.progress_window.update_info(self.projekt, self.algorithm, iterations,
                                                     round(average_iou * 100, 2),
                                                     round(iteration_time * (self.num_iterations - iterations)),
                                                     f"{batch_num + 1}/{self.num_batches}")
                print(
                    f"Project: {self.projekt}, Algorithm: {self.algorithm}, Iteration: {iterations}, Batch:{batch_num + 1}/{self.num_batches} IoU: {round(iou * 100, 2)}%")

            average_iou = np.average(iteration_iou)
            IoU_array.append(average_iou)

            self.cont_normalized_parameters = cont_normalized_parameters
            self.cont_parameters = self.denormalize_parameters(cont_normalized_parameters)

            rounded_disc_parameters = {key: round(value) for key, value in disc_parameters.items()}
            self.disc_parameters = rounded_disc_parameters

            parameters_new_values = np.r_[
                list(self.cont_parameters.values()), list(self.disc_parameters.values())]

            parameters_new = {**self.cont_parameters, **self.disc_parameters}

            parameter_change = np.linalg.norm(
                np.divide(
                    np.subtract(parameters_new_values, parameters_prev),
                    np.subtract(
                        np.r_[(list(self.cont_param_ranges.values()), list(self.disc_param_ranges.values()))][:,
                        1],
                        np.r_[(list(self.cont_param_ranges.values()), list(self.disc_param_ranges.values()))][:,
                        0]))) / (
                                       len(self.disc_param_ranges) + len(self.cont_param_ranges))

            parameters_array.append(parameters_new)

            iteration_time = time.time() - start_time

            rounded_parameters_new = {key: round(value, 3) for key, value in parameters_new.items()}

            if self.progress_window:
                self.progress_window.update_info(self.projekt, self.algorithm, iterations, round(average_iou * 100, 2),
                                                 round(iteration_time * (self.num_iterations - iterations)),
                                                 f"FINISHED")

            print(f"Project: {self.projekt}, Algorithm: {self.algorithm}, Iteration: {iterations}, "
                  f"IoU: {round(average_iou * 100, 2)}%, Parameters = {rounded_parameters_new}, "
                  f"parameter delta = {round(parameter_change, 4)}, Iteration time: {round(iteration_time)} seconds, "
                  f"Estimated time remaining: {round(iteration_time * (self.num_iterations - iterations))} seconds")

            iterations += 1

            if parameter_change < self.delta:
                break

        # Výběr nejlepší sady parametrů
        index = np.argmax(IoU_array)
        json_data_list = []
        for batch in self.data_loader:
            json_data_list.append(self.instance.run(batch, parameters_array[index], True))

        self.instance.save_parameters_json(IoU_array[index], json_data_list)

        print(f"Optimalization done. IoU: {round(IoU_array[index] * 100, 2)}%")
        return parameters_array[index], IoU_array[index]

    def run(self):
        parameters = {**self.cont_parameters, **self.disc_parameters}
        print(f"Project: {self.projekt}, Algorithm: {self.algorithm}, Iteration {0}, Parameters = {parameters}")

        parameters, iou = self.adam_optimizer_parallel()  # Zavolání Adam optimizeru

        return parameters, iou
