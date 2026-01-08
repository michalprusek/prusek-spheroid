import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox, Toplevel, Label
from prusek_spheroid_bayesian import BayesianOptimizerGUI as bo
from prusek_spheroid_bayesian import GradientDescentGUI as gd
from prusek_spheroid_bayesian import ContoursClassGUI as F
from prusek_spheroid_bayesian import Funkce as f
import threading
import time
import json
import zipfile
import os
from prusek_spheroid_bayesian import characteristic_functions as cf
import pandas as pd
import cv2 as cv


def update_addresses(annotationsAddress):
    annotation_address = os.path.join(annotationsAddress, "annotations")
    images_address = os.path.join(annotationsAddress, "images")

    return annotation_address, images_address


def unzip(zip_file_path, extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Assuming the content is directly in the extracted folder
    extracted_folder = extract_path

    return extracted_folder


class ProcessingProgressWindow(tk.Toplevel):
    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.title("Processing Progress")

        self.geometry("500x250")
        self.resizable(width=False, height=False)

        # Frame for centering content
        center_frame = tk.Frame(self)
        center_frame.pack(expand=True)

        # Label in frame for centering
        self.label_progress = tk.Label(center_frame, text="Progress: ")
        self.label_progress.pack()

    def update_progress(self, progress):
        self.label_progress.config(text=f"Progress: {progress}")


class OptimizationProgressWindow(tk.Toplevel):
    def __init__(self, master):
        tk.Toplevel.__init__(self, master)
        self.title("Optimization Progress")

        self.geometry("500x250")  # Width x Height
        self.resizable(width=False, height=False)

        # Frame for centering content
        center_frame = tk.Frame(self)
        center_frame.pack(expand=True)

        # Labels in frame for centering
        self.label_project = tk.Label(center_frame, text="Project: ")
        self.label_project.pack()

        self.label_algorithm = tk.Label(center_frame, text="Algorithm: ")
        self.label_algorithm.pack()

        self.label_iteration = tk.Label(center_frame, text="Iteration: ")
        self.label_iteration.pack()

        self.label_iou = tk.Label(center_frame, text="IoU: ")
        self.label_iou.pack()

        self.label_parameters = tk.Label(center_frame, text="Estimated time remaining: ")
        self.label_parameters.pack()

        self.label_batch_num = tk.Label(center_frame, text="Batch number: ")
        self.label_batch_num.pack()

    def update_info(self, project, algorithm, iteration, IoU, time_rem, batch_info):
        self.label_project.config(text=f"Project: {project}")
        self.label_algorithm.config(text=f"Algorithm: {algorithm}")
        self.label_iteration.config(text=f"Iteration: {iteration}")
        self.label_iou.config(text=f"IoU: {IoU}%")
        self.label_parameters.config(text=f"Estimated time remaining: {time_rem} seconds")
        self.label_batch_num.config(text=f"Batch number: {batch_info}")


class ParameterEntryDialog(tk.Toplevel):
    def __init__(self, master, algorithm, parameters):
        tk.Toplevel.__init__(self, master)
        self.title("Enter Parameters")

        self.algorithm = algorithm
        self.parameters = parameters  # Store loaded parameters from JSON
        self.result = None

        # Create labels and entry widgets for each parameter with descriptions and ranges
        self.entries = {}
        for param, value in self.parameters.items():
            label = tk.Label(self, text=f"{param}:")
            label.pack()
            entry = tk.Entry(self)
            entry.insert(0, str(value))
            entry.pack()
            self.entries[param] = entry

        # OK button to confirm parameter values
        ok_button = tk.Button(self, text="OK", command=self.confirm_parameters)
        ok_button.pack()

    def confirm_parameters(self):
        # Retrieve parameter values from entry widgets and update self.parameters
        for param, entry in self.entries.items():
            value = entry.get()
            try:
                value = float(value)
                self.parameters[param] = value
            except ValueError:
                # Handle invalid input, e.g., non-numeric values
                pass
        self.result = self.parameters  # Set the result to the updated parameters
        self.destroy()  # Close the dialog

    def close_dialog(self):
        self.result = None
        self.destroy()

    def get_parameters(self):
        return self.parameters

    def set_parameters(self, parameters):
        for param, value in parameters.items():
            if param in self.entries:
                self.entries[param].delete(0, tk.END)
                self.entries[param].insert(0, str(value))

    def update_parameters(self, new_parameters):
        # Update parameters in the main application when entering them manually
        self.parameters = new_parameters


def browse_file(var, title, label):
    file_path = filedialog.askopenfilename()
    var.set(file_path)
    # Update the passed label with custom text
    label.config(text=f"{title}: {file_path}")


def browse_directory(var, title, label):
    directory_path = filedialog.askdirectory()
    var.set(directory_path)
    # Update the passed label with custom text
    label.config(text=f"{title}: {directory_path}")


class SferoidSegmentationGUI:
    def __init__(self, master):
        self.loaded_parameters = None
        self.loaded_method = None

        self.master = master
        self.master.title("Spheroid segmentation")

        # Address section
        self.address_section_label = tk.Label(master, text="Address Settings", font=("Helvetica", 12, "bold"))
        self.address_section_label.pack()

        self.annotation_frame = tk.Frame(master)
        self.annotation_frame.pack()

        self.COCO_frame = tk.Frame(self.annotation_frame)
        self.COCO_frame.pack(side=tk.LEFT)

        self.load_coco_var = tk.IntVar()
        self.load_coco_checkbox = tk.Checkbutton(self.COCO_frame,
                                                 text="Load annotations in COCO 1.0 format",
                                                 variable=self.load_coco_var,
                                                 command=lambda: self.checkbox_changed('coco'))
        self.load_coco_checkbox.pack(side=tk.TOP)

        # File Dialog for COCO annotations
        self.coco_annotation_button = tk.Button(self.COCO_frame,
                                                text="Retrieve the COCO 1.0 ZIP file address",
                                                command=lambda: browse_file(self.coco_annotation_path,
                                                                            "Selected ZIP file address",
                                                                            self.coco_address_label))
        self.coco_annotation_button.pack(side=tk.TOP)

        self.coco_address_label = tk.Label(self.COCO_frame, text="")
        self.coco_address_label.pack()

        # Add "or" label
        self.or_label = tk.Label(self.annotation_frame, text="or")
        self.or_label.pack(side=tk.LEFT, padx=5)

        self.masks_frame = tk.Frame(self.annotation_frame)
        self.masks_frame.pack(side=tk.TOP)

        self.load_masks_var = tk.IntVar()
        self.load_masks_checkbox = tk.Checkbutton(self.masks_frame,
                                                  text="Load annotations using spheroid masks",
                                                  variable=self.load_masks_var,
                                                  command=lambda: self.checkbox_changed('mask'))
        self.load_masks_checkbox.pack(side=tk.TOP)

        # Retrieve mask address button
        self.retrieve_mask_button = tk.Button(self.masks_frame,
                                              text="Retrieve the folder address of masks of annotated spheroids",
                                              command=lambda: browse_directory(self.masks_annotation_path,
                                                                               "Selected mask address",
                                                                               self.mask_address_label))
        self.retrieve_mask_button.pack(side=tk.TOP)

        # Create label for displaying mask path
        self.mask_address_label = tk.Label(self.masks_frame, text="")
        self.mask_address_label.pack()

        # Retrieve spheroid images address button
        self.retrieve_images_button = tk.Button(self.masks_frame,
                                                text="Retrieve the folder address of spheroid images",
                                                command=lambda: browse_directory(self.image_dataset_path,
                                                                                 "Selected spheroid images address",
                                                                                 self.images_address_label))
        self.retrieve_images_button.pack(side=tk.TOP)

        self.images_address_label = tk.Label(self.masks_frame, text="")
        self.images_address_label.pack()

        self.coco_annotation_path = tk.StringVar()
        self.masks_annotation_path = tk.StringVar()
        self.image_dataset_path = tk.StringVar()

        # File Dialog for output segmented images (changed to askdirectory)
        self.dataset_address_button = tk.Button(master,
                                                text="Dataset of all images address (folder of images you want to segment)",
                                                command=lambda: browse_directory(self.dataset_path,
                                                                                 "Selected dataset path",
                                                                                 self.dataset_address_label))
        self.dataset_address_button.pack(side=tk.TOP)

        self.dataset_path = tk.StringVar()

        # Indicator for address filled
        self.dataset_address_label = tk.Label(master, text="")
        self.dataset_address_label.pack()

        # File Dialog for output segmented images (changed to askdirectory)
        self.output_address_button = tk.Button(master,
                                               text="Output address (folder where to save the output)",
                                               command=lambda: browse_directory(self.output_path,
                                                                                "Select output path:",
                                                                                self.output_address_label))
        self.output_address_button.pack(side=tk.TOP)

        self.output_path = tk.StringVar()

        # Indicator for address filled
        self.output_address_label = tk.Label(master, text="")
        self.output_address_label.pack()

        # Separator line between address and methods section
        self.address_separator = tk.Frame(master, height=2, bd=1, relief=tk.SUNKEN)
        self.address_separator.pack(fill=tk.X, padx=5, pady=5)

        # Method section
        self.method_section_label = tk.Label(master, text="Method Settings", font=("Helvetica", 12, "bold"))
        self.method_section_label.pack()

        # Textbox for project name
        self.project_name_label = tk.Label(master, text="Project Name:")
        self.project_name_label.pack()
        self.project_name_entry = tk.Entry(master)
        self.project_name_entry.pack()

        # Button for loading parameters from JSON file
        self.load_parameters_button = tk.Button(master,
                                                text="I already know the parameters (load JSON file with parameters)",
                                                command=self.load_and_run_parameters)
        self.load_parameters_button.pack()

        # Create a frame to contain the label and button
        self.parameters_frame = tk.Frame(self.master)
        self.parameters_frame.pack()

        # Initially disable the "Parameters loaded" label and the "Cancel" button
        self.parameters_loaded_label = tk.Label(self.parameters_frame, text="Parameters loaded", state=tk.DISABLED)
        self.parameters_loaded_label.pack(side=tk.LEFT)

        self.cancel_button = tk.Button(self.parameters_frame, text="Cancel", command=self.cancel_parameters_loaded,
                                       state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT)

        # Checkboxes for method selection
        self.methods_frame = tk.Frame(master)
        self.methods_frame.pack()
        self.method_labels = ["Sauvola", "Niblack", "Gaussian"]

        self.methods_checkboxes = []

        for i, method_label in enumerate(self.method_labels):
            method_var = tk.IntVar()
            method_checkbox = tk.Checkbutton(self.methods_frame, text=method_label, variable=method_var)
            method_checkbox.grid(row=0, column=i, padx=5, pady=5)

            self.methods_checkboxes.append((method_checkbox, method_var))

        # Frame for optimizer selection (GD vs BO)
        self.optimizer_frame = tk.Frame(master)
        self.optimizer_frame.pack()

        self.optimizer_label = tk.Label(self.optimizer_frame, text="Optimizer:")
        self.optimizer_label.grid(row=0, column=0, padx=5, pady=5)

        self.optimizer_var = tk.StringVar(value="BO")  # Default: Bayesian Optimization

        self.gd_radio = tk.Radiobutton(self.optimizer_frame, text="Gradient Descent (Adam)",
                                        variable=self.optimizer_var, value="GD")
        self.gd_radio.grid(row=0, column=1, padx=5, pady=5)

        self.bo_radio = tk.Radiobutton(self.optimizer_frame, text="Bayesian Optimization (GP+EI)",
                                        variable=self.optimizer_var, value="BO")
        self.bo_radio.grid(row=0, column=2, padx=5, pady=5)

        # Text fields for parameter input
        self.parameters_frame = tk.Frame(master)
        self.parameters_frame.pack()
        self.learning_rate_label = tk.Label(self.parameters_frame, text="Learning Rate:")
        self.learning_rate_label.grid(row=0, column=0, padx=5, pady=5)
        self.learning_rate_entry = tk.Entry(self.parameters_frame)
        self.learning_rate_entry.grid(row=0, column=1, padx=5, pady=5)

        self.iterations_label = tk.Label(self.parameters_frame, text="Number of Iterations:")
        self.iterations_label.grid(row=1, column=0, padx=5, pady=5)
        self.iterations_entry = tk.Entry(self.parameters_frame)
        self.iterations_entry.grid(row=1, column=1, padx=5, pady=5)

        self.stop_condition_label = tk.Label(self.parameters_frame, text="Stop Condition:")
        self.stop_condition_label.grid(row=2, column=0, padx=5, pady=5)
        self.stop_condition_entry = tk.Entry(self.parameters_frame)
        self.stop_condition_entry.grid(row=2, column=1, padx=5, pady=5)

        self.batch_size_label = tk.Label(self.parameters_frame, text="Batch Size:")
        self.batch_size_label.grid(row=3, column=0, padx=5, pady=5)
        self.batch_size_entry = tk.Entry(self.parameters_frame)
        self.batch_size_entry.grid(row=3, column=1, padx=5, pady=5)

        # Preset parameter values
        self.learning_rate_entry.insert(0, "0.01")
        self.iterations_entry.insert(0, "50")
        self.stop_condition_entry.insert(0, "0.0002")
        self.batch_size_entry.insert(0, "10")

        # Separator line between address and methods section
        self.method_separator = tk.Frame(master, height=2, bd=1, relief=tk.SUNKEN)
        self.method_separator.pack(fill=tk.X, padx=5, pady=5)

        self.other_section_label = tk.Label(master, text="Other Settings", font=("Helvetica", 12, "bold"))
        self.other_section_label.pack()

        checkbox_frame = tk.Frame(master)
        checkbox_frame.pack()

        # Checkbox for 'also find the inner contours'
        self.inner_contours_var = tk.BooleanVar()
        self.checkbox_inner_contours = tk.Checkbutton(checkbox_frame, text="Also find the inner contours",
                                                      variable=self.inner_contours_var,
                                                      onvalue=True, offvalue=False)
        self.checkbox_inner_contours.pack(side=tk.LEFT)

        self.detect_corrupted_var = tk.BooleanVar()
        self.checkbox_detect_corrupted = tk.Checkbutton(checkbox_frame, text="Detect and discard corrupted images",
                                                        variable=self.detect_corrupted_var, onvalue=True,
                                                        offvalue=False)
        self.checkbox_detect_corrupted.pack(side=tk.LEFT)

        # Checkbox for 'Create JSON file for export to CVAT'
        self.create_json_var = tk.BooleanVar()
        self.checkbox_create_json = tk.Checkbutton(checkbox_frame, text="Create JSON file for export to CVAT",
                                                   variable=self.create_json_var,
                                                   onvalue=True, offvalue=False)
        self.checkbox_create_json.pack(side=tk.LEFT)

        # Checkbox for 'Calculate contour properties'
        self.calculate_contours_var = tk.BooleanVar()
        self.checkbox_calculate_contours = tk.Checkbutton(checkbox_frame, text="Calculate spheroid properties",
                                                          variable=self.calculate_contours_var,
                                                          onvalue=True, offvalue=False)
        self.checkbox_calculate_contours.pack(side=tk.LEFT)

        # Run button
        self.run_button = tk.Button(master, text="Run", command=self.run_method)
        self.run_button.pack()

        # Adding a visual separator
        self.run_separator = tk.Frame(master, height=2, bd=1, relief=tk.SUNKEN)
        self.run_separator.pack(fill=tk.X, padx=5, pady=5)

        # Spheroids quantification section title
        self.quantification_section_label = tk.Label(master, text="Spheroids Quantification",
                                                     font=("Helvetica", 12, "bold"))
        self.quantification_section_label.pack()

        # Button to retrieve the folder address of masks of annotated spheroids
        self.retrieve_masks_button_quantification = tk.Button(master,
                                                              text="Retrieve the folder address of masks of annotated spheroids",
                                                              command=lambda: browse_directory(
                                                                  self.masks_annotation_path_quantification,
                                                                  "Selected masks folder",
                                                                  self.masks_address_label_quantification))
        self.retrieve_masks_button_quantification.pack()

        self.masks_annotation_path_quantification = tk.StringVar()
        self.masks_address_label_quantification = tk.Label(master, text="")
        self.masks_address_label_quantification.pack()

        # Button to retrieve the output folder address
        self.retrieve_output_button_quantification = tk.Button(master,
                                                               text="Retrieve the output folder address",
                                                               command=lambda: browse_directory(
                                                                   self.output_path_quantification,
                                                                   "Selected output folder",
                                                                   self.output_address_label_quantification))
        self.retrieve_output_button_quantification.pack()

        self.output_path_quantification = tk.StringVar()
        self.output_address_label_quantification = tk.Label(master, text="")
        self.output_address_label_quantification.pack()

        # Button to calculate spheroid properties
        self.calculate_properties_button = tk.Button(master, text="Calculate spheroid properties",
                                                     command=self.calculate_spheroid_properties)
        self.calculate_properties_button.pack()

        # Update button state on startup
        self.update_buttons_state()

    def calculate_spheroid_properties(self):
        if not self.masks_annotation_path_quantification.get() or not self.output_path_quantification.get():
            messagebox.showerror("Error", "Both mask folder and output folder addresses must be filled.")
            return

        masks_data = f.load_masks(self.masks_annotation_path_quantification.get())
        total_masks = len(masks_data)

        # Dialog for progress
        # Dialog for progress
        progress_dialog = Toplevel(self.master)
        progress_dialog.title("Processing Progress")
        progress_dialog.geometry("400x100")  # Sets window dimensions to 400x100 pixels
        Label(progress_dialog, text=f"Total images: {total_masks}").pack()
        progress_label = Label(progress_dialog, text="Starting...")
        progress_label.pack()

        all_contour_data = []

        for index, (mask, name) in enumerate(masks_data, start=1):
            progress_label.config(text=f"Processing image {index} of {total_masks}")
            progress_dialog.update()

            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv.contourArea, reverse=True)

            for order, contour in enumerate(contours, start=1):
                # Your contour processing and feature calculation goes here
                additional_data = cf.calculate_all(contour)  # This is your custom calculation function
                contour_data = {
                    'MaskName': os.path.basename(name),
                    'ContourOrder': order,
                    **additional_data
                }
                all_contour_data.append(contour_data)

        columns = [
            'MaskName', 'ContourOrder', 'Area', 'Circularity', 'Compactness', 'Convexity',
            'EquivalentDiameter', 'FeretAspectRatio', 'FeretDiameterMax',
            'FeretDiameterMaxOrthogonalDistance', 'FeretDiameterMin',
            'LengthMajorDiameterThroughCentroid', 'LengthMinorDiameterThroughCentroid',
            'Perimeter', 'Solidity', 'Sphericity'
        ]
        df = pd.DataFrame(all_contour_data, columns=columns)
        output_path = self.output_path_quantification.get()
        df.to_excel(f"{output_path}/contour_properties.xlsx")

        # Close the progress dialog
        progress_dialog.destroy()

        # Show completion message
        messagebox.showinfo("Completed", f"Spheroid properties calculated and saved.\nOutput path: {output_path}")
        print("Spheroid properties calculated and saved.")

    def show_completion_dialog(self, time_taken, output_folder):
        dialog = tk.Toplevel(self.master)
        dialog.title("Segmentation Completed")

        message = f"DONE.\nSegmentation took {time_taken:.2f} seconds.\nOutput stored in a folder: {output_folder}"
        tk.Label(dialog, text=message).pack(padx=20, pady=10)

        ok_button = tk.Button(dialog, text="OK", command=dialog.destroy)
        ok_button.pack(pady=10)

    def cancel_parameters_loaded(self):
        # Clear the loaded parameters
        self.loaded_parameters = {}

        # Reset the method selection
        for method_checkbox, _ in self.methods_checkboxes:
            method_checkbox.config(state=tk.NORMAL)

        # Unlock learning rate, number of iterations, and stop condition text fields
        self.learning_rate_entry.config(state=tk.NORMAL)
        self.iterations_entry.config(state=tk.NORMAL)
        self.stop_condition_entry.config(state=tk.NORMAL)
        self.batch_size_entry.config(state=tk.NORMAL)

        self.parameters_loaded_label.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.DISABLED)

    def update_buttons_state(self):
        # Update button and checkbox state based on currently selected checkbox
        if self.load_coco_var.get() == 1:
            self.retrieve_mask_button.config(state=tk.DISABLED)
            self.retrieve_images_button.config(state=tk.DISABLED)
            self.coco_annotation_button.config(state=tk.NORMAL)
        elif self.load_masks_var.get() == 1:
            self.retrieve_mask_button.config(state=tk.NORMAL)
            self.retrieve_images_button.config(state=tk.NORMAL)
            self.coco_annotation_button.config(state=tk.DISABLED)
        else:
            self.retrieve_mask_button.config(state=tk.DISABLED)
            self.retrieve_images_button.config(state=tk.DISABLED)
            self.coco_annotation_button.config(state=tk.DISABLED)

    def checkbox_changed(self, var):
        # This method is called when any checkbox changes
        if var == 'coco' and self.load_coco_var.get() == 1:
            self.load_masks_var.set(0)
        elif var == 'mask' and self.load_masks_var.get() == 1:
            self.load_coco_var.set(0)
        self.update_buttons_state()

    def enable_parameter_entry(self):
        state = tk.NORMAL
        self.learning_rate_entry.config(state=state)
        self.iterations_entry.config(state=state)
        self.stop_condition_entry.config(state=state)
        self.batch_size_entry.config(state=state)

    def load_and_run_parameters(self):
        json_file_path = filedialog.askopenfilename(
            title="Select the JSON file in which the already found optimal parameters are uploaded:",
            filetypes=[("JSON files", "*.json")])
        if json_file_path:
            self.load_json_parameters(json_file_path)
            self.run_method_with_loaded_parameters()

    def load_json_parameters(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract required data
        self.loaded_method = data.get("method", "")
        self.loaded_parameters = data.get("parameters", {})
        self.inner_contours_var.set(data.get("inner_contours", False))
        self.detect_corrupted_var.set(data.get("detect_corrupted", False))

    def run_method_with_loaded_parameters(self):
        if not self.loaded_parameters:
            messagebox.showerror("Error", "No parameters loaded.")
            return

        # This method assumes it opens a dialog to confirm parameters
        parameter_entry_dialog = ParameterEntryDialog(self.master, self.loaded_method, self.loaded_parameters)
        self.master.wait_window(parameter_entry_dialog)

        if parameter_entry_dialog.result is None:
            messagebox.showinfo("Info", "Parameters were not saved.")
        else:
            self.parameters_loaded_label.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.NORMAL)

            loaded_method = self.loaded_method

            # Set the loaded method and lock all method checkboxes
            for checkbox, var in self.methods_checkboxes:
                if checkbox.cget("text") == loaded_method:
                    var.set(1)  # Check the loaded method's checkbox
                    checkbox.config(state=tk.DISABLED)
                else:
                    var.set(0)  # Uncheck other method checkboxes
                    checkbox.config(state=tk.DISABLED)

            # Lock learning rate, number of iterations, and stop condition text fields
            self.learning_rate_entry.config(state=tk.DISABLED)
            self.iterations_entry.config(state=tk.DISABLED)
            self.stop_condition_entry.config(state=tk.DISABLED)
            self.batch_size_entry.config(state=tk.DISABLED)

    def run_method(self):
        project_name = self.project_name_entry.get()

        if self.load_coco_var.get() + self.load_masks_var.get() != 1:
            if self.load_coco_var.get() == self.load_masks_var.get() == 0:
                messagebox.showerror("Error", "At least one annotation loading method must be selected.")
            return

        # Check if required paths are filled
        if self.load_coco_var.get() == 1 and not self.coco_annotation_path.get():
            messagebox.showerror("Error", "The COCO annotations path must be filled.")
            return
        elif self.load_masks_var.get() == 1:
            if not self.masks_annotation_path.get() or not self.image_dataset_path.get():
                messagebox.showerror("Error", "Both masks and image dataset paths must be filled.")
                return

        # Check that the second and third addresses are always filled
        if not all([self.dataset_path.get(), self.output_path.get()]):
            messagebox.showerror("Error", "Both 'Image Dataset Path' and 'Output Path' must be filled.")
            return

        if not project_name:
            messagebox.showerror("Error", "Project name must be filled in.")
            return

        if not any(var.get() == 1 for _, var in self.methods_checkboxes):
            messagebox.showerror("Error", "At least one of the four segmentation methods must be selected.")
            return

        # Retrieve values from entry fields
        annotation_addresses = []
        if self.load_coco_var.get() == 1:
            annotation_address = self.coco_annotation_path.get()
            annotation_address = unzip(annotation_address, os.path.dirname(annotation_address))
            annotations_address, images_address = update_addresses(annotation_address)
            annotation_data = f.load_annotations(
                os.path.join(annotations_address, 'instances_default.json'), images_address)
            print(f"Loaded {len(annotation_data)} annotated images")
        else:
            mask_adress = self.masks_annotation_path.get()
            images_address = self.image_dataset_path.get()
            annotation_data = f.load_masks_from_images(mask_adress, images_address)
            print(f"Loaded {len(annotation_data)} annotated images")

        dataset_address = self.dataset_path.get()
        output_address = self.output_path.get()

        algorithms = [method for method, (checkbox, var) in
                      zip(["Sauvola", "Niblack", "Gaussian"], self.methods_checkboxes) if
                      var.get() == 1]

        # Loaded parameters from JSON file, if available
        if self.loaded_parameters:
            parameters = self.loaded_parameters
            progress_window1 = None
        else:
            parameters = None
            progress_window1 = OptimizationProgressWindow(self.master)
            progress_window1.update_info(project_name, algorithms, 0, "Unknown", "Unknown", "Unknown")
            progress_window1.withdraw()

        inner_contours_value = self.inner_contours_var.get()
        detect_corrupted = self.detect_corrupted_var.get()
        create_json = self.create_json_var.get()
        calculate_properties = self.calculate_contours_var.get()

        progress_window2 = ProcessingProgressWindow(self.master)
        progress_window2.update_progress("Initializing...")
        progress_window2.withdraw()

        run_thread = threading.Thread(target=self.run_main, args=(
            annotation_data, dataset_address, output_address, project_name, algorithms, parameters,
            inner_contours_value, detect_corrupted, create_json, calculate_properties, progress_window1,
            progress_window2))
        run_thread.start()

    def run_main(self, annotation_data, dataset_address, output_address, project_name, algorithms, known_parameters,
                 inner_contours, detect_corrupted, create_json, calculate_properties, progress_window1,
                 progress_window2):
        totalTime = time.time()

        for algorithm in algorithms:
            startTime = time.time()

            if known_parameters is None:
                learning_rate = float(self.learning_rate_entry.get())
                num_iterations = int(self.iterations_entry.get())
                stop_condition = float(self.stop_condition_entry.get())
                batch_size = int(self.batch_size_entry.get())
                progress_window1.deiconify()
                # Select optimizer based on radiobutton
                if self.optimizer_var.get() == "BO":
                    optimizer = bo.BayesianOptimizer(annotation_data, output_address, project_name, algorithm,
                                                    learning_rate, num_iterations, stop_condition, batch_size,
                                                    F.IoU, progress_window1,
                                                    inner_contours=inner_contours,
                                                    detect_corrupted=detect_corrupted)
                else:
                    optimizer = gd.GradientDescent(annotation_data, output_address, project_name, algorithm,
                                                   learning_rate, num_iterations, stop_condition, batch_size,
                                                   F.IoU, progress_window1,
                                                   inner_contours=inner_contours,
                                                   detect_corrupted=detect_corrupted)
                parameters, iou = optimizer.run()
                print(f"Optimizer: {self.optimizer_var.get()}, Resulting parameters: {parameters}, IoU: {round(iou * 100, 2)}%")
                progress_window1.withdraw()
            else:
                parameters = known_parameters

            progress_window2.deiconify()
            F.Contours(dataset_address, output_address, project_name, algorithm, parameters,
                       False, F.IoU, inner_contours, detect_corrupted, create_json, calculate_properties,
                       progress_window2).run()
            print(f"Segmentation of the project took: {round(time.time() - startTime)} seconds")
            progress_window2.withdraw()

        print(f"Total time: {round(time.time() - totalTime)} seconds")

        # Show dialog after segmentation is completed
        self.show_completion_dialog(round(time.time() - totalTime), output_address)


if __name__ == "__main__":
    root = tk.Tk()
    app = SferoidSegmentationGUI(root)
    root.mainloop()
