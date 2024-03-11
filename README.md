# Prusek-Spheroid

Prusek-Spheroid is a Python package designed for spheroid segmentation based on provided microscope images. This package provides an easy-to-use interface and functionalities that are essential for determination of properties and characteristics of the spheroids.

## Installation

### Installing Python

#### For Windows:

1. **Download Python** from [python.org](https://python.org).
2. **Run the downloaded installer**. Ensure to check the "Add Python to PATH" option.
3. **Verify the installation** by opening CMD and typing `python --version`.

#### For MacOS/Linux:

1. **Check if Python is installed** by typing `python3 --version` in the terminal. If not installed:
   - On MacOS: Install via Homebrew with `brew install python3`.
   - On Linux: Install using `sudo apt-get install python3`.
2. **Verify the installation** by typing `python3 --version` in the terminal.

### Installing Miniconda

1. **Download Miniconda** from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html).
2. **Install Miniconda** and follow the on-screen instructions.
3. **Verify the installation** by opening a new terminal or CMD and typing `conda list`.

### Creating a Virtual Environment and Installing Prusek-Spheroid

1. **Create a virtual environment** using Miniconda: `conda create -n myenv python=3.x`.
2. **Activate the virtual environment**: `conda activate myenv` (Windows) or `source activate myenv` (MacOS/Linux).
3. **Install the Prusek-Spheroid package**: `pip install prusek_spheroid`.

## Running the Package

To use the package, first ensure it is up to date: `pip install --upgrade prusek_spheroid`

then run the program using the command: `python -m prusek_spheroid.GUI`

## User Guide for Prusek-Spheroid GUI

Prusek-Spheroid is a sophisticated Python package equipped with a user-friendly graphical interface (GUI) that facilitates image segmentation and optimization tasks. This guide provides an overview of the GUI functionalities and how to use them effectively. The whole program is based on the knowledge of segmentations of several (approximately 15 to 20) spheroids. Based on these segmentations, the program learns to segment the remaining spheroid images in the dataset (project). So far the only possible formats in which the segmentations can be loaded are the COCO 1.0 format or loading masks and corresponding images. This is the format supported by the CVAT (Computer Vision Annotation Tool) platform.

### Key Features

1. **File Selection**: Users can easily select image files or datasets for processing. This feature allows you to work with your preferred data seamlessly.

2. **Progress Windows**: The GUI includes dedicated windows displaying the ongoing progress of image processing and optimization tasks, keeping you informed every step of the way.

3. **Segmentation Parameters**: Customize your segmentation process with adjustable parameters. These settings allow you to fine-tune the segmentation to suit your specific needs.

4. **Optimization Tools**: Enhance the accuracy of your segmentation with built-in optimization algorithms based on the well-known Gradient descent algorithm. These tools are designed to improve the outcome of your image processing tasks.

### Using the GUI

The Prusek-Spheroid GUI is designed to be intuitive, providing a range of functionalities for effective image segmentation. Here’s a detailed overview of the key elements:

1. **Input File and Folder Selection**: These two sections are used for selecting and loading the annotations from CVAT in COCO 1.0 format (or loading masks and their corresponding images) and directory where your dataset images for segmentation are stored. You can easily navigate and choose the required folders and zip file (in COCO 1.0 format from downloaded from CVAT) that contain your image files.

2. **Output Folder Selection**: After processing, the results will be saved in the selected output folder. This feature allows you to specify where you want the segmented images and other outputs to be stored. If selected, the properties of the contours are also saved in the resulting folder as an excel file. Furthermore, the JSON file with the optimal parameters is also saved in the selected output folder in the "IoU" subfolder.

3. **Project Name Field**: Here, you enter the name of your project. This name will be used for organizing and saving the results in a structured manner.

4. **Method Selection**: This checkbox menu lets you choose the segmentation method. The GUI includes various methods like Sauvola, Niblack, Gaussian and Mean Shift. Notably, Sauvola and Niblack methods are considered the best choices due to their effectiveness and robustness in segmentation of spheroids on a variable background subject to inhomogeneous light conditions.

5. **Gradient Descent Parameters**: These parameters are crucial for the Gradient Descent algorithm. They include settings like learning rate, number of iterations, and delta stop condition (the number below which, if the change of parameters in two consecutive iterations of the Gradient descent algorithm falls, the optimization stops). It's generally recommended not to alter these parameters unless you have specific requirements or understanding of the algorithm.

6. **Other Settings Checkboxes**: This section contains various checkboxes for additional settings. These settings might include options for finding also inner contours, the possibility to create a zip file in COCO 1.0 format, which will contain the resulting segmentations and can then be uploaded to the CVAT platform. It is also possible to calculate the characteristics and properties from the obtained outer contours, whose values are then stored in a separate Excel file.

7. **'I Already Know the Parameters' Checkbox**: If you have pre-determined parameters from a previous Gradient descent project, you can use this checkbox. It's useful when you want to segment images using already optimized parameters without going through the Gradient descent process again. Is also recommended to use when knowing the parameters from a visually similar project.

### Additional Information

- The GUI is structured to facilitate both beginners and advanced users in the field of image processing.
- Each feature and button is designed to provide maximum control over the segmentation process, ensuring that users can tailor the process to their specific data and requirements.
- Regular updates and feedback from users are encouraged to continuously improve the functionality and user experience.

For detailed explanations of the segmentation methods and Gradient Descent algorithm, refer to the technical documentation provided with the package at Github [prusek-spheroid](https://github.com/michalprusek/prusek-spheroid).

If you encounter any issues or need further assistance, please feel free to reach out to prusemic@fjfi.cvut.cz


## Support 

For any issues or queries, contact me at: prusemic@fjfi.cvut.cz