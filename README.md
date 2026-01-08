# Spheroid Segmentation Optimizer

Automatic optimization of spheroid segmentation parameters using **Gradient Descent (Adam)** or **Bayesian Optimization (GP + EI)**.

## Features

- **Two optimization algorithms:**
  - **Gradient Descent (Adam)** - classical approach with finite differences
  - **Bayesian Optimization (GP+EI)** - global optimization using Gaussian Process

- **Three segmentation methods:**
  - Sauvola adaptive thresholding
  - Niblack adaptive thresholding
  - Gaussian adaptive thresholding

- **Automatic parameter optimization:**
  - `window_size` - window size for thresholding
  - `min_area` - minimum contour area
  - `sigma` - Canny edge detection parameter
  - `std_k` - standard deviation coefficient
  - `dilation_size` - morphological dilation size

## Installation

```bash
# Clone the repository
git clone https://github.com/michalprusek/prusek-spheroid.git
cd prusek-spheroid

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### GUI Application

```bash
python GUI.py
```

### Programmatic Usage

```python
from BayesianOptimizerGUI import BayesianOptimizer
from GradientDescentGUI import GradientDescent
from ContoursClassGUI import IoU
import Funkce as f

# Load data
data = f.load_masks_from_images("path/to/masks", "path/to/images")

# Bayesian Optimization
optimizer = BayesianOptimizer(
    annotation_data=data,
    outputAddress="output/",
    projekt="my_project",
    algorithm="Gaussian",  # or "Sauvola", "Niblack"
    learning_rate=0.01,
    num_iterations=100,
    delta=0,
    batch_size=10,
    f=IoU,
    inner_contours=False,
    detect_corrupted=True
)

best_params, best_iou = optimizer.run()
print(f"Best IoU: {best_iou*100:.2f}%")
```

## Optimizer Comparison

| Property | Gradient Descent | Bayesian Optimization |
|----------|------------------|----------------------|
| Algorithm | Adam + finite diff | GP + Expected Improvement |
| Convergence | Local minimum | Global minimum |
| Discrete parameters | Hill climbing | Native Integer space |

## Project Structure

```
prusek-spheroid/
├── GUI.py                    # Main GUI application
├── BayesianOptimizerGUI.py   # Bayesian Optimization
├── GradientDescentGUI.py     # Gradient Descent (Adam)
├── ContoursClassGUI.py       # Segmentation and IoU calculation
├── Funkce.py                 # Helper functions
├── gpu_utils.py              # GPU/CUDA utility
├── requirements.txt          # Python dependencies
└── README.md
```

## License

MIT License
