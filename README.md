# Spheroid Segmentation Optimizer

Automatická optimalizace parametrů segmentace sféroidů pomocí **Gradient Descent (Adam)** nebo **Bayesian Optimization (GP + EI)**.

## Funkce

- **Dva optimizační algoritmy:**
  - **Gradient Descent (Adam)** - klasický přístup s konečnými diferencemi
  - **Bayesian Optimization (GP+EI)** - globální optimalizace pomocí Gaussian Process

- **Tři segmentační metody:**
  - Sauvola adaptive thresholding
  - Niblack adaptive thresholding
  - Gaussian adaptive thresholding

- **Automatická optimalizace parametrů:**
  - `window_size` - velikost okna pro prahování
  - `min_area` - minimální plocha kontury
  - `sigma` - parametr Canny edge detection
  - `std_k` - koeficient standardní odchylky
  - `dilation_size` - velikost morfologické dilatace

## Instalace

```bash
# Klonovat repozitář
git clone https://github.com/michalprusek/prusek-spheroid.git
cd prusek-spheroid

# Vytvořit virtuální prostředí
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# nebo: .venv\Scripts\activate  # Windows

# Nainstalovat závislosti
pip install -r requirements.txt
```

## Spuštění

### GUI aplikace

```bash
python GUI.py
```

### Programatické použití

```python
from BayesianOptimizerGUI import BayesianOptimizer
from GradientDescentGUI import GradientDescent
from ContoursClassGUI import IoU
import Funkce as f

# Načíst data
data = f.load_masks_from_images("path/to/masks", "path/to/images")

# Bayesian Optimization
optimizer = BayesianOptimizer(
    annotation_data=data,
    outputAddress="output/",
    projekt="my_project",
    algorithm="Gaussian",  # nebo "Sauvola", "Niblack"
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

## Porovnání optimizérů

| Vlastnost | Gradient Descent | Bayesian Optimization |
|-----------|------------------|----------------------|
| Algoritmus | Adam + finite diff | GP + Expected Improvement |
| Konvergence | Lokální minimum | Globální minimum |
| Diskrétní parametry | Hill climbing | Native Integer space |

## Struktura projektu

```
prusek-spheroid/
├── GUI.py                    # Hlavní GUI aplikace
├── BayesianOptimizerGUI.py   # Bayesian Optimization
├── GradientDescentGUI.py     # Gradient Descent (Adam)
├── ContoursClassGUI.py       # Segmentace a IoU výpočet
├── Funkce.py                 # Pomocné funkce
├── gpu_utils.py              # GPU/CUDA utility
├── requirements.txt          # Python závislosti
└── README.md
```

## Licence

MIT License
