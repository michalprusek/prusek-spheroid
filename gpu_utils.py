"""
GPU-accelerated image processing utilities using Kornia and PyTorch.
Provides GPU implementations of morphological operations, edge detection,
local thresholding, and IoU calculation.
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    import kornia
    from kornia.morphology import erosion, dilation
    from kornia.filters import canny
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False

# Global device configuration
_device = None
_use_gpu = False


def init_gpu(force_cpu=False):
    """
    Initialize GPU if available.

    Args:
        force_cpu: If True, force CPU usage even if GPU is available

    Returns:
        tuple: (device, use_gpu flag)
    """
    global _device, _use_gpu

    if force_cpu:
        _device = torch.device('cpu')
        _use_gpu = False
    elif torch.cuda.is_available() and KORNIA_AVAILABLE:
        _device = torch.device('cuda')
        _use_gpu = True
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        print(f"GPU initialized: {torch.cuda.get_device_name(0)}")
    else:
        _device = torch.device('cpu')
        _use_gpu = False
        if not KORNIA_AVAILABLE:
            print("Kornia not available, using CPU")
        else:
            print("CUDA not available, using CPU")

    return _device, _use_gpu


def get_device():
    """Get current device, initializing if needed."""
    global _device
    if _device is None:
        init_gpu()
    return _device


def is_gpu_available():
    """Check if GPU processing is available."""
    global _use_gpu, _device
    if _device is None:
        init_gpu()
    return _use_gpu


def numpy_to_tensor(img, add_batch=True):
    """
    Convert numpy image to PyTorch tensor on GPU.

    Args:
        img: numpy array (H, W) or (H, W, C)
        add_batch: if True, adds batch dimension

    Returns:
        torch.Tensor on GPU with shape (1, 1, H, W) for grayscale
    """
    device = get_device()

    if img.ndim == 2:
        # Grayscale: (H, W) -> (1, H, W)
        tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
    else:
        # Color: (H, W, C) -> (C, H, W)
        tensor = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)

    if add_batch:
        tensor = tensor.unsqueeze(0)  # Add batch dimension: (1, C, H, W)

    return tensor.to(device)


def tensor_to_numpy(tensor):
    """
    Convert PyTorch tensor back to numpy.

    Args:
        tensor: PyTorch tensor

    Returns:
        numpy array
    """
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    return tensor.squeeze().numpy()


def _get_elliptical_kernel(size, device):
    """
    Create an elliptical structuring element.

    Args:
        size: kernel size (will create 2*size+1 kernel)
        device: torch device

    Returns:
        Kernel tensor of shape (2*size+1, 2*size+1)
    """
    k = 2 * size + 1
    y, x = torch.meshgrid(torch.arange(k, device=device), torch.arange(k, device=device), indexing='ij')
    center = size
    # Elliptical mask
    mask = ((x - center) ** 2 + (y - center) ** 2) <= (size ** 2)
    return mask.float()


def gpu_erosion(img_tensor, kernel_size, iterations=1):
    """
    GPU-accelerated erosion using Kornia.

    Args:
        img_tensor: (1, 1, H, W) tensor
        kernel_size: int, will create elliptical kernel of size (2*k+1, 2*k+1)
        iterations: number of erosion iterations

    Returns:
        Eroded tensor
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    device = get_device()
    kernel = _get_elliptical_kernel(kernel_size, device)

    result = img_tensor
    for _ in range(iterations):
        result = erosion(result, kernel)

    return result


def gpu_dilation(img_tensor, kernel_size, iterations=1):
    """
    GPU-accelerated dilation using Kornia.

    Args:
        img_tensor: (1, 1, H, W) tensor
        kernel_size: int, will create elliptical kernel of size (2*k+1, 2*k+1)
        iterations: number of dilation iterations

    Returns:
        Dilated tensor
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    device = get_device()
    kernel = _get_elliptical_kernel(kernel_size, device)

    result = img_tensor
    for _ in range(iterations):
        result = dilation(result, kernel)

    return result


def gpu_canny(img_tensor, sigma, low_threshold, high_threshold):
    """
    GPU-accelerated Canny edge detection using Kornia.

    Args:
        img_tensor: (1, 1, H, W) tensor, values should be normalized to [0, 1]
        sigma: Gaussian blur sigma
        low_threshold: low threshold for hysteresis (normalized 0-1)
        high_threshold: high threshold for hysteresis (normalized 0-1)

    Returns:
        Edge tensor (binary)
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    # Ensure values are in [0, 1]
    img_normalized = img_tensor / 255.0 if img_tensor.max() > 1 else img_tensor

    # Kornia canny returns (magnitude, edges)
    magnitude, edges = canny(
        img_normalized,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        kernel_size=(5, 5),
        sigma=(sigma, sigma),
        hysteresis=True
    )

    return edges


def gpu_sauvola_threshold(img_tensor, window_size, k=0.5, r=128.0):
    """
    GPU-accelerated Sauvola local thresholding.

    Sauvola formula: T = mean * (1 + k * ((std / r) - 1))

    Uses efficient convolution-based local mean and variance calculation.

    Args:
        img_tensor: (1, 1, H, W) tensor with grayscale values
        window_size: size of local window
        k: Sauvola parameter (default 0.5)
        r: dynamic range parameter (default 128 for 8-bit images)

    Returns:
        Threshold tensor of same shape
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    padding = window_size // 2

    # Local mean using average pooling
    local_mean = F.avg_pool2d(
        img_tensor,
        kernel_size=window_size,
        stride=1,
        padding=padding
    )

    # Local mean of squared values
    local_mean_sq = F.avg_pool2d(
        img_tensor ** 2,
        kernel_size=window_size,
        stride=1,
        padding=padding
    )

    # Local variance: E[X^2] - E[X]^2
    local_var = local_mean_sq - local_mean ** 2
    local_var = torch.clamp(local_var, min=0)  # Ensure non-negative
    local_std = torch.sqrt(local_var)

    # Sauvola threshold
    threshold = local_mean * (1 + k * ((local_std / r) - 1))

    return threshold


def gpu_niblack_threshold(img_tensor, window_size, k=-0.2):
    """
    GPU-accelerated Niblack local thresholding.

    Niblack formula: T = mean + k * std

    Args:
        img_tensor: (1, 1, H, W) tensor with grayscale values
        window_size: size of local window
        k: Niblack parameter (typically negative, default -0.2)

    Returns:
        Threshold tensor of same shape
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    padding = window_size // 2

    # Local mean
    local_mean = F.avg_pool2d(
        img_tensor,
        kernel_size=window_size,
        stride=1,
        padding=padding
    )

    # Local variance
    local_mean_sq = F.avg_pool2d(
        img_tensor ** 2,
        kernel_size=window_size,
        stride=1,
        padding=padding
    )
    local_var = local_mean_sq - local_mean ** 2
    local_var = torch.clamp(local_var, min=0)
    local_std = torch.sqrt(local_var)

    # Niblack threshold
    threshold = local_mean + k * local_std

    return threshold


def gpu_iou(mask_ref_tensor, mask_pred_tensor):
    """
    GPU-accelerated IoU (Intersection over Union) calculation.

    Args:
        mask_ref_tensor: reference mask tensor (any shape)
        mask_pred_tensor: predicted mask tensor (same shape)

    Returns:
        tuple: (iou, tpr, ppv) as Python floats
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available")

    # Binarize masks
    ref = (mask_ref_tensor > 0).float()
    pred = (mask_pred_tensor > 0).float()

    # Calculate intersection and union
    intersection = (ref * pred).sum()
    union = ((ref + pred) > 0).float().sum()

    # IoU
    iou = intersection / (union + 1e-8)

    # TPR (True Positive Rate / Recall) and PPV (Positive Predictive Value / Precision)
    true_positive = intersection
    false_negative = ((ref - pred) > 0).float().sum()
    false_positive = ((pred - ref) > 0).float().sum()

    tpr = true_positive / (true_positive + false_negative + 1e-8)
    ppv = true_positive / (true_positive + false_positive + 1e-8)

    return iou.item(), tpr.item(), ppv.item()


def gpu_create_binary_mask(img_tensor, threshold_tensor, dilation_size, erosion_size=None):
    """
    GPU-accelerated binary mask creation with morphological operations.

    Args:
        img_tensor: grayscale image tensor (1, 1, H, W)
        threshold_tensor: threshold values tensor (same shape)
        dilation_size: dilation kernel size
        erosion_size: erosion kernel size (optional)

    Returns:
        Binary mask tensor
    """
    # Threshold
    binary = (img_tensor > threshold_tensor).float()

    # Invert
    binary = 1.0 - binary

    # Morphological operations
    if erosion_size is not None and erosion_size > 0:
        binary = gpu_erosion(binary * 255, erosion_size, 1) / 255.0

    if dilation_size > 0:
        binary = gpu_dilation(binary * 255, dilation_size, 1) / 255.0

    return binary


def gpu_calculate_canny_edges(img_tensor, std_k, sigma):
    """
    GPU-accelerated Canny edge calculation with automatic threshold determination.

    Args:
        img_tensor: grayscale image tensor (1, 1, H, W), values 0-255
        std_k: standard deviation multiplier for threshold
        sigma: Gaussian blur sigma

    Returns:
        Edge tensor (binary)
    """
    # Calculate statistics
    mean_val = img_tensor.mean()
    std_val = img_tensor.std()

    # Calculate thresholds (normalized to 0-1)
    low_threshold = (mean_val - std_k * std_val / 2) / 255.0
    high_threshold = (mean_val + std_k * std_val / 2) / 255.0

    # Clamp thresholds to valid range
    low_threshold = torch.clamp(low_threshold, 0.0, 1.0)
    high_threshold = torch.clamp(high_threshold, 0.0, 1.0)

    return gpu_canny(img_tensor, sigma, low_threshold.item(), high_threshold.item())


# Convenience function for batch processing
def batch_process_images(images, process_fn, batch_size=8):
    """
    Process multiple images in batches on GPU.

    Args:
        images: list of numpy arrays (H, W)
        process_fn: function that takes batched tensor and returns processed tensor
        batch_size: number of images per batch

    Returns:
        list of numpy arrays
    """
    results = []
    device = get_device()

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

        # Stack into batch tensor
        batch_tensor = torch.stack([
            torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
            for img in batch
        ]).to(device)

        # Process batch
        with torch.no_grad():
            processed = process_fn(batch_tensor)

        # Convert back to list of numpy arrays
        for j in range(processed.shape[0]):
            results.append(tensor_to_numpy(processed[j]))

    return results
