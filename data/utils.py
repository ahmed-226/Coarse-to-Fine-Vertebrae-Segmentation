"""
Utility functions for data loading and processing.
"""
import numpy as np
import SimpleITK as sitk
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.ndimage import gaussian_filter, zoom


def load_image(path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
    """
    Load a medical image (NIfTI format).
    
    Args:
        path: Path to .nii or .nii.gz file
    
    Returns:
        image_array: NumPy array [Z, Y, X] (SimpleITK convention)
        metadata: Dict with spacing, origin, direction, size
    """
    image = sitk.ReadImage(str(path))
    
    metadata = {
        'spacing': np.array(image.GetSpacing()),      # [X, Y, Z]
        'origin': np.array(image.GetOrigin()),        # [X, Y, Z]
        'direction': np.array(image.GetDirection()).reshape(3, 3),
        'size': np.array(image.GetSize()),            # [X, Y, Z]
    }
    
    # Convert to numpy array [Z, Y, X]
    image_array = sitk.GetArrayFromImage(image).astype(np.float32)
    
    return image_array, metadata


def save_image(
    image_array: np.ndarray,
    path: Union[str, Path],
    metadata: Optional[Dict] = None,
    reference_image: Optional[sitk.Image] = None
) -> None:
    """
    Save a NumPy array as a medical image.
    
    Args:
        image_array: NumPy array [Z, Y, X]
        path: Output path
        metadata: Dict with spacing, origin, direction
        reference_image: Reference SimpleITK image for metadata
    """
    image = sitk.GetImageFromArray(image_array)
    
    if reference_image is not None:
        image.CopyInformation(reference_image)
    elif metadata is not None:
        image.SetSpacing(metadata['spacing'].tolist())
        image.SetOrigin(metadata['origin'].tolist())
        image.SetDirection(metadata['direction'].flatten().tolist())
    
    sitk.WriteImage(image, str(path))


def load_landmarks(json_path: Union[str, Path]) -> Tuple[Dict[int, np.ndarray], Optional[List[str]]]:
    """
    Load vertebra centroids from VerSe JSON format.
    
    VerSe JSON format:
    [
        {"direction": ["P", "I", "R"]},  # Optional direction info
        {"label": 17, "X": 94.8, "Y": 46.1, "Z": 19.1},
        {"label": 18, "X": 90.3, "Y": 71.7, "Z": 18.6},
        ...
    ]
    
    Label mapping (VerSe):
        C1=1...C7=7, T1=8...T12=19, L1=20...L5=24, L6=25, S1=26
    
    Returns:
        landmarks: Dict mapping vertebra label to centroid [X, Y, Z] in physical coordinates
        direction: Direction info if present (e.g., ["P", "I", "R"])
    """
    with open(str(json_path), 'r') as f:
        data = json.load(f)
    
    landmarks = {}
    direction = None
    
    for entry in data:
        # Check for direction entry (first item in some VerSe files)
        if 'direction' in entry:
            direction = entry['direction']
            continue
        
        # Skip entries without label
        if 'label' not in entry:
            continue
            
        label = int(entry['label'])
        # Physical coordinates [X, Y, Z]
        centroid = np.array([
            float(entry['X']),
            float(entry['Y']),
            float(entry['Z'])
        ])
        landmarks[label] = centroid
    
    return landmarks, direction


# ============================================================
# Label Mapping for VerSe Dataset
# ============================================================
LABEL_TO_VERTEBRA = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12',
    20: 'L1', 21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6',
    26: 'S1', 27: 'S2', 28: 'Sacrum'
}

VERTEBRA_TO_LABEL = {v: k for k, v in LABEL_TO_VERTEBRA.items()}


def get_vertebra_name(label: int) -> str:
    """Convert numeric label to vertebra name."""
    return LABEL_TO_VERTEBRA.get(label, f'V{label}')


def get_vertebra_label(name: str) -> int:
    """Convert vertebra name to numeric label."""
    return VERTEBRA_TO_LABEL.get(name, 0)


def physical_to_voxel(
    physical_coords: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert physical coordinates to voxel indices.
    
    Args:
        physical_coords: [X, Y, Z] or [N, 3] array of physical coordinates
        spacing: [X, Y, Z] voxel spacing in mm
        origin: [X, Y, Z] image origin in physical space
        direction: 3x3 direction matrix (optional, assumes identity)
    
    Returns:
        Voxel indices [X, Y, Z] or [N, 3]
    """
    if direction is not None and not np.allclose(direction, np.eye(3)):
        # Apply inverse direction matrix
        inv_direction = np.linalg.inv(direction)
        shifted = physical_coords - origin
        if shifted.ndim == 1:
            rotated = inv_direction @ shifted
        else:
            rotated = (inv_direction @ shifted.T).T
        voxel_coords = rotated / spacing
    else:
        voxel_coords = (physical_coords - origin) / spacing
    
    return voxel_coords


def voxel_to_physical(
    voxel_coords: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    direction: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert voxel indices to physical coordinates.
    
    Args:
        voxel_coords: [X, Y, Z] or [N, 3] array of voxel indices
        spacing: [X, Y, Z] voxel spacing in mm
        origin: [X, Y, Z] image origin in physical space
        direction: 3x3 direction matrix (optional, assumes identity)
    
    Returns:
        Physical coordinates [X, Y, Z] or [N, 3]
    """
    if direction is not None and not np.allclose(direction, np.eye(3)):
        scaled = voxel_coords * spacing
        if scaled.ndim == 1:
            rotated = direction @ scaled
        else:
            rotated = (direction @ scaled.T).T
        physical_coords = rotated + origin
    else:
        physical_coords = voxel_coords * spacing + origin
    
    return physical_coords


def resample_image(
    image: np.ndarray,
    original_spacing: np.ndarray,
    target_spacing: np.ndarray,
    target_size: Optional[Tuple[int, int, int]] = None,
    order: int = 1,
    mode: str = 'constant',
    cval: float = 0.0
) -> np.ndarray:
    """
    Resample image to target spacing/size.
    
    Args:
        image: Input array [Z, Y, X]
        original_spacing: Current spacing [X, Y, Z]
        target_spacing: Desired spacing [X, Y, Z]
        target_size: Desired size [X, Y, Z] (optional)
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)
        mode: Boundary mode ('constant', 'nearest', etc.)
        cval: Constant value for 'constant' mode
    
    Returns:
        Resampled array [Z', Y', X']
    """
    # Compute zoom factors (note: image is [Z, Y, X], spacing is [X, Y, Z])
    zoom_factors = np.array([
        original_spacing[2] / target_spacing[2],  # Z
        original_spacing[1] / target_spacing[1],  # Y
        original_spacing[0] / target_spacing[0],  # X
    ])
    
    if target_size is not None:
        # Override zoom factors to match target size
        current_size = np.array([image.shape[2], image.shape[1], image.shape[0]])  # [X, Y, Z]
        zoom_factors = np.array([
            target_size[2] / image.shape[0],  # Z
            target_size[1] / image.shape[1],  # Y
            target_size[0] / image.shape[2],  # X
        ])
    
    resampled = zoom(image, zoom_factors, order=order, mode=mode, cval=cval)
    
    return resampled


def crop_image(
    image: np.ndarray,
    center: np.ndarray,
    output_size: Tuple[int, int, int],
    pad_value: float = 0.0
) -> np.ndarray:
    """
    Crop image around center with padding if needed.
    
    Args:
        image: Input array [Z, Y, X]
        center: Crop center in voxel coordinates [X, Y, Z]
        output_size: Output size [X, Y, Z]
        pad_value: Padding value for out-of-bounds regions
    
    Returns:
        Cropped array [Z', Y', X'] with shape matching output_size (reordered)
    """
    # Convert center to [Z, Y, X] order
    center_zyx = np.array([center[2], center[1], center[0]])
    size_zyx = np.array([output_size[2], output_size[1], output_size[0]])
    
    # Calculate start and end indices
    half_size = size_zyx // 2
    start = (center_zyx - half_size).astype(int)
    end = start + size_zyx
    
    # Calculate padding needed
    pad_before = np.maximum(-start, 0)
    pad_after = np.maximum(end - np.array(image.shape), 0)
    
    # Adjust start/end to valid range
    start_valid = np.maximum(start, 0)
    end_valid = np.minimum(end, np.array(image.shape))
    
    # Extract valid region
    crop = image[
        start_valid[0]:end_valid[0],
        start_valid[1]:end_valid[1],
        start_valid[2]:end_valid[2]
    ]
    
    # Pad if necessary
    if np.any(pad_before > 0) or np.any(pad_after > 0):
        pad_width = [
            (pad_before[0], pad_after[0]),
            (pad_before[1], pad_after[1]),
            (pad_before[2], pad_after[2])
        ]
        crop = np.pad(crop, pad_width, mode='constant', constant_values=pad_value)
    
    return crop


def generate_gaussian_heatmap(
    size: Tuple[int, int, int],
    center: np.ndarray,
    sigma: float,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a 3D Gaussian heatmap.
    
    Args:
        size: Output size [X, Y, Z]
        center: Gaussian center in voxel coordinates [X, Y, Z]
        sigma: Gaussian sigma in voxels
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Heatmap array [Z, Y, X]
    """
    # Create coordinate grids
    x = np.arange(size[0])
    y = np.arange(size[1])
    z = np.arange(size[2])
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Compute squared distance from center
    dist_sq = (
        (xx - center[0]) ** 2 +
        (yy - center[1]) ** 2 +
        (zz - center[2]) ** 2
    )
    
    # Gaussian
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))
    
    # Transpose to [Z, Y, X]
    heatmap = heatmap.transpose(2, 1, 0)
    
    if normalize:
        heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return heatmap.astype(np.float32)


def generate_multi_landmark_heatmap(
    size: Tuple[int, int, int],
    landmarks: Dict[int, np.ndarray],
    sigma: float,
    num_classes: int = 26
) -> np.ndarray:
    """
    Generate multi-channel heatmap for all landmarks.
    
    Args:
        size: Output size [X, Y, Z]
        landmarks: Dict mapping label to voxel coordinates [X, Y, Z]
        sigma: Gaussian sigma in voxels
        num_classes: Number of output channels
    
    Returns:
        Heatmap array [num_classes, Z, Y, X]
    """
    heatmaps = np.zeros((num_classes, size[2], size[1], size[0]), dtype=np.float32)
    
    for label, center in landmarks.items():
        if 1 <= label <= num_classes:
            idx = label - 1  # Convert 1-indexed to 0-indexed
            heatmaps[idx] = generate_gaussian_heatmap(size, center, sigma)
    
    return heatmaps


def normalize_intensity(
    image: np.ndarray,
    min_val: float = -1024.0,
    max_val: float = 4096.0
) -> np.ndarray:
    """
    Normalize image intensity to [0, 1].
    
    Args:
        image: Input image array
        min_val: Minimum HU value (clipped below)
        max_val: Maximum HU value (clipped above)
    
    Returns:
        Normalized image in [0, 1]
    """
    # Clip to valid range
    image = np.clip(image, min_val, max_val)
    
    # Normalize to [0, 1]
    image = (image - min_val) / (max_val - min_val)
    
    return image.astype(np.float32)


def compute_spine_center(landmarks: Dict[int, np.ndarray]) -> np.ndarray:
    """
    Compute spine center as mean of all landmark positions.
    
    Args:
        landmarks: Dict mapping label to coordinates [X, Y, Z]
    
    Returns:
        Spine center [X, Y, Z]
    """
    if len(landmarks) == 0:
        raise ValueError("No landmarks provided")
    
    positions = np.array(list(landmarks.values()))
    center = positions.mean(axis=0)
    
    return center


def get_valid_vertebrae_mask(landmarks: Dict[int, np.ndarray], num_classes: int = 26) -> np.ndarray:
    """
    Create binary mask indicating which vertebrae are present.
    
    Args:
        landmarks: Dict mapping label to coordinates
        num_classes: Total number of vertebra classes
    
    Returns:
        Binary mask [num_classes] with 1 for present vertebrae
    """
    mask = np.zeros(num_classes, dtype=np.float32)
    
    for label in landmarks.keys():
        if 1 <= label <= num_classes:
            mask[label - 1] = 1.0
    
    return mask
