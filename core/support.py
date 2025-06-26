# glass/core/support.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.kernels import Kernel  # Ensure this import remains consistent across all modules and aligns with your file structure. This confirms the correct class-based usage per MGWR.


def smooth_source_to_target(source_vals, source_coords, target_coords, K=30, kernel_type='gaussian'):
    """
    Smooth values from source support onto target support using spatial kernel smoothing.
    """
    if len(source_vals) != len(source_coords):
        raise ValueError("Mismatch: 'source_vals' and 'source_coords' must have the same length.")

    smoothed_vals = np.zeros(len(target_coords))
    nbrs = NearestNeighbors(n_neighbors=K).fit(source_coords)
    dists, indices = nbrs.kneighbors(target_coords)

    for i, (distances, idx) in enumerate(zip(dists, indices)):
        sub_coords = source_coords[idx]
        bw = distances.max() if distances.max() > 0 else 1e-6  # Avoid division by zero

        kernel_obj = Kernel(0, sub_coords, bw=bw, fixed=True, function=kernel_type)
        weights = kernel_obj.kernel

        if weights.sum() == 0:
            smoothed_vals[i] = np.nan
        else:
            smoothed_vals[i] = np.average(source_vals[idx].ravel(), weights=weights.ravel())

    return smoothed_vals
