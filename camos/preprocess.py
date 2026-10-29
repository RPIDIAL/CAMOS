"""478 -> 282 landmark selection and the normalization expected by the network."""
import numpy as np

from .indices import all_idx, GB_ID, PRN_ID, RIGHT_EYE_ID, LEFT_EYE_ID


def select_282(land478):
    pts = np.asarray(land478, dtype=float)[all_idx]
    bad = np.where(~np.all(np.isfinite(pts), axis=1))[0]
    if len(bad):
        ids = [all_idx[i] for i in bad]
        raise RuntimeError(f"{len(bad)} of the 282 model landmarks were not recovered from any view "
                           f"(MediaPipe ids {ids[:20]}{'...' if len(ids) > 20 else ''}). "
                           "Try the full 81-view grid or check the mesh orientation.")
    return pts


def normalization_stats(land282):
    """Center on glabella-x / eye-midpoint-y / pronasale-z and scale by the max abs extent.

    During training these statistics came from the (symmetrized) post-operative landmarks;
    for a new patient only the pre-operative face is available, so they are computed from it.
    """
    pos = {i: all_idx.index(i) for i in (GB_ID, PRN_ID, RIGHT_EYE_ID, LEFT_EYE_ID)}
    x_mean = float(land282[pos[GB_ID], 0])
    y_mean = float((land282[pos[RIGHT_EYE_ID], 1] + land282[pos[LEFT_EYE_ID], 1]) / 2)
    z_mean = float(land282[pos[PRN_ID], 2])
    centered = land282 - (x_mean, y_mean, z_mean)
    all_std = float(np.max(np.abs(centered)))
    return {"x_mean": x_mean, "y_mean": y_mean, "z_mean": z_mean, "all_std": all_std}


def normalize(land282, stats):
    return (land282 - (stats["x_mean"], stats["y_mean"], stats["z_mean"])) / stats["all_std"]


def denormalize(norm282, stats):
    return norm282 * stats["all_std"] + (stats["x_mean"], stats["y_mean"], stats["z_mean"])
