"""Thin-plate-spline surface reconstruction from predicted landmarks."""
import numpy as np

from .indices import TPS_EXCLUDE_POS
from .tps import TPS


def warp_mesh(mesh, pre282, pred282, lambda_=0.001):
    """Deform ``mesh`` (pre-operative surface) so that the pre-operative landmarks move onto
    the predicted ones. The inner-lip control points are excluded as in the original code."""
    keep = [i for i in range(len(pre282)) if i not in TPS_EXCLUDE_POS]
    tps = TPS(np.asarray(pre282)[keep], np.asarray(pred282)[keep], lambda_=lambda_)
    out = mesh.copy()
    out.vertices = tps(np.asarray(mesh.vertices, dtype=float))
    return out
