"""Canonical alignment and optional face cropping, driven only by the dense landmarks."""
import numpy as np
import trimesh

from .indices import ALIGN_IDS, CROP_GB_ID, CROP_GOR_ID, CROP_GOL_ID, CROP_CER_ID


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def canonical_transform(land478):
    """4x4 rigid transform that moves the face into the frame used for training.

    The nasion-area point (MediaPipe 6) becomes the origin and the line between the two
    cheek points (117 -> 346) is rotated onto the +x axis (zero y and z components).
    Pitch (rotation about x) is left unchanged, so the input scan must already be in an
    upright, natural-head-position-like orientation.
    """
    pts = land478[ALIGN_IDS] - land478[ALIGN_IDS[1]]
    v = pts[3] - pts[2]
    Rz = _rot_z(-np.arctan2(v[1], v[0]))
    v2 = Rz @ v
    Ry = _rot_y(np.arctan2(v2[2], v2[0]))
    R = Ry @ Rz
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ land478[ALIGN_IDS[1]]
    return M


def apply_transform(points, M):
    pts = np.asarray(points, dtype=float)
    out = np.full_like(pts, np.nan)
    ok = np.all(np.isfinite(pts), axis=1)
    out[ok] = pts[ok] @ M[:3, :3].T + M[:3, 3]
    return out


def orientation_report(land478):
    """Heuristic checks that the face looks toward -y and is upright (+z)."""
    nose, glabella, forehead, chin = land478[4], land478[9], land478[151], land478[152]
    msgs = []
    if not nose[1] < glabella[1]:
        msgs.append("nose tip is not in front of the glabella along -y (face should look toward -y)")
    if not forehead[2] > chin[2]:
        msgs.append("forehead is not above the chin along +z (head should point to +z)")
    return msgs


def crop_face(mesh, land478):
    """Keep the facial region used by CAMOS: below the glabella, in front of and above the
    gonion/chin planes. Mirrors the clipping used to build the training meshes."""
    gb, gor, gol = land478[CROP_GB_ID], land478[CROP_GOR_ID], land478[CROP_GOL_ID]
    cer = land478[CROP_CER_ID] - np.array([0, 0, 10.0])
    m = mesh.copy()
    m = trimesh.intersections.slice_mesh_plane(m, plane_normal=[0, 0, -1], plane_origin=gb)
    n2 = -np.cross((gor + [0, 0, 1]) - gor, gol - gor)
    m = trimesh.intersections.slice_mesh_plane(m, plane_normal=n2, plane_origin=gor)
    n3 = np.cross(cer - gor, gol - gor)
    m = trimesh.intersections.slice_mesh_plane(m, plane_normal=n3, plane_origin=gor)
    parts = m.split(only_watertight=False)
    if len(parts) > 1:
        m = max(parts, key=lambda p: len(p.faces))
    return m
