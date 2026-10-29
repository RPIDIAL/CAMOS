"""Dense facial landmark extraction from a 3D face mesh (STL).

Procedure (identical to the one used to build the CAMOS training data):
  1. Render the mesh off-screen with VTK from a grid of camera pitch/roll angles.
  2. Run MediaPipe FaceLandmarker (478 points) on every rendered image.
  3. Back-project the 2D points of every view onto the mesh with a VTK picker.
     Only the landmark subset that is reliable for that view (``81sections.json``)
     is kept, lip points are only trusted from views with pitch <= ``rot_th``,
     and picks that land on the background are discarded.
  4. Merge the views with a per-point ``nanmedian`` and snap the result onto the
     nearest mesh vertex.
"""
import json
import math
import os

import numpy as np
import trimesh
import vtk
from scipy.spatial import KDTree

from .indices import lip_idx

N_MP_LANDMARKS = 478
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
DEFAULT_TASK = os.path.join(ASSET_DIR, "face_landmarker_v2_with_blendshapes.task")
DEFAULT_SECTIONS = os.path.join(ASSET_DIR, "81sections.json")


# --------------------------------------------------------------------------- #
# camera / rendering
# --------------------------------------------------------------------------- #
def rotate_position(position, focal_point, pitch_deg, roll_deg):
    """Rotate the camera position around the focal point (pitch about x, roll about z)."""
    direction = position - focal_point
    pitch, roll = math.radians(pitch_deg), math.radians(roll_deg)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch), math.cos(pitch)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll), math.cos(roll), 0],
                   [0, 0, 1]])
    return focal_point + (Rx @ Rz) @ direction


def default_camera(vertices, c_dist):
    """Camera placed in front of the face (-y side), looking at the centroid, +z up."""
    v_mean, v_max, v_min = vertices.mean(0), vertices.max(0), vertices.min(0)
    position = np.array([v_mean[0], v_mean[1] - (v_max[0] - v_min[0]) * c_dist, v_mean[2]])
    return position, v_mean.copy()


def render_view(stl_path, vertices, window_size, rot1, rot2, c_dist):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.98, 0.83, 0.76)

    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(window_size, window_size)
    window.AddRenderer(renderer)

    position, focal = default_camera(vertices, c_dist)
    camera = vtk.vtkCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(*focal)
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(*rotate_position(position, focal, rot1, rot2))

    renderer.AddActor(actor)
    renderer.SetActiveCamera(camera)
    renderer.SetBackground(0.68, 0.85, 0.9)
    window.Render()
    return renderer, window


def capture_png(window, save_path):
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(window)
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(save_path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


# --------------------------------------------------------------------------- #
# MediaPipe
# --------------------------------------------------------------------------- #
class FaceLandmarker:
    def __init__(self, task_path=DEFAULT_TASK):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=task_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, img_path):
        """Return (478, 2) pixel coordinates, or None when no face is found."""
        image = self.mp.Image.create_from_file(img_path)
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None
        h, w = image.numpy_view().shape[:2]
        lm = result.face_landmarks[0]
        return np.array([[p.x * w, p.y * h] for p in lm])


# --------------------------------------------------------------------------- #
# 2D -> 3D
# --------------------------------------------------------------------------- #
def read_zbuffer(window):
    """Depth buffer of the rendered frame as a (H, W) array (row 0 = bottom, OpenGL convention).

    It is read right after rendering, before MediaPipe runs, because MediaPipe creates its own
    GL context and the VTK off-screen context can no longer be read reliably afterwards."""
    from vtk.util.numpy_support import vtk_to_numpy
    w, h = window.GetSize()
    arr = vtk.vtkFloatArray()
    window.GetZbufferData(0, 0, w - 1, h - 1, arr)
    return vtk_to_numpy(arr).reshape(h, w).copy()


def pick_3d(renderer, zbuffer, points_2d):
    """World coordinates of the surface under each pixel (NaN where the pixel shows background).

    Same computation as vtkWorldPointPicker (and therefore the same positions as the hardware
    vtkPropPicker used originally): depth from the z-buffer, then display -> world through the
    camera, but without re-rendering the scene for every point."""
    h, w = zbuffer.shape
    out = np.full((len(points_2d), 3), np.nan)
    for i, (x, y) in enumerate(points_2d):
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        z = float(zbuffer[yi, xi])
        if z >= 1.0:                      # far plane = background
            continue
        renderer.SetDisplayPoint(float(x), float(y), z)
        renderer.DisplayToWorld()
        wx, wy, wz, ww = renderer.GetWorldPoint()
        if ww != 0:
            out[i] = (wx / ww, wy / ww, wz / ww)
    return out


def backproject_view(renderer, zbuffer, points_2d, vertices, valid_idx, rot1, rot2, c_dist, window_size):
    """Map the 2D landmarks of one view back onto the mesh. Returns (478, 3) with NaNs."""
    pts = points_2d.copy()
    pts[:, 1] = window_size - pts[:, 1]          # image row -> OpenGL y
    picked = pick_3d(renderer, zbuffer, pts[valid_idx])
    # drop picks that hit the background far behind the face
    position, focal = default_camera(vertices, c_dist)
    cam = rotate_position(position, focal, rot1, rot2)
    thr = math.dist(cam, focal) * 1.5
    far = np.linalg.norm(picked - cam, axis=1) >= thr
    picked[far] = np.nan
    out = np.full((N_MP_LANDMARKS, 3), np.nan)
    out[valid_idx] = picked
    return out


def view_valid_indices(sections, idx1, idx2, rot1, seg_nb, rot_th):
    valid = [i for i in sections[f"section{idx1 * seg_nb + idx2}"] if i not in lip_idx]
    if rot1 <= rot_th:
        valid = list(set(valid + lip_idx))
    return np.array(valid)


# --------------------------------------------------------------------------- #
# 3D Slicer markup json helpers
# --------------------------------------------------------------------------- #
def write_markups(points, path, labels=None):
    pts = np.asarray(points, dtype=float)
    cps = []
    for i, p in enumerate(pts):
        cps.append({
            "id": f"vtkMRMLMarkupsFiducialNode_{i}",
            "label": labels[i] if labels is not None else f"P{i + 1}",
            "description": "", "associatedNodeID": "",
            "position": [None if np.isnan(v) else float(v) for v in p],
            "orientation": [-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
            "selected": True, "locked": False, "visibility": True,
            "positionStatus": "undefined" if np.isnan(p).any() else "defined"})
    data = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
            "markups": [{"type": "Fiducial", "coordinateSystem": "LPS", "coordinateUnits": "mm",
                         "locked": False, "fixedNumberOfControlPoints": False, "labelFormat": "%N-%d",
                         "lastUsedControlPointNumber": len(cps), "controlPoints": cps,
                         "display": {"color": [1.0, 0.0, 0.0], "glyphScale": 1.0, "textScale": 0.0}}]}
    with open(path, "w") as f:
        json.dump(data, f)


def read_markups(path):
    with open(path) as f:
        cps = json.load(f)["markups"][0]["controlPoints"]
    return np.array([[np.nan if v is None else v for v in c["position"]] for c in cps], dtype=float)


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def extract_landmarks(stl_path, work_dir, rot_range=40, rot_interval=10, rot_th=20,
                      sparse_views=False, window_size=800, c_dist=3.0,
                      task_path=DEFAULT_TASK, sections_path=DEFAULT_SECTIONS,
                      keep_images=False, log=print):
    """Return (landmarks_478_raw, landmarks_478_on_surface) in the mesh coordinate frame.

    ``sparse_views=True`` skips the +-10 and +-30 degree views (25 instead of 81 views),
    which is the setting used for the public pretraining datasets; the patient model was
    fine-tuned with the full 81-view grid (default).
    """
    os.makedirs(work_dir, exist_ok=True)
    mesh = trimesh.load(stl_path, force="mesh")
    vertices = np.asarray(mesh.vertices)
    with open(sections_path) as f:
        sections = json.load(f)
    seg_nb = int(rot_range / rot_interval * 2 + 1)
    assert f"section{seg_nb * seg_nb - 1}" in sections, "sections file does not match the view grid"
    detector = FaceLandmarker(task_path)

    angles = list(range(-rot_range, rot_range + 1, rot_interval))
    per_view = []
    n_fail = 0
    for idx1, rot1 in enumerate(angles):
        for idx2, rot2 in enumerate(angles):
            if sparse_views and (abs(rot1) in (10, 30) or abs(rot2) in (10, 30)):
                continue
            img = os.path.join(work_dir, f"{rot1}_{rot2}.png")
            renderer, window = render_view(stl_path, vertices, window_size, rot1, rot2, c_dist)
            try:
                capture_png(window, img)
                # Re-render with a tight clipping range before reading the depth buffer: with the
                # default (0.01, 1000) range the depth is quantized to several millimetres.
                renderer.ResetCameraClippingRange()
                window.Render()
                zbuffer = read_zbuffer(window)
                pts2d = detector.detect(img)
                if pts2d is None:
                    n_fail += 1
                    log(f"  view ({rot1:+d},{rot2:+d}): no face detected, skipped")
                    continue
                valid = view_valid_indices(sections, idx1, idx2, rot1, seg_nb, rot_th)
                pts3d = backproject_view(renderer, zbuffer, pts2d, vertices, valid, rot1, rot2, c_dist, window_size)
                per_view.append(pts3d)
                np.save(os.path.join(work_dir, f"{rot1}_{rot2}_mapping.npy"), pts3d)
            finally:
                window.Finalize()
                if not keep_images and os.path.exists(img):
                    os.remove(img)
    if not per_view:
        raise RuntimeError("MediaPipe found no face in any view. Check the mesh orientation "
                           "(face must look toward -y with +z up) and units (mm).")
    log(f"  {len(per_view)} views merged ({n_fail} without detection)")
    with np.errstate(all="ignore"):
        merged = np.nanmedian(np.stack(per_view), axis=0)
    ok = np.all(np.isfinite(merged), axis=1)
    on_surface = np.full_like(merged, np.nan)
    _, nn = KDTree(vertices).query(merged[ok])
    on_surface[ok] = vertices[nn]
    return merged, on_surface
