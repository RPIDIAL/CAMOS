"""Tiny replacements for the batchgenerators file helpers used by the training scripts."""
import json
import os


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def maybe_mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def subfolders(folder, join=True, sort=True):
    out = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
    if sort:
        out.sort()
    return [os.path.join(folder, f) for f in out] if join else out


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    out = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
           and (prefix is None or f.startswith(prefix)) and (suffix is None or f.endswith(suffix))]
    if sort:
        out.sort()
    return [os.path.join(folder, f) for f in out] if join else out
