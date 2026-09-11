"""Data loading and label parsing for the ICCAD hotspot dataset."""
import os
from PIL import Image

ROOT = "/content/raw/iccad-official"
BENCHMARKS = ["iccad1", "iccad2", "iccad3", "iccad4", "iccad5"]

def label_from_filename(fname: str) -> int:
    """Returns 1 for hotspot, 0 for non-hotspot, based on filename convention.
    Handles both HS/NHS (iccad1) and HSCADn/NHSCADn (iccad2-5) naming,
    plus the doubled-N prefix quirk found in some test-set files."""
    if "NHS" in fname:
        return 0
    elif "HS" in fname:
        return 1
    raise ValueError(f"Unrecognized filename pattern: {fname}")

def load_clip(root: str, benchmark: str, split: str, filename: str) -> Image.Image:
    """Loads a single layout clip image."""
    path = os.path.join(root, benchmark, split, filename)
    return Image.open(path)
