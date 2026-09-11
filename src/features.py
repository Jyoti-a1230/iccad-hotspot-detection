"""Hand-crafted features used during EDA (transition counts)."""
import numpy as np
from PIL import Image

def count_transitions(img: Image.Image) -> tuple[float, float]:
    """Returns (horizontal_transitions, vertical_transitions) for a binary layout image."""
    array = np.array(img) / 255
    horizontal = np.sum(np.abs(array[:, 1:] - array[:, :-1]))
    vertical = np.sum(np.abs(array[1:, :] - array[:-1, :]))
    return horizontal, vertical
