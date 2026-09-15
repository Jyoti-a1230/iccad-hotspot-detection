"""Train/val splitting utilities that respect augmented base-pattern groups."""
import re

def get_base_pattern(fname: str) -> str:
    """Strips the augmentation-variant suffix and extension, so augmented
    copies of the same underlying layout pattern map to the same group."""
    stripped = re.sub(r'\.png(\d+)\.png$', '', fname)
    return stripped.replace('.png', '')
