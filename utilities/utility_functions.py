import numpy as np


def bilinear_interpolate(img: np.ndarray, x: float, y: float) -> float:
    """
    Perform bilinear interpolation for a single point (x, y) on the image.

    Parameters:
        img (np.ndarray): The input image.
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.

    Returns:
        float: The interpolated intensity value at position (x, y).
    """
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, img.shape[1] - 1)
    y0 = int(np.floor(y))
    y1 = min(y0 + 1, img.shape[0] - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * img[y0, x0] + wb * img[y1, x0] + wc * img[y0, x1] + wd * img[y1, x1]