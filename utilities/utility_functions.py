import numpy as np


def bilinear_interpolate(img, x, y):
    """
    Perform bilinear interpolation for a single point (x, y) on the image img.
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