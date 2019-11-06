import numpy as np
from PIL import Image

def clip_normalize(x, low, high):
    """ Clip lower and upper bounds for an array (color values), and then normalize to 0~1

    Parameters
    ----------
    x : ndarray, one-dimension
    low, high : float
        the bounds to clip
    """

    x[x>=high] = high
    x[x<=low] = low
    ma = x.max()
    mi = x.min()

    # If divid by 0, set to 0
    a = (x-mi)
    b = (ma-mi)
    if b == 0:
        x = np.zeros_like(a)
    else:
        x = np.divide(a, b)*255

    return x.astype(np.int32)


def simplestColorBalance(img, satLevel):
    """
    Parameter
    ---------
    img : ndarray, (H, W, C) #RGB
    satLevel : float, 0~1
        satLevel controls the percentage of pixels to clip to white and black (satLevel/2 for both white and black).
    """
    img = np.array(img)
    H, W, C = img.shape
    flat_img = img.reshape([H*W, C])

    lower_bound = satLevel/2
    upper_bound = 1-satLevel/2

    for ch in range(3):
        low = np.quantile(flat_img[:,ch], lower_bound)
        upp = np.quantile(flat_img[:,ch], upper_bound)
        flat_img[:,ch] = clip_normalize(flat_img[:,ch], low, upp)

    return Image.fromarray(flat_img.reshape([H, W, C]))


class ColorBalance:
    def __init__(self, satLevel):
        self.satLevel = satLevel

    def __call__(self, img):
        return simplestColorBalance(img, self.satLevel)
