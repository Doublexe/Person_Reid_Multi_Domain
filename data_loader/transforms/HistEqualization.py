import cv2
import numpy as np
from PIL import Image

def hisEqulColor(img):
    img_yuv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img)


class HistEqualization(object):
    def __call__(self, img):
        """
        Parameters
        ----------
        img : ndarray
            The unnormalized image tensor in format (H, W, C) #RGB.
        """
        return hisEqulColor(img)
