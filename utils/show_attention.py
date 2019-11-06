from matplotlib import pyplot as plt
import cv2


def get_colors(inp, colormap=plt.cm.jet, vmin=None, vmax=None):
    norm = plt.Normalize(vmin, vmax)
    return colormap(norm(inp))


def show_attention(feature_map, image, lambd=0.5, text=''):
    """Show attentions from different channels for a feature map.

    Parameters
    ----------
    out : ndarray
        The feature map output, in format (1, H, W)
    img : ndarray
        A torchvision image, in format (C, H, W)
    lambda : float
        between 0~1. The weight for the heatmap when combining with the image.
    text : str
        annotation text
    """

    heatmap = get_colors(cv2.resize(feature_map,
                                    (image.shape[2], image.shape[1])), plt.cm.jet)

    result = heatmap[:, :, :3][:, :, ::-1] * \
        lambd * 255 + image[:, :, ::-1] * (1-lambd) * 255

    cv2.putText(result, text,
                (0, image.shape[1]), 0, .5, (0, 255, 0), 2)

    return result
