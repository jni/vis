"""
imshows.py: wrappers around matplotlib.pyplot.imshow with saner
defaults for 3D scientific images.
"""
from matplotlib import pyplot as plt, cm


def cshow(im):
    """Show an image (or cross-section) with the cubehelix colormap.

    Parameters
    ----------
    im : array
        An array of intensity values (ie not multichannel).

    Returns
    -------
    ax : matplotlib AxesImage object
        The figure axes.

    Notes
    -----
    For nD images with n > 2, ``cshow`` repeatedly takes the middle
    cross- section of leading axes until a 2D image remains. For
    example, given an array `im` of shape ``(7, 512, 512)``, ``cshow``
    will display `im[3]`. For shape ``(4, 50, 50, 50)``, ``cshow`` will
    display `im[2, 25]`.
    """
    if im.ndim > 2:
        mid = im.shape[0] // 2
        ax = cshow(im[mid])
    else:
        ax = plt.imshow(im, cmap=cm.cubehelix, interpolation='nearest')
    return ax

