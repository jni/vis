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


def _factors(n):
    """Return integer factors of `n`, not including 1 or `n`.

    Parameters
    ----------
    n : int
        Integer for which we want a factorization.

    Returns
    -------
    fs : list of int
        The list of factors of `n` (empty if `n` is prime).

    Examples
    --------
    >>> _factors(10)
    [2, 5]
    >>> _factors(20)
    [2, 4, 5, 10]
    """
    fs = filter(lambda i: (n % i == 0), range(2, 1 + n/2))
    return fs


def rshow(values):
    """Show a 1D vector of values in a rectangular grid.

    Parameters
    ----------
    values : 1D array
        The values to be plotted.

    Returns
    -------
    ax : matplotlib AxesImage object
        The figure axes.
    """
    n = len(values)
    fs = _factors(n)
    if len(fs) == 0:
        values_im = values.reshape((1, n))
    else:
        new_shape = (-1, fs[k // 2])
        values_im = values.reshape(new_shape)
        return cshow(values_im)

