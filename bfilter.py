import numpy as np
from scipy import signal


def gaussian_kernel(n, std, normalised=True):
    """
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1."""
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= gaussian2D.sum()
    return gaussian2D


# img - gray image with [0 1] values
# halfSize - half of kernel's width(=height)
# sigmaD - Domain filter variance
# sigmaR - range filter variance
def bfilter(img, halfSize, sigmaD, sigmaR):

    H, W = img.shape

    #  Gaussian kernel for domain filter
    DKernel = gaussian_kernel(2 * halfSize + 1, sigmaD)

    resultImg = np.zeros([H, W], np.float32)
    for y in range(H):
        for x in range(W):
            # Set local region
            xMin = max(x - halfSize, 0)
            xMax = min(x + halfSize + 1, W)
            yMin = max(y - halfSize, 0)
            yMax = min(y + halfSize + 1, H)
            localImg = img[yMin:yMax, xMin:xMax]

            # Gaussian kernel for range filter
            RKernel = np.exp(-0.5 * ((localImg - img[y, x]) / sigmaR) ** 2)

            # Bilateral Filtering
            BKernel = (
                RKernel
                * DKernel[
                    yMin - y + halfSize : yMax - y + halfSize,
                    xMin - x + halfSize : xMax - x + halfSize,
                ]
            )
            resultImg[y, x] = (localImg * BKernel).sum() / BKernel.sum()

    return resultImg
