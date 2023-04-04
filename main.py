from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.linalg import spsolve


def imread_float(path: str | Path) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path).astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def colorize(color_region, yuvIm):

    H, W, _ = yuvIm.shape
    num_pixels = H * W

    row_numbering = np.arange(num_pixels).reshape(H, W)

    # sparse matrix components
    Ai = []  # row of A = pixel number of window center
    Aj = []  # column of A = pixel number of pixels in window
    AVal = []  # value of A = weighting function between row and column

    windowR = 1  # window radius
    for y in range(H):
        for x in range(W):
            cur_pixel_number = row_numbering[y, x]  # current pixel order(column-wise)

            # when cur_pixel_number is scribbled already, pass.
            # They'll just have value of scribbled color.
            if not color_region[y, x]:
                wVal = []  # temporary repository for Y value of window.

                # transverse window whose center is cur_pixel_number. ignore outside of image.
                for yy in range(max(0, y - windowR), min(y + windowR + 1, H)):
                    for xx in range(max(0, x - windowR), min(x + windowR + 1, W)):
                        if yy == y and xx == x:
                            continue

                        Ai.append(cur_pixel_number)
                        Aj.append(row_numbering[yy, xx])
                        wVal.append(yuvIm[yy, xx, 0])

                centerY = yuvIm[y, x, 0]  # Y value of window center
                wVal.append(centerY)
                wVar = np.var(wVal)  # variance of Y in the window
                wVar = max(wVar, 2e-6)  # prevent division by 0

                # calculate weighting function
                weight_func = np.exp(-((wVal[:-1] - centerY) ** 2) / wVar)
                weight_func = weight_func / weight_func.sum()  # normalization of weight function
                AVal += list(-weight_func)

            Ai.append(cur_pixel_number)
            Aj.append(row_numbering[y, x])
            AVal.append(1)  # diagonal of A are all 1

    A = csr_array((AVal, (Ai, Aj)), shape=(num_pixels, num_pixels))  # make sparse matrix
    b = np.zeros(num_pixels)

    color_index = np.flatnonzero(color_region)

    result_img = np.zeros_like(yuvIm)
    result_img[:, :, 0] = yuvIm[:, :, 0]  # identical Y value with original
    for t in (1, 2):
        curIm = yuvIm[:, :, t]
        b[color_index] = curIm.flatten()[color_index]
        new_vals = spsolve(A, b)
        result_img[:, :, t] = new_vals.reshape(H, W)

    return result_img


def main():
    g_name = "example.bmp"  # gray image file name
    c_name = "example_marked.bmp"  # scribbled image file name
    out_name = "example_res.bmp"  # result output file name

    gI = imread_float(g_name)  # read gray image
    cI = imread_float(c_name)  # read scribbled image
    color_region = np.sum(np.abs(gI - cI), axis=2) > 0.01  # find scribbled region

    gI_yuv = cv2.cvtColor(gI, cv2.COLOR_RGB2YUV)
    cI_yuv = cv2.cvtColor(cI, cv2.COLOR_RGB2YUV)

    yuvIm = np.zeros_like(gI_yuv)
    yuvIm[:, :, 0] = gI_yuv[:, :, 0]
    yuvIm[:, :, 1] = cI_yuv[:, :, 1]
    yuvIm[:, :, 2] = cI_yuv[:, :, 2]

    # main algorithm of colorization
    result_img = colorize(color_region, yuvIm)
    cv2.imwrite(out_name, cv2.cvtColor(result_img, cv2.COLOR_YUV2BGR) * 255)


if __name__ == "__main__":
    main()
