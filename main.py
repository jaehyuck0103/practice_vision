from pathlib import Path

import cv2
import exif
import matplotlib.pyplot as plt
import numpy as np

# from bfilter import bfilter


def gsolve(Z, B, l, w):
    # Z(i,j)  Pixel values of pixel location number i in image j
    # B(j)    log shutter speed for image j
    # l       lamda, constant that determines the amount of smoothness
    # w(z)    weighting function value for pixel value z

    num_pixels, num_imgs = Z.shape

    n = 256
    A = np.zeros([num_pixels * num_imgs + n - 1, n + num_pixels])
    b = np.zeros(A.shape[0])

    k = 0  # Include the data-fitting equations
    for i in range(num_pixels):
        for j in range(num_imgs):
            wij = w[Z[i, j]]
            A[k, Z[i, j]] = wij
            A[k, n + i] = -wij
            b[k] = wij * B[j]
            k += 1

    A[k, 128] = 1  # Fix the curve by setting its middle value to 0
    k += 1

    for i in range(n - 2):  # Include the smoothness equations
        A[k, i] = l * w[i + 1]
        A[k, i + 1] = -2 * l * w[i + 1]
        A[k, i + 2] = l * w[i + 1]
        k += 1

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return x[:n]


def main():
    file_list = sorted(Path("./InputImages/2").glob("*.jpg"))
    print(file_list)

    # Read image train
    imgs = [cv2.cvtColor(cv2.imread(str(x)), cv2.COLOR_BGR2RGB) for x in file_list]
    imgH, imgW, _ = imgs[0].shape

    # Get Shutter Info
    exposures = []
    for file in file_list:
        with open(file, "rb") as fp:
            exposures.append(exif.Image(fp).exposure_time)

    log_exposures = np.log(exposures)
    w = np.concatenate([np.arange(0, 128), np.arange(127, -1, -1)])

    # Select the sample pixels
    num_sample_pixels = 300
    yy = list(range(num_sample_pixels))  # np.random.randint(0, imgH, size=num_sample_pixels)
    xx = list(range(num_sample_pixels))  # np.random.randint(0, imgW, size=num_sample_pixels)
    samples = np.stack(
        [each_img[yy, xx, :] for each_img in imgs]
    )  # (num_imgs, num_sample_pixels, ch)

    # Get Response function
    g = np.zeros((3, 256))
    g[0] = gsolve(samples[:, :, 0].transpose(), log_exposures, 100, w)
    g[1] = gsolve(samples[:, :, 1].transpose(), log_exposures, 100, w)
    g[2] = gsolve(samples[:, :, 2].transpose(), log_exposures, 100, w)
    # fig, ax = plt.subplots()
    # plt.show()

    # Make Radiance Map
    rMap = np.zeros([imgH, imgW, 3])

    w -= 3
    w[w < 0] = 0  # to ignore side values ( val <= 3 || val >= 252 )

    for ch in range(3):
        wSum = np.sum([w[img[..., ch]] for img in imgs], axis=0)
        eSum = np.sum(
            [
                w[img[..., ch]] * (g[ch][img[..., ch]] - le)
                for img, le in zip(imgs, log_exposures, strict=True)
            ],
            axis=0,
        )

        # Handle saturation
        under_satur_mask = (wSum == 0) & (imgs[0][..., ch] < 5)
        eSum[under_satur_mask] = g[ch, 3] - log_exposures.max()
        wSum[under_satur_mask] = 1

        over_satur_mask = (wSum == 0) & (imgs[0][..., ch] > 250)
        eSum[over_satur_mask] = g[ch, 252] - log_exposures.min()
        wSum[over_satur_mask] = 1

        assert (wSum > 0).all()

        # Get radiance
        rMap[..., ch] = eSum / wSum

    rMap = np.exp(rMap)
    rMap = rMap.astype(np.float32)

    # Visualize Radiance Map
    rMapV = cv2.cvtColor(rMap, cv2.COLOR_RGB2GRAY)
    rMapV = np.log(rMapV)
    rMapV = (rMapV / rMapV.max()) * 255
    cv2.imwrite("RadianceMap.png", rMapV)

    # Tone Mapping
    rMapY = cv2.cvtColor(rMap, cv2.COLOR_RGB2GRAY)
    rMapYLog = np.log10(rMapY)
    rMapYLow = cv2.bilateralFilter(rMapYLog, 9, 0.6, 2.0)
    # rMapYLow = bfilter(rMapYLog, 4, 2, 0.6)
    rMapYHigh = rMapYLog - rMapYLow
    largeRange = rMapYLow.max() - rMapYLow.min()
    k = np.log10(10) / largeRange
    rMapYLow = k * rMapYLow
    rMapYScaled = 10 ** (rMapYLow + rMapYHigh)

    scale = rMapYScaled / rMapY

    resultImg = rMap * scale[:, :, np.newaxis]

    rMapYLowMax = 10 ** rMapYLow.max()
    resultImg = resultImg / rMapYLowMax  # normalization
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_RGB2BGR)
    cv2.imwrite("Result.png", resultImg * 255)


if __name__ == "__main__":
    main()
