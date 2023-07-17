import cv2
import numpy as np
import matplotlib.pyplot as plt


def hist_equal(img, L=255):
    h, w = img.shape
    # pixel number, add 1 to avoid divide by zero
    A0 = h * w + 1

    out = img.copy()
    sum_h = 0

    for i in range(1, 255):
        index = np.where(img == i)
        sum_h += len(img[index])
        fDA = L / A0 * sum_h
        out[index] = fDA

    return out


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    origin = cv2.imread("original.png", cv2.IMREAD_GRAYSCALE)
    out = hist_equal(origin)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(origin.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.xlabel("gray scale")
    plt.ylabel("pixel number")
    plt.title("origin")
    plt.subplot(1, 2, 2)
    plt.hist(out.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.xlabel("gray scale")
    plt.ylabel("pixel number")
    plt.title("output")
    plt.savefig("histograms_contrast.png")
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(origin, cmap='gray')
    ax1.set_title("origin")
    ax1.axis('off')
    ax2.imshow(out, cmap='gray')
    ax2.set_title("output")
    ax2.axis('off')
    plt.savefig("image_contrast.png")
    plt.show()

