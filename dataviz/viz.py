import matplotlib.pyplot as plt
from matplotlib import cm


def dual_plot(img1, img2):
    fig = plt.figure()
    fig.add_subplot(1,2, 1)
    plt.imshow(img1)
    fig.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.show()
    return None


if __name__ == '__main__':
    import numpy as np
    img = np.random.random((50, 50))
    dual_plot(img, img)