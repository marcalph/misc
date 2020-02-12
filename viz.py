import matplotlib.pyplot as plt
from matplotlib import cm


def dual_plot(img1, img2):
    """
    simple side by side plot
    useful to inspect image transform
    """
    fig = plt.figure()
    fig.add_subplot(1,2, 1)
    plt.imshow(img1)
    fig.add_subplot(1,2, 2)
    plt.imshow(img2)
    plt.show()
    return None



def batch_plot(img_batch, label_batch=None, side=5):
    """
    viz function to inspect image data
    displays `side`**2 images of a batch by default
    """
    fig = plt.figure(figsize=(2*side, 2*side))
    for i in range(side**2):
        plt.subplot(side, side, i+1)
        plt.yticks([])
        plt.xticks([])
        plt.grid(False)
        plt.imshow(img_batch[i], cmap=plt.cm.binary)
        if label_batch is not None:
            plt.xlabel(label_batch[i])
    plt.show()
    return None





if __name__ == '__main__':
    import numpy as np
    img = np.random.random((50, 50))
    dual_plot(img, img.T)
    batch_plot([img]*9, 3, ["white_noise"]*9)