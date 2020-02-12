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



def square_plot(img_seq, square, train_labels=None):
    fig = plt.figure(figsize=(2*square, 2*square))
    for i, img in enumerate(img_seq):
        plt.subplot(square, square, i+1)
        plt.yticks([])
        plt.xticks([])
        plt.grid(False)
        plt.imshow(img, cmap=plt.cm.binary)
        if train_labels:
            plt.xlabel(train_labels[i])
    plt.show()
    return None





if __name__ == '__main__':
    import numpy as np
    img = np.random.random((50, 50))
    dual_plot(img, img.T)
    square_plot([img]*9, 3, ["white_noise"]*9)