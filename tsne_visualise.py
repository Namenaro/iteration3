from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from utils import get_contrast

def tnse_visialise(data):
    data = data.reshape(data.shape[0], -1 )
    data_embedded = TSNE(n_components=2).fit_transform(data)
    print ("data_embedded.shape=" + str(data_embedded.shape))
    plt.scatter(data_embedded[:,0],data_embedded[:,1], c='red')
    plt.show()


def tnse_visialise_with_colors(data, colors):
    data = data.reshape(data.shape[0], -1)
    data_embedded = TSNE(n_components=2).fit_transform(data)
    print("data_embedded.shape=" + str(data_embedded.shape))
    cm = plt.cm.get_cmap('summer')
    plt.scatter(data_embedded[:, 0], data_embedded[:, 1], c=colors, cmap=cm)
    plt.colorbar()
    plt.show()


def tnse_visialise_with_contrast(data):
    contrast = get_contrast(data.shape)
    data = data.reshape(data.shape[0], -1)
    contrast = contrast.reshape(contrast.shape[0], -1)
    cdata = np.concatenate((data, contrast), axis=0)

    data_embedded = TSNE(n_components=2).fit_transform(cdata)
    print("data_embedded.shape=" + str(data_embedded.shape))
    cm = plt.cm.get_cmap('summer')
    plt.scatter(data_embedded[:len(data), 0], data_embedded[:len(data), 1], c='red')
    plt.scatter(data_embedded[len(data):, 0], data_embedded[len(data):, 1], c='blue')
    plt.colorbar()
    plt.show()