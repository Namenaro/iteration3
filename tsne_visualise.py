from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tnse_visialise(data):
    data = data.reshape(data.shape[0], -1 )
    print (data.shape)
    data_embedded = TSNE(n_components=2).fit_transform(data)
    print (data_embedded.shape)
    plt.scatter(data_embedded[:,0],data_embedded[:,1], c='red')
    plt.show()

