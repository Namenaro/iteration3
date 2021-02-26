# загрузить мнист
import numpy as np
def get_train_mnist():
    import torchvision.datasets as datasets
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    return mnist_trainset

def draw_several(np_images, np_labels):
    import matplotlib.pyplot as plt
    plt.figure()
    rows = 5
    cols = 7
    num_of_images = rows*cols
    for index in range(1, num_of_images + 1):
        plt.subplot(rows, cols, index)
        plt.axis('off')
        if np_labels is not None:
            plt.title(np_labels[index])
        plt.imshow(np_images[index].squeeze(), cmap='gray_r')
    plt.show()

def get_exact_numbers(np_images, np_labels):
    results = []
    number= 0
    for i in range(len(np_labels)):
        if np_labels[i] == number:
            results.append(np_images[i])
    return np.array(results)



if __name__ == "__main__":
    mnist_train = get_train_mnist()
    print (mnist_train)
    imgs = mnist_train.train_data.numpy()
    labels = mnist_train.train_labels.numpy()

    import matplotlib.pyplot as plt

    plt.imshow(imgs[1].squeeze(), cmap='gray_r')
    plt.savefig("example.png")
    print(labels[1])
    imgs = get_exact_numbers(imgs, labels)
    draw_several(imgs, None)

