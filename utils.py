import numpy as np

def get_train_mnist():
    import torchvision.datasets as datasets
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    return mnist_trainset

def get_exact_numbers(the_number):
    results = []
    mnist_train = get_train_mnist()
    print(mnist_train)
    np_images = mnist_train.train_data.numpy()
    np_labels = mnist_train.train_labels.numpy()
    for i in range(len(np_labels)):
        if np_labels[i] == the_number:
            results.append(np_images[i])
    return np.array(results)

def get_mnist_number(the_number):
    return get_exact_numbers(the_number)[0].squeeze()



