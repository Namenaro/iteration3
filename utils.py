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

def get_all_images_np():
    mnist_train = get_train_mnist()
    np_images = mnist_train.train_data.numpy()
    return np.squeeze(np_images)

def get_contrast(data_shape):
    results = []
    num_of_samples = data_shape[0]*5
    side= data_shape[1]
    hside = int(side/2)
    mnist_train = get_train_mnist()
    np_images = mnist_train.train_data.numpy()
    random_indices = np.random.choice(len(np_images), num_of_samples)
    for i in random_indices:
        image = np_images[i]
        imside = image.shape[0]
        mincoord = hside
        maxcoord = imside - hside - 1
        xy = np.random.choice(range(mincoord, maxcoord), 2)
        x_center = xy[0]
        y_center = xy[1]
        x_min = x_center - hside
        x_max = x_center + hside + 1
        y_min = y_center - hside
        y_max = y_center + hside + 1
        patch = image[y_min:y_max, x_min:x_max]
        results.append(patch)
    return np.array(results)





