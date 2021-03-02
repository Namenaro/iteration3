# загрузить мнист
import numpy as np
import matplotlib.pyplot as plt
fig = None
def get_train_mnist():
    import torchvision.datasets as datasets
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    return mnist_trainset

def draw_several(np_images, np_labels):
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

def get_exact_numbers(the_number, np_images, np_labels):
    results = []

    for i in range(len(np_labels)):
        if np_labels[i] == the_number:
            results.append(np_images[i])
    return np.array(results)

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
    plt.scatter(event.xdata,  event.ydata, s=500, c='red', marker='o')
    fig.canvas.draw()
    fig.canvas.flush_events()

if __name__ == "__main__":
    mnist_train = get_train_mnist()
    print (mnist_train)
    imgs = mnist_train.train_data.numpy()
    labels = mnist_train.train_labels.numpy()
    fig = plt.figure()
    image = imgs[1].squeeze()
    plt.imshow(image, cmap='gray_r')

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    #plt.savefig("example.png")
    #print(labels[1])
    #imgs = get_exact_numbers(0, imgs, labels)
    #draw_several(imgs, None)

