from utils import get_mnist_number
from first_device import FirstDeviceCreator
import numpy as np
import matplotlib.pyplot as plt

class KernelApplicator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.hside = int(kernel.shape[0]/2)


    def show_hist(self, image):
        matrix_result = self.get_matrix_activations(image)
        plt.hist(matrix_result.flatten())
        plt.ylabel('how many times')
        plt.xlabel('lavel of activation (lower better)')
        plt.title("histogram")


    def get_matrix_activations(self, image):
        ilen = image.shape[0]
        matrix_result = np.empty([ilen - (2 * self.hside + 1), ilen - (2 * self.hside + 1)])
        for x_center in range(self.hside, ilen - self.hside-1):
            for y_center in range(self.hside, ilen - self.hside-1):
                x_min = x_center - self.hside
                x_max = x_center + self.hside + 1
                y_min = y_center - self.hside
                y_max = y_center + self.hside + 1
                patch = image[x_min:x_max, y_min:y_max]
                dist = np.linalg.norm(patch - self.kernel)
                matrix_result[x_min, y_min] = dist
        return matrix_result


    def show_activations_as_mask(self, image):
        cm = plt.cm.get_cmap('Blues')
        matrix_result = self.get_matrix_activations(image)

        plt.imshow(image, cmap='gray_r')

        mask_side = matrix_result.shape[0]
        x = []
        y = []
        c = []
        for i in range(mask_side):
            for j in range(mask_side):
                x.append(i + self.hside)
                y.append(j + self.hside)
                c.append(matrix_result[i,j])
        t= plt.scatter(x, y, c=c, cmap=cm)
        plt.colorbar()


if __name__ == "__main__":
    mnist_number = 0
    image = get_mnist_number(mnist_number)
    devcr = FirstDeviceCreator(3, image)
    matrix = devcr.create_device()
    plt.subplot(1,3,1)
    plt.imshow(matrix, cmap='gray_r')
    plt.subplot(1,3,2)
    kerap = KernelApplicator(matrix)
    kerap.show_hist(image)
    plt.subplot(1, 3, 3)
    kerap.show_activations_as_mask(image)
    plt.show()