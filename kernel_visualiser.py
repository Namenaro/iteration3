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


    def show_activations_as_mask_no_threshold(self, image):
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

    def show_activations_as_mask(self, image, threshold):
        cm = plt.cm.get_cmap('Blues')
        matrix_result = self.get_matrix_activations(image)

        plt.imshow(image, cmap='gray_r')

        mask_side = matrix_result.shape[0]
        x = []
        y = []
        c = []
        for i in range(mask_side):
            for j in range(mask_side):
                if matrix_result[i,j] <= threshold:
                    x.append(i + self.hside)
                    y.append(j + self.hside)
                    c.append(matrix_result[i,j])
        plt.scatter(x, y, c=c, cmap=cm)
        plt.colorbar()


if __name__ == "__main__":
    mnist_number = 0
    image = get_mnist_number(mnist_number)
    devcr = FirstDeviceCreator(2, image)
    matrix = devcr.create_device()

    n=4
    plt.subplot(1,n,1)
    plt.imshow(matrix, cmap='gray_r')
    plt.subplot(1,n,2)
    kerap = KernelApplicator(matrix)
    kerap.show_hist(image)
    plt.subplot(1, n, 3)
    kerap.show_activations_as_mask_no_threshold(image)
    plt.subplot(1, n, 4)
    thr = 600
    kerap.show_activations_as_mask(image,thr)
    plt.title("threshold=" + str(thr))
    plt.show()