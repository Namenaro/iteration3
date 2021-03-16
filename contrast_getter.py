from utils import get_mnist_exclude_number

import numpy as np


class ContrastGetter:
    def __init__(self, data_shape, num_to_exlude=2):
        self.mnist = get_mnist_exclude_number(num_to_exlude)
        self.data_shape = data_shape

    def get_n_random_contrast_examples(self, n):
        results = []
        side = self.data_shape[1]
        random_indices = np.random.choice(len(self.mnist), n)
        for i in random_indices:
            image = self.mnist[i]
            imside = image.shape[0]
            mincoord = 0
            maxcoord = imside - side - 1
            xy = np.random.choice(range(mincoord, maxcoord), 2)
            x_min = xy[0]
            y_min = xy[1]
            x_max = x_min + side
            y_max = y_min + side
            patch = image[y_min:y_max, x_min:x_max]
            results.append(patch)
        return np.array(results)
