from get_two_starts_dataset import get_2_starts_np_dataset_from_bank
from contrast_getter import ContrastGetter

import numpy as np


def get_XY():
    true_examples = get_2_starts_np_dataset_from_bank()
    contrast_getter = ContrastGetter(data_shape=(true_examples.shape[1], true_examples.shape[2]), num_to_exlude=2)
    contrast_len = len(true_examples)*5
    contrast = contrast_getter.get_n_random_contrast_examples(contrast_len)
    np_x = np.concatenate((true_examples, contrast), axis=0)\

    TRUE_LABEL = 1.0
    FALSE_LABEL = 0.0

    true_labels = np.full((len(true_examples),), TRUE_LABEL)
    false_labels = np.full((len(contrast),), FALSE_LABEL)

    np_y = np.concatenate((true_labels, false_labels))
    return np_x, np_y

if __name__ == "__main__":
    X, Y = get_XY()