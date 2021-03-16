from conv_one_class_classifier import ConvOneClassClassifier
from contrast_getter import ContrastGetter

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

def prepare_batch_xy_np(true_examples, contrast_getter):
    contrast_len = len(true_examples)
    contrast = contrast_getter.get_n_random_contrast_examples(contrast_len)
    np_x = np.concatenate(true_examples, contrast)

    TRUE_LABEL = 1.0
    FALSE_LABEL = 0.0

    true_labels = np.full((len(true_examples), ), TRUE_LABEL)
    false_labels = np.full((len(contrast),), FALSE_LABEL)

    np_y = np.concatenate(true_labels, false_labels)
    return np_x, np_y


def get_trained_classifier(true_examples, num_kernels, kernel_side, epochs):
    contrast_getter = ContrastGetter(data_shape=(true_examples.shape[1], true_examples.shape[2]), num_to_exlude=2)
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    net = ConvOneClassClassifier(num_kernels, kernel_side).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    losses = []
    for i in range(epochs):
        X_np, Y_np = prepare_batch_xy_np(true_examples, contrast_getter)

        X = Variable(FloatTensor(X_np))
        Y = Variable(FloatTensor(Y_np))

        Y_pred = net.forward(X)
        loss = criterion(Y_pred, Y)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net

if __name__ == "__main__":
    from get_two_hecks_bank import get_2_necks_bank
    true_examples = get_2_necks_bank()
    net = get_trained_classifier(true_examples, num_kernels=1, kernel_side=5, epochs=10)