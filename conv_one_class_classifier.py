import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class ConvOneClassClassifier(nn.Module):
    def __init__(self, num_kernels, kernel_side):
        super(ConvOneClassClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=num_kernels,
                               kernel_size=kernel_side,
                               stride=1,
                               padding=0,
                               bias=True)
        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = nn.Flatten(x)
        out = self.out_act(x)
        return out


def train_net(true_examples, num_kernels, kernel_side, epochs):
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    net = ConvOneClassClassifier(num_kernels, kernel_side).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    losses = []
    for i in range(epochs):
        y_pred = net.forward(X)
        loss = criterion(y_pred, y)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return net




if __name__ == "__main__":
    true_examples =
    contrast_examples =
    net = train_net(true_examples, contrast_examples, num_kernels=1, kernel_side=5, epochs=10000)
