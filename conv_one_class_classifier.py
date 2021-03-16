
import torch.nn as nn


class ConvOneClassClassifier(nn.Module):
    def __init__(self, num_kernels, kernel_side):
        super(ConvOneClassClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=num_kernels,
                               kernel_size=kernel_side,
                               stride=1,
                               padding=0,
                               bias=False)
        self.pool2d = nn.AdaptiveMaxPool2d((1, 1))
        self.pool1d = nn.AdaptiveMaxPool1d(1)
        self.lin = nn.Linear(num_kernels, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool2d(x)
        x = x.view(x.shape[0], 1, -1)
        x = self.lin(x)

        x = self.out_act(x)

        return x



