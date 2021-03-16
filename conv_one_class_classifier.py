
import torch.nn as nn


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

        x = self.out_act(x)

        return x



