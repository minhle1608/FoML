import torch
import torch.nn as nn

class NN(nn.Module):
    def __init__(self, input_shape):
        super(NN, self).__init__()
        self.input = nn.Linear(input_shape, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.hidden_1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.hidden_2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        x = self.input(x)
        x = self.bn1(x)
        x = self.ReLU(x)

        x = self.hidden_1(x)
        x = self.bn2(x)
        x = self.ReLU(x)

        x = self.hidden_2(x)
        x = self.bn3(x)
        x = self.ReLU(x)

        x = self.output(x)
        return x