import torch
from torch import nn


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


model = Model()
x = torch.tensor(1.0)
print(x)
y = model(x)
print(y)

