import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
                                 MaxPool2d(kernel_size=2),
                                 Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
                                 MaxPool2d(kernel_size=2),
                                 Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                 MaxPool2d(kernel_size=2),
                                 Flatten(),
                                 Linear(in_features=1024, out_features=64),
                                 Linear(in_features=64, out_features=10))

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    dataset = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=False)

    model = Model()
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.01)    # stochastic gradient descent

    for epoch in range(20):
        total_loss = 0.0
        for imgs, targets in dataloader:
            outputs = model(imgs)
            results_loss = loss(outputs, targets)
            optim.zero_grad()
            results_loss.backward()
            optim.step()
            total_loss += results_loss
        print(total_loss)
