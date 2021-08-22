import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = Linear(64*3*32*32, 10)

    def forward(self, x):
        y = self.linear1(x)
        return y


test_data = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)     # set drop_last=True

model = Model()

writer = SummaryWriter("logs")
step = 0
for imgs, targets in test_loader:
    imgs = torch.flatten(imgs)  # flatten
    output = model(imgs)
    print(imgs.shape)
    print(output.shape)
    step += 1

writer.close()
