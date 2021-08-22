import torch
import torchvision
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = ReLU(inplace=False)
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        y = self.sigmoid1(x)
        return y


# input = torch.tensor([[1, -0.5],
#                       [-1, 3]])

# input = torch.reshape(input, (-1, 1, 2, 2))

test_data = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

model = Model()

writer = SummaryWriter("logs")
step = 0
for imgs, targets in test_loader:
    output = model(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)
    step += 1

writer.close()
