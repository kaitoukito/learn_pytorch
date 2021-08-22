import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        return y


test_data = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

model = Model()
print(model)

writer = SummaryWriter("logs")
step = 0
for imgs, targets in test_loader:
    output = model(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))     # (xxx, 6, 30, 30) -> (xxx, 3, 30, 30)
    writer.add_images("output", output, step)
    step += 1

writer.close()
