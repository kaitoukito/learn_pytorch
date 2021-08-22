import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        y = self.maxpool1(x)
        return y


# input = torch.tensor([[1, 2, 0, 3, 1],
#                       [0, 1, 2, 3, 1],
#                       [1, 2, 1, 0, 0],
#                       [5, 2, 3, 1, 1],
#                       [2, 1, 0, 1, 1]], dtype=torch.float32)

# input = torch.reshape(input, (-1, 1, 5, 5))

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
