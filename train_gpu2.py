import time
import torch
import torchvision
from torch import nn
from torch.nn.modules.flatten import Flatten
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                                    nn.MaxPool2d(kernel_size=2),
                                    Flatten(),
                                    nn.Linear(in_features=1024, out_features=64),
                                    nn.Linear(in_features=64, out_features=10))

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    # device = torch.device("cpu")
    device = torch.device("cuda:0")

    # dataset
    train_dataset = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=True,
                                                 transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False,
                                                transform=torchvision.transforms.ToTensor(), download=True)

    # dataset size
    train_dataset_size = len(train_dataset)
    test_dataset_size = len(test_dataset)
    print("train_dataset_size = {}".format(train_dataset_size))
    print("test_dataset_size = {}".format(test_dataset_size))

    # dataset loader
    train_dataset_loader = DataLoader(train_dataset, batch_size=64)
    test_dataset_loader = DataLoader(test_dataset, batch_size=64)

    # nn structure
    model = Model()
    model = model.to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)

    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training
    total_train_step = 0
    total_test_step = 0
    epoch = 30

    # tensor board
    writer = SummaryWriter("logs")

    for i in range(epoch):
        print("-------- starting {}-th epoch training --------".format(i))

        start_time = time.time()

        # training step in each epoch
        model.train()
        for imgs, targets in train_dataset_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            # optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_train_step % 100 == 0:
                print("total_train_step = {}, loss = {}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), global_step=total_train_step)
            total_train_step += 1

        end_time = time.time()
        total_time = end_time - start_time
        print("total_time = {}".format(total_time))

        # test step in each epoch
        model.eval()
        total_test_loss = 0
        total_correct_num = 0
        with torch.no_grad():
            for imgs, targets in test_dataset_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()

                correct_num = (outputs.argmax(dim=1) == targets).sum().item()
                total_correct_num += correct_num

        total_accuracy = total_correct_num / test_dataset_size
        # print(total_correct_num, type(total_correct_num))
        # print(test_dataset_size, type(test_dataset_size))
        # print(total_accuracy, type(total_accuracy))
        print("total_test_loss = {}".format(total_test_loss))
        print("total_correct_num = {}".format(total_correct_num))
        print("total_accuracy = {}".format(total_accuracy))
        writer.add_scalar("test_loss", total_test_loss, global_step=total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy, global_step=total_test_step)
        total_test_step += 1

        torch.save(model.state_dict(), "model of {}-th training".format(i))
        print("model is saved")

    writer.close()
