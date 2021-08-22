import torch
import torchvision
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


if __name__ == '__main__':
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

    # loss function
    loss_fn = CrossEntropyLoss()

    # optimizer
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training
    total_train_step = 0
    total_test_step = 0
    epoch = 10

    # tensor board
    writer = SummaryWriter("logs")

    for i in range(epoch):
        print("-------- starting {}-th epoch training --------".format(i))

        # training step in each epoch
        model.train()
        for imgs, targets in train_dataset_loader:
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

        # test step in each epoch
        model.eval()
        total_test_loss = 0
        total_correct_num = 0
        with torch.no_grad():
            for imgs, targets in test_dataset_loader:
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
