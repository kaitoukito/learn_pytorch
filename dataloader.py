import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("logs")
step = 0
for epoch in range(2):
    for imgs, targets in test_loader:
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)  # here is add_images, not add_image
        step += 1

writer.close()
