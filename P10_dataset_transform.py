import torchvision

train_set = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", download=True)
test_set = torchvision.datasets.CIFAR10(root="dataset_CIFAR10", train=False, download=True)
