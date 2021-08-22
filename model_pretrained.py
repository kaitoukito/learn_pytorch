import torchvision

# train_dataset = torchvision.datasets.ImageNet(root="dataset_ImageNet", split='train', download=True,
#                                               transform=torchvision.transforms.ToTensor())

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)      # need download
print('ok')
