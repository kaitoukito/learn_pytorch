import torch
import torchvision


# method1
# vgg16 = torch.load("vgg16_method1.pth")
# print(vgg16)

# method2, recommended
# vgg16_dict = torch.load("vgg16_method2.pth")
# print(vgg16_dict)
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
