import torch
import torchvision


vgg16 = torchvision.models.vgg16(pretrained=False)

# method1
# torch.save(vgg16, "vgg16_method1.pth")

# method2, recommended
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
