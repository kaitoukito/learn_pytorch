import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn.modules.flatten import Flatten


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


img_path = "images/dog.png"
img = Image.open(img_path)
print(img)
img = img.convert("RGB")
print(img)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
img = transform(img)
print(img.shape)
img = torch.reshape(img, (1, 3, 32, 32))
print(img.shape)

model = Model()
model.load_state_dict(torch.load("model of 29-th training"))
print(model)

model.eval()
with torch.no_grad():
    output = model(img)
print(output)
print(output.argmax(1).item())
