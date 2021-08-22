from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# how to use transforms?
tensor_trans = transforms.ToTensor()    # create tool
tensor_img = tensor_trans(img)          # use tool

writer.add_image("Tensor_img", tensor_img)

writer.close()

print(tensor_img)
