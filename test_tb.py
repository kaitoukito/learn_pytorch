from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants_image/0013035.jpg"
# image_path = "data/train/bees_image/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(type(img_array))
print(img_array.shape)

writer.add_image("train", img_array, global_step=1, dataformats='HWC')
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)

writer.close()
