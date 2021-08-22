from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("data/train/ants_image/0013035.jpg")

# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([6, 3, 2], [3, 2, 1])
img_norm = trans_norm(img_tensor)
writer.add_image("Normalize", img_norm)

# Resize
trans_resize = transforms.Resize((300, 400))
img_resize = trans_totensor(trans_resize(img))
writer.add_image("Resize", img_resize)

# Compose
trans_compose = transforms.Compose([trans_resize, trans_totensor])
img_compose = trans_compose(img)
writer.add_image("Compose", img_resize)

# RandomCrop
trans_randomcrop = transforms.RandomCrop((300, 400))
trans_compose2 = transforms.Compose([trans_randomcrop, trans_totensor])
img_compose2 = trans_compose2(img)
writer.add_image("Compose2", img_compose2)

writer.close()
