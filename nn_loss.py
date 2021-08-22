import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = L1Loss(reduction="sum")
result = loss(inputs, targets)
print(result)

loss2 = MSELoss(reduction="sum")
result2 = loss2(inputs, targets)
print(result2)

# cross entropy loss
x = torch.tensor([0.1, 0.8, 0.1])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss3 = CrossEntropyLoss()
result3 = loss3(x, y)
print(result3)

