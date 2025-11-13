import os
os.system("cls")
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from models.resnet import ASH_ResNet, DA_Resnet
import numpy as np

# p_zeros = 0.9  # Percentuale di zeri desiderata (modificare secondo le tue esigenze)
# p_ones = 1 - p_zeros

A = torch.rand(3, 10, 10)

# # Crea un tensore di maschera con zeri e uno in base alle percentuali specificate
# M = torch.tensor(np.random.choice([0, 1], A.shape, p=[p_zeros, p_ones]))


# print(M)

# num_zeros = torch.sum(M == 0).item()
# num_ones = torch.sum(M == 1).item()

# print(num_zeros/300)

# count_conv=7
# print((count_conv-1) % 3==0)
model1 = nn.Sequential(
    nn.Linear(32*32*3, 64),
    # save/apply the activation maps
    nn.ReLU(),
    nn.Linear(64, 64),
    # save/apply the activation maps
    nn.ReLU(),
    nn.Linear(64, 64),
    # save/apply the activation maps
    nn.ReLU(),

    nn.Linear(64, 7) # Classifier
)
count_conv=0

model = DA_Resnet()
#print(model)

# for i, layer in enumerate(model.modules()):
#     pass

# print(i)

# num_layers = len(list(model.modules()))
# print(num_layers)
               


# model = ModifiedResNet()
# print(model.layer2[0].downsample[0])
print(model.resnet.conv1)

# ind_conv = list(range(1,21))
# nc=0
# for name, module in model.resnet.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 nc = nc+1
#                 print(nc)
#                 print(name)
#                 if nc in ind_conv and nc not in [8,13,18]: #or (count_conv-1) % 3==0:
#                     print (nc)


# # print(nc) 18 13 8
#                     ind_conv = list(range(1,21))
# nc=0
# for name, module in model.resnet.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 nc = nc+1
#                 print(nc)
#                 print(name)

# con questo script abbiamo capito che i downsample laayer convoluzionali sono 18 13 8