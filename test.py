import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import MyNet
import MyDataset


data_tf = transforms.Compose([transforms.Resize(256),
                            transforms.RandomCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
train_dataset = MyDataset.MyDataset(transform=data_tf)
train_Loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
model = MyNet.ResNet(19)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


if torch.cuda.is_available():
    model = model.cuda()


losses = []

for im, label in train_Loader:
    im = Variable(im)
    print(label)
    break

