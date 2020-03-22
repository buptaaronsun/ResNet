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
train_Loader = DataLoader(train_dataset, batch_size=512, shuffle=True, drop_last=True)
model = MyNet.ResNet(19)

criterion = MyDataset.MyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


if torch.cuda.is_available():
    model = model.cuda()


losses = []

n = 1
count = 0
for i in range(n):
    train_loss = 0
    train_acc = 0
    model.train()
    for im, label in train_Loader:
        im = Variable(im)
        label = torch.Tensor(label)
        label = Variable(label)
        if torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()
        out = model(im)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print(count)
        count += 1
        losses.append(train_loss/count)

    
    print('epoch:',i)

plt.plot(np.arange(len(losses)), losses)
plt.savefig('loss.png')
plt.show()

state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
torch.save(state, './result')
