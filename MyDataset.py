from torch.utils.data import Dataset
from PIL import Image
import torch
from torch import nn

# *************************************数据集的设置****************************************************************************
img_root1 = '/home/ug_wh/Documents/ExpW/data/image/'  # 数据集的地址
label_root1 = '/home/ug_wh/Documents/ExpW/data/label/training_label.lst'
img_root2 = '/home/ug_wh/Documents/CelebA/Img/img_celeba/'
label_root2_0 = '/home/ug_wh/Documents/CelebA/label/training_bbox_celeba.txt'
label_root2_1 = '/home/ug_wh/Documents/CelebA/label/training_attr_celeba.txt'

# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, transform=None):
        super(MyDataset, self).__init__()
        # 对继承自父类的属性进行初始化
        f1 = open(label_root1, 'r')
        f2 = open(label_root2_0, 'r')
        f3 = open(label_root2_1, 'r')
        self.imgs = []
        self.positions = []
        self.dataset1_num = 0
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in f1:  # 迭代该列表#按行循环txt文本中的内
            self.dataset1_num = self.dataset1_num+1
            line = line.strip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()
            label = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            label[12+int(words[7])] = 1
            # 用split将该行分割成列表  split的默认参数是空格，所以不传递任何参数时分割空格
            self.imgs.append([img_root1+words[0], label])
            self.positions.append([int(words[3]), int(words[2]), int(words[4]), int(words[5])])

        for line in f2:
            line = line.strip()
            words = line.split()
            position = [int(words[1]), int(words[2]), int(words[1])+int(words[3]), int(words[2])+int(words[4])]
            self.positions.append(position)

        for line in f3:
            line = line.strip()
            words = line.split()
            label = []
            for i in range(12):
                temp = int(words[1+i])
                if temp == 1:
                    label.append(1)
                else:
                    label.append(0)


            for i in range(7):
                label.append(0)
            self.imgs.append([img_root2+words[0], label])

            # 把txt里的内容读入imgs列表保存
        self.transform = transform
        f1.close()
        f2.close()
        f3.close()
        # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************


    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        img, label = self.imgs[index]
        # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = default_loader(img)
        #img = img.crop(self.positions[index])
        # 按照路径读取图片
        if self.transform is not None:
            img = self.transform(img)
        label = torch.Tensor(label)
            # 数据标签转换为Tensor
        return img, label
        # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容
        # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, label):  # 定义前向的函数运算即可
        loss = 0
        for (x, y) in zip(output, label):
            x = torch.log(x)
            x = x.view(1,-1)
            y = y.view(-1,1)
            loss -= torch.mm(x, y)

        return loss
