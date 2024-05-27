import torch
import torch.nn as nn
import torch.nn.functional as F

## torch tensor 4维的意义 以图片为例
## 第一维代表 batch-size    第二维代表 channel     第三维代表 row     第四维代表 col

## torch padding truple 的含义
## 第一维代表 增加row   第二维代表 增加col

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(4,8))
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.relu(y)

        return y
    

net = Net()

print(net)