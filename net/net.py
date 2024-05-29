import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## torch tensor 4维的意义 以图片为例
## 第一维代表 batch-size    第二维代表 channel     第三维代表 row     第四维代表 col
## tensorflow 4维的意义 以图片为例
## 第一维代表 batch-size    第二维代表 row         第三维代表 col     第四维代表 channel

## torch padding truple 的含义
## 第一维代表 增加row   第二维代表 增加col

class DNAOnly(nn.Module):
    def __init__(self):
        super(DNAOnly, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(4,8))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(1,3), padding=(0,1))

        self.conv7 = nn.Conv2d(64, 128, kernel_size=1)

        self.fc1 = None
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.maxpool = nn.MaxPool2d(kernel_size=(1,2))
        self.relu = nn.ReLU()
        self.drop = nn.Dropout()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.relu(self.conv1(x))
        y2 = self.relu(self.conv2(y1))
        y3 = self.relu(self.conv3(y2))
        y4 = self.relu(self.conv4(y3))
        y4p = self.maxpool(y4)
        y5 = self.relu(self.conv5(y4p))
        y6 = self.relu(self.conv6(y5))
        y7 = self.relu(self.conv7(y6))
        y7p = self.maxpool(y7)

        y2_y3 = torch.concat((y2, y3), dim=1)
        y5_y6 = torch.concat((y5, y6), dim=1)
        y = torch.concat((y1, y2_y3, y5_y6, y7p), dim=-1)

        y = torch.flatten(y, start_dim=1)

        if self.fc1 is None:
            in_feat = y.size(1)
            self.fc1 = nn.Linear(in_feat, 512)
        
        y = self.relu(self.fc1(y))
        y = self.relu(self.fc2(y))
        y = self.drop(y)
        y = self.relu(self.fc3(y))
        y = self.sigmoid(self.fc4(y))

        return y
    

DNAOnlyNet = DNAOnly()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(lr=1e-4, weight_decay=1e-6)

for epoch in range(200):
    # need input data
    optimizer.zero_grad()
    outputs = DNAOnlyNet()
    loss = loss_function()
    loss.backward()
    optimizer.step()