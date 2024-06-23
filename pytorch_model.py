import torch
import torch.nn as nn
from torch.utils.data import Dataset

## torch tensor (batch_size, channel, row, col)
## tensorflow (batch_size, row, col, channel)
## torch padding truple (row, col)

class MyDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


class DNAModule(nn.Module):
    def __init__(self):
        super(DNAModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(4, 8)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.fc1 = None
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        p1 = self.pool1(c4)
        c5 = self.conv5(p1)
        c6 = self.conv6(c5)
        c7 = self.conv7(c6)
        p2 = self.pool2(c7)

        concat_c2_c3 = torch.concat((c2, c3), dim=1)
        concat_c5_c6 = torch.concat((c5, c6), dim=1)

        y = torch.concat((c1, concat_c2_c3, concat_c5_c6, p2), dim=-1)
        y = torch.flatten(y, start_dim=1)

        if self.fc1 is None:
            in_features = y.size(1)
            self.fc1 = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU()
            )
        
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.drop(y)
        y = self.fc3(y)
        y = self.fc4(y)

        return y
