import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_and_model import DNAOnly, MyDataSet
from preprocess import Preprocessor

# preprocess data
preprocess = Preprocessor()
preprocess.generateSamples()

# create dataset
train_set = MyDataSet(preprocess.train_sample_file_paths[0])
train_data = DataLoader(train_set, batch_size=128, shuffle=True)

# create model
DNAOnlyNet = DNAOnly()
loss_function = nn.BCELoss()
optimizer = optim.Adam(DNAOnlyNet.parameters(), lr=1e-4, weight_decay=1e-6)

# train model
for epoch in range(30):
    running_loss = 0
    for i, data in enumerate(train_data):
        train_X, train_y = data
        optimizer.zero_grad()
        outputs = DNAOnlyNet(train_X)
        loss = loss_function(outputs, train_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


# test model