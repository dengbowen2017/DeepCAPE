import torch.nn as nn
import torch.optim as optim

from dataset_and_model import DNAOnly

# preprocess data


# create dataset


# create model
DNAOnlyNet = DNAOnly()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(lr=1e-4, weight_decay=1e-6)


# train model
for epoch in range(200):
    # need input data
    optimizer.zero_grad()
    outputs = DNAOnlyNet()
    loss = loss_function()
    loss.backward()
    optimizer.step()

