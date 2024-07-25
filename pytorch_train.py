import torch
import torchmetrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_model import MyDataSet, DNAModule
from preprocess import Preprocessor



# preprocess data
preprocess = Preprocessor()
X_train, X_test, y_train, y_test = preprocess.generateSamples()

# create dataset
train_set = MyDataSet(X_train, y_train)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)

# create model
DNAOnlyNet = DNAModule()
loss_function = nn.BCELoss()
optimizer = optim.Adam(DNAOnlyNet.parameters(), lr=1e-4, weight_decay=1e-6)

# train model
for epoch in range(1):
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
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

# test model
test_set = MyDataSet(X_test, y_test)
test_data = DataLoader(train_set, batch_size=32, shuffle=False)

auroc_score = torchmetrics.AUROC(task='binary')
aupr_score = torchmetrics.AveragePrecision(task='binary')

for i, data in enumerate(test_data):
    test_X, test_y = data
    torch.no_grad()
    outputs = DNAOnlyNet(test_X)
    test_y = test_y.to(torch.int)
    roc_score = auroc_score(outputs, test_y)
    pr_score = aupr_score(outputs, test_y)

total_auroc_score = auroc_score.compute()
total_aupr_score = aupr_score.compute()
print('auROC = {}'.format(total_auroc_score))
print('auPR  = {}'.format(total_aupr_score))