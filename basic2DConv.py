# This file imports the data and trains a 2D convolutional neural network
# This method has depreciated and is not in current use for final model

import sys
# Change path so that load_dataset function can be used
sys.path.append('./elpv-dataset-1.0/utils')
import elpv_reader
from elpv_reader import load_dataset
import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
import collections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset, test train split and convert to tensors
images, proba, types = load_dataset()
proba = proba * 3
# images, proba = images[:100], proba[:100]
X_train, X_test, y_train, y_test = train_test_split(images, proba, test_size=0.2, random_state=42, stratify=proba)
print(collections.Counter(y_test.tolist()))
X_train_tensor = torch.tensor(X_train).unsqueeze(1).float()
X_test_tensor = torch.tensor(X_test).unsqueeze(1).float()
y_train_tensor = torch.tensor(y_train).long()
y_test_tensor = torch.tensor(y_test).long()
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_len = len(list(y_train_tensor))
test_len = len(list(y_test_tensor))

# Neural network object
class ConvNetwork(nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 13)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 10, 13)
        self.linear1 = nn.Linear(10 * 66 * 66, 4000)
        self.linear2 = nn.Linear(4000, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


# prepare neural network and data for training
net = ConvNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.8)

transform = transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor()])

batch_size = 64

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Train neural network and get results 
print("Training will begin")
for epoch in range(5):
    train_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = torch.max(outputs, 1)[1]
        train_correct = (pred == labels).sum()
        train_acc += train_correct.item()
        
    print(f'Epoch {epoch + 1}, Loss: {train_loss / len(trainloader)}, Acc: {train_acc / train_len}')

# test the trained model
net.eval()
test_loss = 0.0
test_acc = 0.0

conf_mat = torch.zeros(4,4)

for data in testloader:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = net(inputs)
    loss = criterion(outputs, labels)

    test_loss += loss.item()
    pred = torch.max(outputs, 1)[1]
    test_correct = (pred == labels).sum()
    test_acc += test_correct.item()
    
    for i in range(len(labels.view(-1))):
        t = labels.view(-1)[i]
        p = pred.view(-1)[i]
        conf_mat[t.long(), p.long()] += 1

print(f'Test Loss: {test_loss / len(testloader)}, Test Acc: {test_acc / test_len}')
print(conf_mat)