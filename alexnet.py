from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
trainset = torchvision.datasets.ImageFolder(
    root="./data/cifar10/train",
    transform=transform
)

testset = torchvision.datasets.ImageFolder(
    root="./data/cifar10/test",
    transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding = 1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding = 1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding = 1)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256*4*4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 10)
        )


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


def train_model(model, optimizer, num_epoch):
  for epoch in range(num_epoch):
    print(f"epoch: {epoch+1}")
    loss_epoch = 0
    model.train()
    for inputs, labels in trainloader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss_epoch += loss.item()
      loss.backward()
      optimizer.step()
    scheduler.step()
    print(loss_epoch / len(trainloader))

def eval_model(model):
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  print(f'Accuracy of the network on the test images: {100 * correct / total}')

import torch
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
alexnet = AlexNet().to(device)
print("Running on:", device)
optimizer = optim.SGD(alexnet.parameters(), lr = 0.01, momentum= 0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

num_epoch = 20
train_model(alexnet, optimizer, num_epoch)
eval_model(alexnet)