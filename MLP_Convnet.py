import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#Importálás és preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

# training adathalmaz
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# test adathalmaz
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Egyszerű neurális háló
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = torch.nn.Linear(32 * 32 * 3, 128)
        self.fc2 = torch.nn.Linear(128,64)
        self.fc3 = torch.nn.Linear(64,64)
        self.fc4 = torch.nn.Linear(64,32)
        self.fc5 = torch.nn.Linear(32,10)

    def forward(self,x):
        x = x.view(-1,32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mlp = MLP().to(device)

#Modell tanítása
print("Első tanítás")
criterion = nn.CrossEntropyLoss() # loss függvény
optimizer_mlp = optim.SGD(mlp.parameters(), lr=0.01, momentum = 0.9) # optimizer
#optimizer = optim.Adam(mlp.parameters(), lr = 0.001)

def train_model(model, optimizer, num_epoch):
  for epoch in range(num_epoch):
    loss_epoch = 0
    model.train()
    for inputs, labels in trainloader:
      # data egy [inputs, labels] formájú lista
      inputs, labels = inputs.to(device), labels.to(device)
      # nullázzuk a gradienseket
      optimizer.zero_grad()
      # forward pass
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss_epoch += loss
      # backward pass
      loss.backward()
      # optimization step
      optimizer.step()
    if epoch % 5 == 0:
        print(f"epoch: {epoch}")
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

num_epoch = 20
train_model(mlp, optimizer_mlp, num_epoch)


# Convnet
class Convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
convnet = Convnet().to(device)

optimizer_conv = optim.SGD(convnet.parameters(), lr=0.01, momentum = 0.9)
train_model(convnet, optimizer_conv, num_epoch)

eval_model(mlp)
eval_model(convnet)
