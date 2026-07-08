import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batch_size = 128
learning_rate =0.001
num_epochs = 10

# Data
transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Modell
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Enkóder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride =2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    # forward függvény
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Model, loss, optimizer
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) 

# Training
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, _ in train_loader:
        images = images.to(device)

        # Forward pass
        output = model(images)
        loss = criterion(output, images)

        # Backward pass

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}")

model.eval()
dataiter = iter(test_loader)
images, _ = next(dataiter)
images = images.to(device)
reconstructed = model(images)

images = images.cpu().detach().numpy()
reconstructed = reconstructed.cpu().detach().numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Eredeti képek
    plt.subplot(2, n, i+1)
    plt.imshow(images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Rekonstruált képek
    plt.subplot(2, n , i+1+n)
    plt.imshow(reconstructed[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()
