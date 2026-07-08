import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 10
    lr = 1e-3
    latent_dim = 32
    seed = 42
    torch.manual_seed(seed)

    # Dataset
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Modell
    class MLP_AE(nn.Module):
        def __init__(self, latent=32):
            super().__init__()
            self.flatten = nn.Flatten()
            self.encoder = nn.Sequential(
                nn.Linear(28 * 28, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, latent),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 28 * 28),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.encoder(self.flatten(x))
            x_hat = self.decoder(z)
            x_hat = x_hat.view(-1, 1, 28, 28)
            return x_hat, z

    model = MLP_AE(latent=latent_dim).to(device)
    criterion = nn.BCELoss(reduction="mean")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, _ in train_dl:
            x = x.to(device, non_blocking=True)

            # Forward
            x_hat, _ = model(x)
            loss = criterion(x_hat, x)

            # Backpropagation
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_dl.dataset)
        print(f"Epoch {epoch:02d}/{epochs} | Train loss: {avg_loss:.4f}")

    # Visualization
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_dl))
        x = x.to(device)
        x_hat, _ = model(x[:8])
        x = x[:8].cpu()
        x_hat = x_hat.cpu()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(x[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Originals")
    axes[1, 0].set_title("Reconstructions")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()