"""
Conditional Variational Autoencoder (CVAE) implementation for the Quick, Draw! dataset.
This script loads the dataset, trains the model,
and saves visualizations of both real samples and generated samples conditioned on the same classes.
"""
import os
import sys

import tqdm
import torch
import matplotlib.pyplot as plt

from dataset import QuickDrawDataset
from model import VariationalAutoencoder
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.utils.data import random_split
import torch.nn.functional as F

# Some parameters that is safe to change
SPLIT_RATIO = 0.8
"""Split ratio for train and eval dataloaders"""

BATCH_SIZE = 128
"""Batch size for train and eval dataloaders"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""The device to train the model on"""

ITERS = 10000
"""Number of iterations to train the model for"""

LEARNING_RATE = 1e-4
"""Learning rate for the AdamW optimizer"""

CLASSES: tuple[str, str, str] = ["cat", "dog", "bird"]
"""The classes to train the model on"""

def main() -> int:
    """The main entry point for the script

    Returns:
        int: Exit code
    """
    assert(len(CLASSES) == 3)
    print("Initializing...")
    train_loader, eval_loader = get_data_loaders(batch_size=BATCH_SIZE, split=SPLIT_RATIO, persistent_workers=True, num_workers=8, pin_memory=True)

    classes = CLASSES[:3]
    latent_dim = 128

    model, min_loss = get_model(checkpoint_path="checkpoint.pth", dataset_shape=(1, 28, 28), latent_dim=latent_dim, classes=len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if not os.path.exists("epochs"):
        os.makedirs("epochs")

    losses: list[tuple[int, float]] = []
    should_break = False

    for i in range(ITERS):
        beta = min(4.0, i / 100) # Simulate annealing, by slowly increasing the weight of the KL divergence term

        try:
            train_loss = train_model(model, optimizer, train_loader, beta)
        except KeyboardInterrupt:
            print("Training interrupted by user.")
            should_break = True

        loss = eval_model(model, optimizer, eval_loader, beta)
        real_images, fake_images = generate_model_samples(model, eval_loader, latent_dim)
        save_model_samples(f"epochs/{i}.png", real_images, fake_images)
        save_model_samples("last.png", real_images, fake_images)

        print(f"Epoch: {i}, Train Loss: {train_loss}, Eval Loss: {loss}")

        if min_loss > loss:
            min_loss = loss
            model.save("checkpoint.pth", min_loss)
            save_model_samples("best.png", real_images, fake_images)
            print("✓ Saved checkpoint.")

        losses.append((i, train_loss, loss))
        plot_losses(losses, "loss.png")

        if should_break:
            break

    return 0

def get_data_loaders(batch_size: int = 128, split: float = 0.8, num_workers: int = 4, pin_memory: bool = True, persistent_workers = True) -> tuple[DataLoader, DataLoader]:
    """Get train and eval dataloaders

    Args:
        batch_size (int, optional): batch size. Defaults to 128.
        split (float, optional): split ratio. Defaults to 0.8.
        num_workers (int, optional): number of workers. Defaults to 4.
        pin_memory (bool, optional): pin memory. Defaults to True.
        persistent_workers (bool, optional): persistent workers. Defaults to True.

    Returns:
        tuple[DataLoader, DataLoader]: train and eval dataloaders for the Quick, Draw! dataset
    """
    dataset = QuickDrawDataset(path="dataset", categories=CLASSES)
    train_size = int(split * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    return train_loader, eval_loader


def _get_loss(x: torch.Tensor, x_hat: torch.Tensor, x_mean: torch.Tensor, x_logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Return the total loss of the model

    Args:
        x (torch.Tensor): the true images
        x_hat (torch.Tensor): the reconstructed images
        x_mean (torch.Tensor): the mean tensor, output of the variational encoder
        x_logvar (torch.Tensor): the log variance tensor, output of the variational encoder
        beta (float, optional): the weight of the KL divergence term. Defaults to 1.0.

    Returns:
        torch.Tensor: the total loss
    """
    recon_loss = F.smooth_l1_loss(x_hat, x, reduction="sum") / x.shape[0]
    kl_loss = beta * (-0.5 * torch.sum(1 + x_logvar - x_mean.pow(2) - x_logvar.exp())) / x.shape[0]
    return recon_loss + kl_loss


def get_model(checkpoint_path: str, dataset_shape: tuple[int, int, int], latent_dim: int, classes: int) -> VariationalAutoencoder:
    """Get the model, eventually loading an existing checkpoint

    Returns:
        VariationalAutoencoder: The model
    """
    model = VariationalAutoencoder(input_shape=dataset_shape, latent_size=latent_dim, class_size=classes).to(DEVICE)
    score = float("inf")
    if os.path.exists(checkpoint_path):
        score = model.load(checkpoint_path)
        print("✓ Loaded checkpoint.")
    return model, score

def train_model(model: VariationalAutoencoder, optimizer: torch.optim.AdamW, train_loader: DataLoader, beta: float) -> float:
    """Execute a train pass the model

    Args:
        model (VariationalAutoencoder): The model
        optimizer (torch.optim.AdamW): The optimizer
        train_loader (DataLoader): The train dataloader
        beta (float): The weight of the KL divergence term
    Returns:
        float: The train loss
    """
    loss_sum = 0
    loss_count = 0
    for i, (x, y) in tqdm.tqdm(list(enumerate(train_loader)), desc="Training"):
        optimizer.zero_grad()
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        x_hat, x_mean, x_logvar = model(x, y)
        loss = _get_loss(x, x_hat, x_mean, x_logvar, beta)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_sum += loss.item()
            loss_count += 1
    return loss_sum / loss_count if loss_count > 0 else 0

def eval_model(model: VariationalAutoencoder, optimizer: torch.optim.AdamW, eval_loader: DataLoader, beta: float) -> float:
    """Execute an eval pass the model

    Args:
        model (VariationalAutoencoder): The model
        optimizer (torch.optim.AdamW): The optimizer
        eval_loader (DataLoader): The eval dataloader
        beta (float): The weight of the KL divergence term

    Returns:
        float: The eval loss
    """
    loss = 0
    count = 0
    with torch.no_grad():
        for i, (x, y) in tqdm.tqdm(list(enumerate(eval_loader)), desc="Evaluating"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            x_hat, x_mean, x_logvar = model(x, y)
            loss += _get_loss(x, x_hat, x_mean, x_logvar, beta).item()
            count += 1
    loss /= count
    return loss

def generate_model_samples(model: VariationalAutoencoder, eval_loader: DataLoader, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate samples from the model. Each fake image is generated using the same class as the corresponding real image

    Args:
        model (VariationalAutoencoder): the model
        eval_loader (DataLoader): the eval dataloader
        latent_dim (int): the latent dimension of the model

    Returns:
        tuple[torch.Tensor, torch.Tensor]:  a tuple of real and fake images
    """
    real_images, real_classes = next(iter(eval_loader))
    real_images = real_images.to(DEVICE)
    real_classes = real_classes.to(DEVICE)
    z = torch.randn(BATCH_SIZE, latent_dim).to(DEVICE)
    fake_images = model.decode(z, real_classes)
    return real_images, fake_images

def save_model_samples(filename: str, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
    """Save samples from the model

    Args:
        filename (str): the filename
        real_images (torch.Tensor): 16 real images
        fake_images (torch.Tensor): 16 fake images
    """
    # Select only the first 16 images
    real_images = real_images[:16]
    fake_images = fake_images[:16]
    with torch.no_grad():
        real_grid = make_grid(real_images.cpu(), nrow=4)
        fake_grid = make_grid(fake_images.cpu(), nrow=4)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Samples")
        plt.imshow(real_grid.permute(1, 2, 0).numpy())
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Generated Samples")
        plt.imshow(fake_grid.permute(1, 2, 0).numpy())
        plt.savefig(filename)
        plt.close()

def plot_losses(losses: list[tuple[int, float, float]], filename: str) -> None:
    """Plot the training and evaluation losses.

    Args:
        losses (list[tuple[int, float, float]]): list of (iteration, train_loss, eval_loss)
        filename (str): output plot filename
    """
    iterations, train_losses, eval_losses = zip(*losses)

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, train_losses, label='Train Loss')
    plt.plot(iterations, eval_losses, label='Eval Loss')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    sys.exit(main())
