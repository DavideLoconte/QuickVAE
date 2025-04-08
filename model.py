"""This module defines the encoder, decoder, and full CVAE architecture"""
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual block for use in convolutional VAEs.

    This block applies two convolutional layers with GELU activations and batch normalization,
    and includes a residual (skip) connection. If the input and output channels differ,
    a 1×1 convolution is applied to match dimensions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Optional skip connection if channel dimensions differ
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H, W)
        """
        return self.skip(x) + self.conv2(self.conv1(x))
class Decoder(torch.nn.Module):
    """
    Upsampling Conditional Decoder for a Variational Autoencoder (VAE).
    """

    def __init__(self, latent_size: int, class_vector_size: int, img_shape: tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        assert img_shape[1] % 4 == 0 and img_shape[2] % 4 == 0

        self.img_shape = img_shape  # (C, H, W)
        init_channels = 64
        init_size_h = img_shape[1] // 4  # Height
        init_size_w = img_shape[2] // 4  # Width

        self.fc = nn.Sequential(
            nn.Linear(latent_size + class_vector_size, init_channels * init_size_h * init_size_w),
            nn.GELU()
        )

        self.deconv = nn.Sequential(
            nn.Unflatten(1, (init_channels, init_size_h, init_size_w)),

            ResidualBlock(init_channels, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            ResidualBlock(32, 32),
            nn.ConvTranspose2d(32, self.img_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        x = torch.cat((z, cls), dim=1)
        x = self.fc(x)
        x = self.deconv(x)
        return x


class VariationalEncoder(torch.nn.Module):
    """
    Conditional Encoder for a Variational Autoencoder (VAE).
    Encodes an image and its class condition into a latent distribution.
    """
    def __init__(self, input_shape, class_size, latent_size):
        """
        Args:
            input_shape (tuple[int, int, int]): Shape of the input image.
            class_size (int): Dimension of the one-hot class vector.
            latent_size (int): Dimension of the latent space.
        """
        super().__init__()
        assert(input_shape[1] % 4 == 0 and input_shape[2] % 4 == 0)

        self._model = torch.nn.Sequential(
            ResidualBlock(input_shape[0], 8),
            torch.nn.GELU(),
            ResidualBlock(8, 16),
            torch.nn.MaxPool2d(2),
            torch.nn.GELU(),
            ResidualBlock(16, 32),
            torch.nn.MaxPool2d(2),
            torch.nn.GELU(),
            ResidualBlock(32, 64),
            torch.nn.MaxPool2d(2),
            torch.nn.GELU(),
            torch.nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy_out = self._model[:-1](dummy)  # remove Flatten()
            lin_input_size = dummy_out.numel() + class_size

        self._means = torch.nn.Sequential(
            torch.nn.Linear(lin_input_size, latent_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size * 2, latent_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size * 2, latent_size)
        )

        self._logvar  = torch.nn.Sequential(
            torch.nn.Linear(lin_input_size, latent_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size * 2, latent_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size * 2, latent_size)
        )


    def forward(self, x: torch.Tensor, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).
            cls (torch.Tensor): One-hot class vector of shape (B, class_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log variance tensors of shape (B, latent_size).
        """
        x = self._model(x)
        x = torch.cat((x, cls), dim=1)
        x_mean = self._means(x)
        x_logvar = self._logvar(x)
        return x_mean, x_logvar


class VariationalAutoencoder(torch.nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) combining an encoder and decoder.
    """
    def __init__(self,
                 input_shape: tuple[int, int, int] = (1, 28, 28),
                 latent_size: int = 128,
                 class_size: int = 10,
                 beta: float = 1.0):
        """
        Args:
            input_shape (tuple[int, int, int]): Shape of the input image.
            latent_size (int): Dimension of the latent space.
            class_size (int): Dimension of the one-hot class vector.
            beta (float): Weight of the KL divergence term.
        """
        super().__init__()
        self._input_shape = input_shape
        self._encoder = VariationalEncoder(input_shape, class_size, latent_size)
        self._decoder = Decoder(latent_size, class_size, input_shape)
        self.beta = beta
        self.latent_size = latent_size

    def forward(self, x: torch.Tensor, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CVAE.

        Args:
            x (torch.Tensor): Input image of shape (B, C, H, W).
            cls (torch.Tensor): One-hot class vector of shape (B, class_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reconstructed image, mean, and log variance.
        """
        x_mean, x_logvar = self._encoder(x, cls)
        z = self._reparameterize(x_mean, x_logvar)
        return self._decoder(z, cls), x_mean, x_logvar

    def _reparameterize(self, x_mean: torch.Tensor, x_logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent space.

        Args:
            x_mean (torch.Tensor): Mean tensor.
            x_logvar (torch.Tensor): Log variance tensor.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std)
        return x_mean + eps * std

    def encode(self, x: torch.Tensor, cls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image and class condition.

        Args:
            x (torch.Tensor): Input image.
            cls (torch.Tensor): Class vector.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Mean and log variance.
        """
        return self._encoder(x, cls)

    def decode(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """Generate a sample from a latent space

        Args:
            x (torch.Tensor): a tensor of shape (B, OUTPUT_SIZE)

        Returns:
            torch.Tensor: an image of shape (B, C, H, W), where C, H, W = INPUT_SHAPE
        """
        return self._decoder(x, cls)

    def save(self, path: str, loss: float = float("inf")):
        """Save the model to a file

        Args:
            path (str): the path to save the model to
            loss (float, optional): the loss of the model. Defaults to float("inf").
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "loss": loss
        }
        torch.save(checkpoint, path)

    def load(self, path: str, loss: float | None = None) -> float:
        """Load the model from a file

        Args:
            path (str): the path to load the model from
            loss (float | None, optional): a loss value or None. Load the model only if it has a lower loss.

        Returns:
            float: the loss of the model
        """
        checkpoint = torch.load(path)

        if loss is not None and loss < checkpoint["loss"]:
            return checkpoint["loss"]

        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint["loss"]
