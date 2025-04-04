"""This module defines the encoder, decoder, and full CVAE architecture"""

import torch

class Decoder(torch.nn.Module):
    """
    Conditional Decoder for a Variational Autoencoder (VAE).
    Reconstructs an image from a latent vector and a class condition.
    """

    def __init__(self, latent_size: int, class_vector_size: int, img_shape: tuple[int, int, int] = (1, 28, 28)):
        """
        Args:
            latent_size (int): Dimension of the latent space.
            class_vector_size (int): Dimension of the one-hot class vector.
            img_shape (tuple[int, int, int]): Shape of the target image.
        """
        super().__init__()
        torch_dims = img_shape[0] * img_shape[1] * img_shape[2]
        self._model = torch.nn.Sequential(
            torch.nn.Linear(latent_size + class_vector_size,latent_size + class_vector_size),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size + class_vector_size, torch_dims * 32),
            torch.nn.GELU(),
            torch.nn.Unflatten(1, (32, img_shape[1], img_shape[2])),
            torch.nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(16, 8, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(8, 4, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(4, img_shape[0], kernel_size=5, stride=1, padding=2),
            torch.nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the decoder.

        Args:
            x (torch.Tensor): Latent vector of shape (B, latent_size).
            cls (torch.Tensor): One-hot class vector of shape (B, class_vector_size).

        Returns:
            torch.Tensor: Reconstructed image of shape (B, C, H, W).
        """
        x = torch.cat((x, cls), dim=1)
        return self._model(x)


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
        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 4, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(4, 8, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * input_shape[1] * input_shape[2], latent_size),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size, latent_size),
            torch.nn.GELU(),
        )

        self._means = torch.nn.Sequential(
            torch.nn.Linear(latent_size+class_size, latent_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size * 2, latent_size * 2),
            torch.nn.GELU(),
            torch.nn.Linear(latent_size * 2, latent_size)
        )

        self._logvar  = torch.nn.Sequential(
            torch.nn.Linear(latent_size+class_size, latent_size * 2),
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
            return checkpoint["loss"] # Early stop, model is already better

        self.load_state_dict(checkpoint["state_dict"])
        return checkpoint["loss"]
