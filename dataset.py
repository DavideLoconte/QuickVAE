"""This file implements the dataset class for the Quick, Draw! project."""

import os

import torch
import requests
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.transforms import v2

class QuickDrawDataset(Dataset):
    """Dataset class for Quick, Draw!"""
    BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
    CATEGORY_LIST_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"

    def __init__(
        self,
        path: os.PathLike = "dataset",
        categories: list[str] | None = None,
    ):
        """Initialize the dataset class

        Args:
            path (os.PathLike): the path to the dataset directory containing .npy files
            categories (list[str], optional): list of category names. If None, load all found.
            transform (v2.Transform, optional): data augmentation transform. Defaults to None.
            max_items_per_class (int, optional): limit number of items per class. Defaults to None.
        """
        self.data = []
        self.labels = []
        self.categories = categories
        self.category_to_idx = {}
        self.path = path
        self._fetch()

        all_files = os.listdir(path)
        if categories is None:
            categories = [f.replace(".npy", "") for f in all_files if f.endswith(".npy")]

        for idx, category in enumerate(categories):
            file_path = os.path.join(path, category + ".npy")
            if not os.path.exists(file_path):
                continue
            samples = np.load(file_path)
            self.data.append(samples)
            self.labels.extend([idx] * len(samples))
            self.category_to_idx[category] = idx

        self.data = np.concatenate(self.data, axis=0).astype(np.float32) / 255.0
        self.labels = np.array(self.labels)
        self.shape = (1, 28, 28)

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),  # small rotation
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # small shift
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the image and one-hot encoded label

        Args:
            idx (int): index

        Returns:
            tuple[torch.Tensor, torch.Tensor]: image and one-hot label
        """
        image = self.data[idx].reshape(28, 28)
        label_idx = self.labels[idx]
        image = self.transform(image)

        # One-hot encoding
        one_hot = torch.zeros(len(self.category_to_idx), dtype=torch.float32)
        one_hot[label_idx] = 1.0

        return image, one_hot

    def _fetch(self):
        """Download missing .npy files"""
        os.makedirs(self.path, exist_ok=True)
        for category in self.categories:
            print(f"Downloading {category}...")
            filename = category.replace(' ', '%20') + ".npy"

            if os.path.exists(os.path.join(self.path, category + ".npy")):
                continue # already downloaded

            url = self.BASE_URL + filename
            output_path = os.path.join(self.path, category + ".npy")

            if os.path.exists(output_path):
                continue

            response = requests.get(url, stream=True)
            if response.status_code != 200:
                raise ConnectionError(f"Failed to download {category}")

            total = int(response.headers.get('content-length', 0))
            with open(output_path, "wb") as f, tqdm( desc=category, total=total, unit='B', unit_scale=True) as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))

    def pick(self, number: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick a random sample from the dataset"""
        idx = np.random.randint(0, len(self.data), number)
        return self[idx]