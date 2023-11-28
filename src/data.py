import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, device):
        """
        Constructor for the Dataset.
        Args:
        embeddings (list of numpy arrays): The embeddings generated from clinical reports.
        """
        self.embeddings = [torch.tensor(e, dtype=torch.float32) for e in embeddings]
        self.device = device

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Retrieves the item (embedding) at the specified index.
        Args:
        idx (int): The index of the item to retrieve.
        Returns:
        Tensor: The embedding at the specified index.
        """
        return self.embeddings[idx].to(self.device)


class CombinedDataset(Dataset):
    def __init__(self, combined_data, labels):
        self.combined_data = combined_data
        self.labels = labels

    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        return torch.tensor(self.combined_data[idx], dtype=torch.float), torch.tensor(
            self.labels[idx], dtype=torch.float
        )
