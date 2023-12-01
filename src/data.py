import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PatientEmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir, max_patches):
        self.embeddings_dir = embeddings_dir
        self.cases = os.listdir(self.embeddings_dir)
        self.max_patches = max_patches

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        clinical_path = f"{self.embeddings_dir}/{case_id}/ehr.npy"
        pathology_path = f"{self.embeddings_dir}/{case_id}/pathology.npy"
        radiology_path = f"{self.embeddings_dir}/{case_id}/radiology.npy"

        clinical_emb = (
            np.load(clinical_path)
            if os.path.exists(clinical_path)
            else np.zeros((1024,))
        )

        # load the maximum number of patches specified
        pathology_emb = (
            np.load(pathology_path)
            if os.path.exists(pathology_path)
            else np.zeros((1, 7, 7, 2048))
        )
        pathology_emb = pathology_emb[: self.max_patches]

        radiology_emb = (
            np.load(radiology_path)
            if os.path.exists(radiology_path)
            else np.zeros((1, 7, 7, 2048))
        )
        radiology_emb = radiology_emb[: self.max_patches]

        return (
            torch.tensor(clinical_emb, dtype=torch.float),
            torch.tensor(pathology_emb, dtype=torch.float),
            torch.tensor(radiology_emb, dtype=torch.float),
        )


def custom_collate_fn(batch):
    # Separate clinical, pathology, and radiology embeddings
    clinical, pathology, radiology = zip(*batch)

    # Pad pathology and radiology sequences to the maximum length in the batch
    pathology_padded = pad_sequence(pathology, batch_first=True)
    radiology_padded = pad_sequence(radiology, batch_first=True)

    # Convert clinical embeddings to tensors
    clinical_tensor = torch.stack(clinical)

    return clinical_tensor, pathology_padded, radiology_padded
