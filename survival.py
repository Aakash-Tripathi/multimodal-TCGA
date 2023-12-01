import json
from tqdm import tqdm
import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import RichProgressBar
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.models.survival import DeepSurv
from src.utils import calculate_survival_time


class SurvivalDataset(TensorDataset):
    def __init__(self, features, labels):
        super().__init__(features, labels)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("/mnt/w/data/TCGA-LUAD.json") as f:
        data = json.load(f)

    patient_data = {}
    for case_id, patient in tqdm(data.items()):
        patient_data[case_id] = {}
        feilds_to_keep = [
            "age_at_index",
            "days_to_birth",
            "vital_status",
            "year_of_birth",
            "age_at_diagnosis",
            "year_of_diagnosis",
        ]
        for feild in feilds_to_keep:
            if feild in patient:
                patient_data[case_id][feild] = patient[feild]
            else:
                patient_data[case_id][feild] = None

    for feild in feilds_to_keep:
        print(feild, len([x for x in patient_data.values() if x[feild] is None]))
    survival_times = calculate_survival_time(patient_data)

    features = torch.tensor(
        np.random.rand(len(patient_data), 1024), dtype=torch.float32
    )
    labels = torch.tensor(
        [survival_times[x] for x in patient_data], dtype=torch.float32
    )

    print("Features shape", features.shape)
    print("Labels shape", labels.shape)

    dataset = SurvivalDataset(features, labels)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=31
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, num_workers=31
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=31
    )

    surv_model = DeepSurv(lr=0.001, dropout=0.1).to(device)
    trainer = pl.Trainer(
        max_epochs=10, log_every_n_steps=1, callbacks=[RichProgressBar()]
    )

    trainer.fit(surv_model, train_dataloader, val_dataloader)
    trainer.test(surv_model, test_dataloader)


if __name__ == "__main__":
    main()
