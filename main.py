import os
import torch
from torch.utils.data import DataLoader
import warnings
from lightning.pytorch.profilers import AdvancedProfiler

warnings.filterwarnings("ignore")

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar


from src.models.transformer import TransformerModel
from src.models.survival import DeepSurv
from src.data import PatientEmbeddingDataset, custom_collate_fn
from src.utils import InitialSetup, generate_pathology_embeddings_shapes_csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    initial_setup = InitialSetup(
        project_id="TCGA-LUAD",
        embeddings_dir=r"/mnt/w/data/TCGA-LUAD/embeddings/",
        raw_data_dir=r"/mnt/d/TCGA-LUAD",
    )
    json_objects = initial_setup.generate_cohort_data()
    initial_setup.generate_clinical_embeddings()
    initial_setup.generate_pathology_embeddings()
    initial_setup.generate_radiology_embeddings()

    if not os.path.exists("pathology-embeddings-shapes.csv"):
        generate_pathology_embeddings_shapes_csv(
            filename="pathology-embeddings-shapes.csv"
        )

    dataset = PatientEmbeddingDataset(
        embeddings_dir=r"/mnt/w/data/TCGA-LUAD/embeddings/",
        max_patches=25,
    )
    train_size = int(0.8 * len(dataset))  # 80% of data for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=custom_collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )
    model = TransformerModel(
        input_dim=1024,
        hidden_dim=1024,
        n_layers=4,
        n_heads=8,
        dropout_rate=0.2,
    ).to(device)
    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        callbacks=RichProgressBar(),
    )
    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)
    trainer.save_checkpoint("models/v1.pth")


if __name__ == "__main__":
    main()
