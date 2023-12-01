import torch
from torch.utils.data import random_split, DataLoader

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

from src.models.transformer import TransformerModel
from src.data import PatientEmbeddingDataset, custom_collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # Generate dataset
    NUM_PATIENTS = 585
    EMBEDDING_SIZE = 1024
    dataset = PatientEmbeddingDataset(NUM_PATIENTS, EMBEDDING_SIZE)

    # Determine split sizes
    train_size = int(0.8 * len(dataset))  # 80% of data for training
    test_size = len(dataset) - train_size  # Remaining 20% for testing

    # Split the dataset
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=31,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=custom_collate_fn,
        num_workers=31,
    )

    model = TransformerModel(
        input_dim=EMBEDDING_SIZE,
        hidden_dim=2048,
        n_layers=4,
        n_heads=8,
        dropout_rate=0.1,
    ).to(device)
    trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        callbacks=[RichProgressBar()],
    )

    trainer.fit(model, train_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
