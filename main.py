import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import RichProgressBar

import imageio
from matplotlib import pyplot as plt

from lib.MINDS import MINDS
from src.models.transformer import TransformerModel
from src.models.survival import MultiClassSurvivabilityModel
from src.models.modality import ClinincalEmbeddings, REMEDISpathology, REMEDISradiology
from src.data import EmbeddingsDataset, CombinedDataset
from src.utils import (
    generate_summary_from_json,
    process_group,
    convert_numpy,
    flatten_json,
)

#  Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_cohort_data(project_id):
    # * Step 1: Generate the cohort data from the MINDS database
    minds = MINDS()
    filename = "data/" + project_id
    tables = minds.get_tables()
    if Path(f"{filename}.json").exists():
        json_objects = json.load(open(f"{filename}.json"))
    else:
        json_objects = {}
        for table in tqdm(tables["Tables_in_nihnci"]):
            query = f"SELECT * FROM nihnci.{table} WHERE project_id='{project_id}'"
            df = minds.query(query)
            for case_id, group in tqdm(df.groupby("case_submitter_id"), leave=False):
                if case_id not in json_objects:
                    json_objects[case_id] = {}
                common_fields, nested_objects = process_group(group)
                json_objects[case_id].update(common_fields)
                json_objects[case_id][table] = nested_objects
        # * Step 2: save the resulting json file for all EHR
        with open(f"{filename}.json", "w") as fp:
            json.dump(json_objects, fp, indent=4, default=convert_numpy)
    # TODO: Step 3: acquire the raw data for pathology and radiology modalities


def generate_modality_embeddings():
    # TODO: Step 1: Generate the embeddings for EHR, pathology, and radiology modalities
    # TODO: Step 2: Save the embeddings and patient data in the data folder for each modality under each patient_id folder
    pass


# * Step 4: train the mapper models to map the embeddings to the transformer models latent space (pathology-mapper.pth, radiology-mapper.pth)
# * Step 5: generate the embeddings using the ehr, pathology, and radiology data from the transformer model
# * Step 6: train the survival model on the embeddings generated and the survival data from the original cohort


def main():
    # * Step 1: generate_cohort_data()
    generate_cohort_data(project_id="TCGA-LUAD")

    # * Step 2: generate_modality_embeddings()
    with open(os.getcwd() + "/data/TCGA-LUAD.json", "r") as file:
        json_objects = json.load(file)

    clinical_model = ClinincalEmbeddings(model="UFNLP/gatortron-base", device=device)

    for case_id, patient_data in tqdm(json_objects.items()):
        if os.path.exists(
            os.getcwd() + f"/data/TCGA-LUAD/embeddings/{case_id}/ehr.npy"
        ):
            continue
        summary = generate_summary_from_json(patient_data)
        embedding = clinical_model.generate_embeddings(summary)
        embedding = embedding.detach().cpu().numpy()
        if not os.path.exists(os.getcwd() + f"/data/TCGA-LUAD/embeddings/{case_id}"):
            os.makedirs(os.getcwd() + f"/data/TCGA-LUAD/embeddings/{case_id}")
        np.save(
            os.getcwd() + f"/data/TCGA-LUAD/embeddings/{case_id}/ehr.npy", embedding
        )

    embeddings = []
    for case_id, patient_data in tqdm(json_objects.items()):
        embedding = np.load(
            os.getcwd() + f"/data/TCGA-LUAD/embeddings/{case_id}/ehr.npy"
        )
        # embedding = embedding.reshape(512, 1024)
        embeddings.append(embedding)
    print("Embeddings shape:", np.array(embeddings).shape)

    # * Step 3: pretrain the transformer model on the EHR embeddings (initial-clinical.pth)
    dataloader = DataLoader(
        dataset=EmbeddingsDataset(embeddings, device), batch_size=8, shuffle=True
    )
    model = TransformerModel(
        input_dim=1024,  # Embedding dimension
        hidden_dim=2048,  # Hidden dimension
        n_layers=4,  # Number of transformer layers
        n_heads=8,  # Number of heads in MultiHeadAttention
        dropout_rate=0.1,  # Dropout rate
    ).to(device)
    trainer = Trainer(
        max_epochs=10,
        log_every_n_steps=1,
        callbacks=RichProgressBar(),
    )
    trainer.fit(model, dataloader)

    images = []
    for i in range(10):
        images.append(imageio.v3.imread(f"plots/{i}.png"))
    # change the fps to change slow down the gif
    imageio.mimsave("plots/movie.gif", images, fps=1)


def main2():
    with open("TCGA-LUAD.json", "r") as file:
        data = json.load(file)
    flattened_data = [flatten_json(data[patient]) for patient in data]
    clinical_data = pd.DataFrame(flattened_data)

    # use the embeddings generated from the Transformer model instead of the ones from Redis
    transformer_model = TransformerModel(
        input_dim=1024, hidden_dim=2048, n_layers=4, n_heads=8, dropout_rate=0.1
    ).to(device)
    transformer_model.load_state_dict(torch.load("models/initial-clinical.pth"))
    transformer_model.eval()

    embeddings = []
    use_input_embeddings = False
    for case_id, patient_data in tqdm(data.items()):
        input_embeddings = np.load(f"data/TCGA-LUAD/embeddings/{case_id}/ehr.npy")
        input_embeddings = input_embeddings.reshape(512, 1024)
        input_embeddings = torch.tensor(input_embeddings, dtype=torch.float32).to(
            device
        )
        if use_input_embeddings:
            embeddings.append(input_embeddings.detach().cpu().numpy())
        else:
            output_embedding = transformer_model(input_embeddings)
            embeddings.append(output_embedding.detach().cpu().numpy())

        # empty the gpu memory
        del input_embeddings
        del output_embedding

    clinical_data["embeddings"] = embeddings
    clinical_data["vital_status_label"] = clinical_data["vital_status"].map(
        {"Alive": 1, "Dead": 2}
    )
    clinical_data["vital_status_label"].fillna(0, inplace=True)  # Unknown as 0
    clinical_data_embeddings = clinical_data["embeddings"].values
    clinical_data_embeddings = np.array(
        [embedding.flatten() for embedding in clinical_data_embeddings]
    )
    labels = clinical_data["vital_status_label"].values

    print("Embeddings shape:", clinical_data_embeddings.shape)
    print("Unique label values:", np.unique(labels))
    print("Labels shape:", labels.shape)

    train_data, test_data, train_labels, test_labels = train_test_split(
        clinical_data_embeddings, labels, test_size=0.2, random_state=42
    )
    train_dataset = CombinedDataset(train_data, train_labels)
    test_dataset = CombinedDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_dim = 512 * 1024
    model = MultiClassSurvivabilityModel(input_dim, 256, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training the model
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, (embeddings, labels) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(embeddings)
            loss = criterion(outputs, labels.long())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print the output predictions and the labels
        max_idx = torch.argmax(outputs, dim=1)
        print(max_idx.cpu().numpy(), "\n", labels.cpu().numpy())
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
    # main2()
