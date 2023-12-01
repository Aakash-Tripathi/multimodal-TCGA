import torch
from torch import nn
import lightning as L
import random
import torch.nn.functional as F


class TransformerModel(L.LightningModule):
    def __init__(
        self, input_dim, hidden_dim, n_layers, n_heads, dropout_rate, patch_size=1024
    ):
        """
        Constructor for the TransformerModel.
        Args:
        input_dim (int): Dimension of the input embeddings.
        hidden_dim (int): The dimension of the hidden layer.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of heads in the MultiHeadAttention layer.
        dropout_rate (float): Dropout rate.
        """
        super(TransformerModel, self).__init__()
        torch.set_float32_matmul_precision("medium")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        self.embedding_transform = nn.Linear(7 * 7 * 2048, patch_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
        x (Tensor): Input tensor (embeddings).
        Returns:
        Tensor: The output of the transformer encoder.
        """
        x = self.dropout(x)
        transformed = self.transformer_encoder(x)
        return transformed

    def configure_optimizers(self):
        """
        Configure the optimizer to use for training.
        Returns:
        torch.optim.Optimizer: The optimizer to use for training.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.95
        )
        return [optimizer], [lr_scheduler]

    def contrastive_loss(self, anchor, positive, negative, margin=1.0, epsilon=1e-6):
        """
        Contrastive loss function.
        Args:
        anchor (Tensor): Anchor embeddings.
        positive (Tensor): Positive embeddings.
        negative (Tensor): Negative embeddings.
        margin (float): Margin for the loss.
        epsilon (float): Epsilon value to prevent log(0).
        Returns:
        Tensor: The loss value.
        """
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = torch.log(distance_positive / (distance_negative + epsilon))
        losses = torch.clamp_min(losses + margin, min=0.0)
        return losses.mean()

    def process_inputs(self, clinical, pathology, radiology):
        batch_size, patches, _, _, _ = pathology.shape

        pathology = pathology.view(batch_size, patches, -1)
        radiology = radiology.view(batch_size, patches, -1)

        pathology = self.embedding_transform(pathology)
        radiology = self.embedding_transform(radiology)

        combined = torch.cat((clinical, pathology, radiology), dim=1)
        return combined.permute(1, 0, 2)

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
        batch (Tensor): The batch of data.
        batch_idx (int): The index of the batch.
        Returns:
        Tensor: The loss for the current step.
        """
        clinical, pathology, radiology = batch
        combined = self.process_inputs(clinical, pathology, radiology)
        print()
        print(combined.shape)  # torch.Size([51, 8, 1024])
        # Forward pass
        output = self(combined)
        print(output.shape)  # torch.Size([51, 8, 1024])
        anchor = output[0, :, :]
        print(anchor.shape)  # torch.Size([8, 1024])
        random_positive = random.choice(range(1, 25))
        random_negative = random.choice(
            range(25, 51)
        )  # only because the radiology is randomly generated for now, in the future we will have to change this to use another patient's samples
        positive = output[random_positive, :, :]
        print(positive.shape)  # torch.Size([8, 1024])
        negative = output[random_negative, :, :]
        print(negative.shape)  # torch.Size([8, 1024])
        print()
        loss = self.contrastive_loss(anchor, positive, negative)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Args:
        batch (Tensor): The batch of data.
        batch_idx (int): The index of the batch.
        Returns:
        Tensor: The loss for the current step.
        """
        clinical, pathology, radiology = batch
        combined = self.process_inputs(clinical, pathology, radiology)
        # Forward pass
        output = self(combined)
        print(output.shape)  # torch.Size([51, 8, 1024])
        anchor = output[0, :, :]
        print(anchor.shape)  # torch.Size([8, 1024])
        random_positive = random.choice(range(1, 25))
        random_negative = random.choice(
            range(25, 51)
        )  # only because the radiology is randomly generated for now, in the future we will have to change this to use another patient's samples
        anchor = output[0, :, :]
        positive = output[random_positive, :, :]
        negative = output[random_negative, :, :]
        loss = self.contrastive_loss(anchor, positive, negative)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
