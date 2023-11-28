import torch
from torch import nn
import lightning as L
from matplotlib import pyplot as plt


class TransformerModel(L.LightningModule):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads, dropout_rate):
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
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
        x (Tensor): Input tensor (embeddings).
        Returns:
        Tensor: The output of the transformer encoder.
        """
        transformed = self.transformer_encoder(x)
        return transformed

    def configure_optimizers(self):
        """
        Configure the optimizer to use for training.
        Returns:
        torch.optim.Optimizer: The optimizer to use for training.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
        batch (Tensor): The batch of data.
        batch_idx (int): The index of the batch.
        Returns:
        Tensor: The loss for the current step.
        """
        embeddings = batch
        outputs = self(embeddings)
        loss = nn.MSELoss()(outputs, embeddings)
        self.log("train_loss", loss)

        if batch_idx % 100 == 0:
            plt.figure(figsize=(10, 5))
            plt.hist(
                outputs.flatten().detach().cpu().numpy(),
                bins=100,
                alpha=0.5,
                label="Outputs",
            )
            plt.hist(
                embeddings.flatten().detach().cpu().numpy(),
                bins=100,
                color="orange",
                alpha=0.5,
                label="Embeddings",
            )
            plt.legend()
            # batch_idx is 0 for some reason so use the epoch number instead
            plt.savefig(f"plots/{self.current_epoch}.png")
            plt.close()
        return loss

    def save_model(self, path):
        """
        Saves the model.
        Args:
        path (str): The path to save the model to.
        """
        torch.save(self.state_dict(), path)
