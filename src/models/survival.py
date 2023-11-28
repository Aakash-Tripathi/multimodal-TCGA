from torch import nn
import torch
import torch.nn as nn
import numpy as np


# Define the DeepSurv neural network model
class DeepSurv(nn.Module):
    def __init__(self, input_size):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Output is a risk score
        return x


# Define the Cox partial likelihood for loss calculation
def cox_partial_likelihood(risk_scores, event_times, event_occurred):
    # Ensure the data is sorted by time
    sorted_indices = np.argsort(-event_times)
    risk_scores_sorted = risk_scores[sorted_indices]
    event_occurred_sorted = event_occurred[sorted_indices]
    risk_scores_cumsum = torch.cumsum(torch.exp(risk_scores_sorted), dim=0)

    # Calculate the Cox partial likelihood
    likelihood = risk_scores_sorted - torch.log(risk_scores_cumsum)
    observed_likelihood = (
        likelihood * event_occurred_sorted
    )  # consider only observed events
    negative_log_likelihood = -torch.sum(observed_likelihood) / torch.sum(
        event_occurred_sorted
    )

    return negative_log_likelihood


class MultiClassSurvivabilityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Constructor for the SurvivabilityPredictionModel.

        Args:
        embedding_dim (int): Dimension of the input embeddings.
        additional_features (int): Number of additional features.
        hidden_dim (int): Dimension of the hidden layer.
        output_dim (int): Dimension of the output layer (usually 1 for binary classification).
        """
        super(MultiClassSurvivabilityModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
        x (Tensor): Input tensor (combined embeddings and additional features).

        Returns:
        Tensor: Output tensor representing survivability prediction.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
