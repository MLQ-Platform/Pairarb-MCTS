import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class SpreadRegressorConfig:
    seq_len: int = 10
    pred_len: int = 5
    num_features: int = 1
    num_filters: int = 64
    kernel_size: int = 3


class SpreadRegressor(nn.Module):
    """
    A Convolutional Neural Network (CNN) model for predicting spreads.
    """

    def __init__(self, config: SpreadRegressorConfig):
        """
        Initializes the SpreadRegressor with the given configuration.

        Args:
            config (SpreadRegressorConfig): Configuration object containing model parameters.
        """
        super(SpreadRegressor, self).__init__()

        self.config = config

        # 1D CNN 레이어
        self.conv1 = nn.Conv1d(
            in_channels=config.num_features,
            out_channels=config.num_filters,
            kernel_size=config.kernel_size,
            padding="same",
        )
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        # Fully Connected Layer
        self.fc = nn.Linear(
            config.num_filters * config.seq_len, config.pred_len * config.num_features
        )

    def forward(self, x):
        """
        Performs the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_len, num_features).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, pred_len, num_features).
        """
        if x.shape[1] != self.config.seq_len:
            raise ValueError(f"Input sequence length must be {self.config.seq_len}")

        if x.shape[2] != self.config.num_features:
            raise ValueError(
                f"Input number of features must be {self.config.num_features}"
            )

        # (batch_size, seq_len, num_features) -> (batch_size, num_features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)

        # (batch_size, pred_len * num_features) -> (batch_size, pred_len, num_features)
        x = x.reshape(-1, self.config.pred_len, self.config.num_features)
        return x

    def predict(self, series: np.array) -> np.array:
        """
        Makes a prediction based on the input series.

        Args:
            series (np.array): Input series with shape (seq_len, num_features).

        Returns:
            np.array: Predicted output with shape (pred_len, num_features).
        """
        # Convert series to tensor (1, seq_len, num_features)
        input_tensor = torch.tensor(series).float().unsqueeze(0)
        # Feed Forwarding
        output = self(input_tensor).squeeze(0).detach().numpy()
        return output
