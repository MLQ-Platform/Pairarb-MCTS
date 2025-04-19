import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataclasses import dataclass
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SpreadRegressorTrainerConfig:
    batch_size: int
    epoch: int
    lr: float


class SpreadRegressorTrainer:
    """
    A trainer class for the SpreadRegressor model.
    """

    def __init__(self, config: SpreadRegressorTrainerConfig):
        """
        Initializes the SpreadRegressorTrainer with the given configuration.

        Args:
            config (SpreadRegressorTrainerConfig): Configuration object containing training parameters.
        """
        self.config = config

    def train(
        self,
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True,
    ) -> list[float]:
        """
        Trains the SpreadRegressor model using the provided training data.

        Args:
            model (nn.Module): The model to be trained.
            x_train (np.ndarray): Training input data with shape (data_len, num_features).
            y_train (np.ndarray): Training target data with shape (data_len, num_features).
            verbose (bool): If True, prints training progress every 10 epochs.

        Returns:
            list[float]: A list of average losses for each epoch.
        """

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)

        # DataLoader 생성
        dataloader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        losses = []
        # 학습 루프
        for epoch in range(self.config.epoch):
            epoch_loss = 0.0

            # Shape: (num_windows, seq_len, num_features)
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Smoothing Learning
                recon = model.smoothing(batch_x)
                loss = criterion(recon, batch_y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            losses.append(epoch_loss / len(dataloader))

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{self.config.epoch}], Loss: {losses[-1]:.5f}"
                )

        return losses

    @staticmethod
    def prepare_tensor(train_data: np.array, model: nn.Module) -> tuple:
        """
        Prepares input (X) and output (y) windows for training from the given data.

        Args:
            train_data (np.array): The time series data to prepare windows from.
            model (nn.Module): The model whose configuration is used to determine window sizes.

        Returns:
            tuple: A tuple containing two tensors, x_train and y_train, with shapes (batch_size, seq_len, num_features) and (batch_size, pred_len, num_features) respectively.
        """

        T = len(train_data)

        x, y = [], []

        config = model.config
        input_size = config.seq_len
        output_size = config.pred_len

        for t in range(input_size, T - output_size):
            # 각 윈도우 생성
            input_window = train_data[t - input_size : t]
            output_window = train_data[t : t + output_size]

            x.append(input_window)
            y.append(output_window)

        # (batch_size, seq_len, num_features)
        x_train = torch.tensor(np.array(x), dtype=torch.float32)
        # (batch_size, pred_len, num_features)
        y_train = torch.tensor(np.array(y), dtype=torch.float32)
        return x_train, y_train
