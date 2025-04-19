import mlflow
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from dataclasses import dataclass
from torch.utils.data import DataLoader
from pmcts.model.dataset import SpreadDataset


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
        spreads: pd.DataFrame,
        verbose: bool = True,
        mlflow_run: str = None,
    ) -> list[float]:
        """
        Trains the SpreadRegressor model using the provided training data.

        Args:
            model (nn.Module): The model to be trained.
            spreads (pd.DataFrame): Training input data with shape (data_len, num_features).
            verbose (bool): If True, prints training progress every 10 epochs.
            mlflow_run (str): The run ID of the MLflow run.

        Returns:
            list[float]: A list of average losses for each epoch.
        """

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr)

        # Set MLflow
        if mlflow_run:
            mlflow.set_experiment(self.config.experiment)
            mlflow.start_run(run_name=mlflow_run)
            mlflow.log_params(self.config.__dict__)

        # Create Spread Dataset
        dataset = SpreadDataset(
            spreads,
            model.config.seq_len,
            model.config.pred_len,
            spreads.shape[1],
        )
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        losses = []
        # Training Loop
        for epoch in range(self.config.epoch):
            epoch_loss = 0.0
            processed_samples = 0

            # Shape: (num_windows, seq_len, num_features)
            for batch_x, batch_y in dataloader:
                # Smoothing Learning
                recon = model(batch_x)
                loss = criterion(recon, batch_y)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                processed_samples += batch_x.size(0)
                processed_rate = 100 * int(processed_samples / len(dataset)) + 1

                if verbose and processed_rate % 10 == 0:
                    print(
                        f"Epoch: {epoch + 1}/{self.config.epoch}, "
                        f"Loss: {epoch_loss / processed_samples:.5f}, "
                        f"Progress: {processed_samples}/{len(dataset)}"
                    )

                    if mlflow_run:
                        mlflow.log_metrics(
                            {
                                "loss": loss.item(),
                            },
                            step=epoch,
                        )

                losses.append(epoch_loss / len(dataset))

        if mlflow_run:
            mlflow.end_run()

        return losses
