import torch
import pandas as pd
from torch.utils.data import Dataset


class SpreadDataset(Dataset):
    def __init__(
        self, spreads: pd.DataFrame, seq_len: int, pred_len: int, num_pairs: int
    ):
        self.spreads = spreads
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_pairs = num_pairs

    @property
    def samples_per_pair(self):
        return len(self.spreads) - self.seq_len - self.pred_len

    @property
    def total_samples(self):
        return self.samples_per_pair * self.num_pairs

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        pair_idx = idx % self.num_pairs
        time_idx = idx // self.num_pairs + self.seq_len

        inp_series = self.spreads.iloc[
            time_idx - self.seq_len : time_idx, pair_idx : pair_idx + 1
        ].values
        out_series = self.spreads.iloc[
            time_idx : time_idx + self.pred_len, pair_idx : pair_idx + 1
        ].values

        x_train = torch.tensor(inp_series, dtype=torch.float32)
        y_train = torch.tensor(out_series, dtype=torch.float32)

        return x_train, y_train
