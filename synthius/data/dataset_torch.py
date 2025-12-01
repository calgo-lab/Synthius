import pandas as pd
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, X: pd.DataFrame, d_numerical: int, categories: list[int]):
        self.X = X.to_numpy()
        self.d_numerical = d_numerical
        self.categories = categories

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]
