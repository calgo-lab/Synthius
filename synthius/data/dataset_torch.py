import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    """A torch Dataset designed for TabDiff."""
    def __init__(self, data: pd.DataFrame, d_numerical: int, categories: list[int]) -> None:
        """Initialize the torch dataset.

        Args:
            data (pd.DataFrame): The data as dataframe.
            d_numerical (int): The number of numerical features in `data`.
            categories(list[int]): A list with the unique number of categories
                for every categorical features in `data`.
        """
        self.X = data.to_numpy()
        self.d_numerical = d_numerical
        self.categories = categories

    def __getitem__(self, index: int) -> np.ndarray:  # noqa: D105
        return self.X[index]

    def __len__(self) -> int:  # noqa: D105
        return self.X.shape[0]
