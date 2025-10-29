# The API for the generation of synthetic datasets.
# 2025.10.29
import warnings
from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from sklearn.model_selection import train_test_split


def preprocess_data(
    original_data_filename: str,
    data_dir: str,
    target_colmn: str | int,
    random_seed: int | None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read, split, preprocess, and save the data

    Args:

    Returns:

    Raises:

    """
    # Read in data
    data_path = Path(data_dir)
    data = pd.read_csv(data_path / original_data_filename, low_memory=False)

    # ERROR CHECKING!

    # Train-Test Split
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=random_seed,
        stratify=data[target_colmn],
    )

    # Save TTS data
    train_data.to_csv(data_path / "train.csv", index=False)
    test_data.to_csv(data_path / "test.csv", index=False)

    return train_data, test_data


def generate(
    original_data_filename: str,
    data_dir: str,
    results_dir: str,
    target_column: str | int,
    id_column = None: str | int | None,
    models = None: List[str] | None,
    random_seed = None: int | None
    ) -> None:
    """Generate synthetic datasets given a source dataset.

    Args:
        
    Returns:

    Raises:

    """
    # Read in the data



    # TTS

    # Create metadata

    # Create OVA target

    # Run suite of models

