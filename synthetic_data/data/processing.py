from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
from sklearn.utils import resample

logger = getLogger()


class NanPlaceholderFiller:
    """A class to fill NaN values in a CSV file with a specified placeholder and save the cleaned data to a new file.

    Attributes:
        input_path (Path): The path to the input CSV file.
        output_path (Path): The path where the cleaned CSV file will be saved.
        placeholder (str): The placeholder value to replace NaN values in the data.
        data (Optional[pd.DataFrame]): The pandas DataFrame loaded from the input CSV file.

    Methods:
        load_data(): Loads data from the input CSV file.
        fill_nan_values(): Fills all NaN values in the DataFrame with the specified placeholder.
        check_and_save(): Checks for NaN values and saves the DataFrame to a CSV file if no NaN values are found.
    """

    def __init__(self: NanPlaceholderFiller, input_path: Path, output_path: Path, placeholder: str = "Missed") -> None:
        """Initializes the NanPlaceholderFiller with the specified input and output paths and placeholder.

        Args:
            input_path (Path): The path to the input CSV file.
            output_path (Path): The path where the cleaned CSV file will be saved.
            placeholder (str): The placeholder value to replace NaN values in the data.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.placeholder = placeholder
        self.data = self.load_data()

        if self.data is not None:
            self.fill_nan_values()
            self.check_and_save()

    def load_data(self: NanPlaceholderFiller) -> pd.DataFrame | None:
        """Loads data from the input CSV file.

        Returns:
            Optional[pd.DataFrame]: The loaded data as a pandas DataFrame, or None if loading fails.
        """
        try:
            data = pd.read_csv(self.input_path)
        except pd.errors.ParserError:
            logger.exception("Failed to load data due to parsing error.")
            return None
        except FileNotFoundError:
            logger.exception("Failed to load data because the file was not found.")
            return None
        else:
            logger.info("Data loaded successfully.")
            return data

    def fill_nan_values(self: NanPlaceholderFiller) -> None:
        """Fills all NaN values in the DataFrame with the specified placeholder."""
        if self.data is not None:
            self.data = self.data.fillna(self.placeholder)
            logger.info("NaN values filled with placeholder.")
        else:
            logger.warning("Attempted to fill NaN values, but data is None.")

    def check_and_save(self: NanPlaceholderFiller) -> None:
        """Checks for NaN values and saves the DataFrame to a CSV file if no NaN values are found."""
        if self.data is not None:
            nan_exists = self.data.isna().any().any()
            if not nan_exists:
                self.data.to_csv(self.output_path, index=False)
                logger.info("Data saved successfully to %s.", self.output_path)
            else:
                logger.info("There are still NaN values in the dataset. Please check your data.")
        else:
            logger.warning("Attempted to check for NaN values and save, but data is None.")


class DatasetSampler:
    """A class for creating balanced or proportionally sampled datasets from binary target data.

    This class is designed to work with binary targets only and does not support multiclass targets.

    Attributes:
        data_path (Path): Path to the CSV file containing the dataset.
        target (str): Name of the target column in the dataset.
        data_dir (Path): Directory containing the dataset file.
        data (pd.DataFrame): The dataset loaded into a pandas DataFrame.


    Usage Example:
    ----------------------

    ```python
    sampler = DatasetSampler(Path("/path/to/dataset.csv"), target="binary_target_column")

    # Create a balanced dataset
    sampler.balanced_dataset()

    # Create a proportionally sampled dataset with a specified total number of records
    sampler.proportional_dataset(total_records=1000)

    # Create a custom balanced dataset with an even total number of records
    sampler.custom_balanced_dataset(total_records=500)
    ```
    """

    def __init__(self: DatasetSampler, data_path: Path, target: str) -> None:
        """Initializes the DatasetSampler with a dataset and target information.

        Args:
            data_path (Path): The file path to the dataset.
            target (str): The target column name in the dataset.
        """
        self.data_path = data_path
        self.target = target
        self.data_dir = self.data_path.parent
        self.data = pd.read_csv(self.data_path, low_memory=False)

    def balanced_dataset(self: DatasetSampler) -> None:
        """Creates a balanced dataset by downsampling the majority class to match the minority class size.

        The resulting dataset is saved as "balanced.csv" in the same directory as the input dataset.
        """
        true_subset = self.data[self.data[self.target]]
        false_subset = self.data[~self.data[self.target]]

        false_downsampled = resample(
            false_subset,
            replace=False,
            n_samples=len(true_subset),
            random_state=123,
        )
        balanced_dataset = pd.concat([true_subset, false_downsampled])
        balanced_dataset = balanced_dataset.sample(frac=1, random_state=123).reset_index(drop=True)
        balanced_dataset.to_csv(self.data_dir / "balanced.csv", index=False)

    def proportional_dataset(self: DatasetSampler, total_records: int) -> None:
        """Creates a dataset with a target class distribution proportional to the original dataset.

        This method allows specifying the total number of records in the resulting dataset.

        Args:
            total_records (int): The total number of records desired in the resulting dataset.
        """
        true_subset = self.data[self.data[self.target]]
        false_subset = self.data[~self.data[self.target]]

        original_total = len(self.data)
        true_count_original = len(true_subset)
        true_sample_size = int((true_count_original / original_total) * total_records)
        false_sample_size = total_records - true_sample_size

        true_sample = resample(
            true_subset,
            replace=False,
            n_samples=true_sample_size,
            random_state=123,
        )

        false_sample = resample(
            false_subset,
            replace=False,
            n_samples=false_sample_size,
            random_state=123,
        )

        ratio_dataset = pd.concat([true_sample, false_sample])
        ratio_dataset = ratio_dataset.sample(frac=1, random_state=123).reset_index(drop=True)
        ratio_dataset.to_csv(self.data_dir / "proportional.csv", index=False)

    def custom_balanced_dataset(self: DatasetSampler, total_records: int) -> None:
        """Creates a balanced dataset with specified total number of records, equally divided between target samples.

        Args:
            total_records (int): The total number of records desired in the resulting dataset, must be an even number.
        """
        if total_records % 2 != 0:
            message = "Total records must be an even number for a custom balanced dataset."
            raise ValueError(message)

        half_records = total_records // 2

        true_subset = self.data[self.data[self.target]]
        false_subset = self.data[~self.data[self.target]]

        true_sample = resample(
            true_subset,
            replace=False,
            n_samples=half_records,
            random_state=123,
        )

        false_sample = resample(
            false_subset,
            replace=False,
            n_samples=half_records,
            random_state=123,
        )

        custom_balanced_dataset = pd.concat([true_sample, false_sample])
        custom_balanced_dataset = custom_balanced_dataset.sample(frac=1, random_state=123).reset_index(drop=True)
        custom_balanced_dataset.to_csv(self.data_dir / "proportional_balanced.csv", index=False)

    def multiple_proportional_datasets(self: DatasetSampler, n_datasets: int, dataset_size: int) -> None:
        """Creates multiple unique proportionally sampled datasets.

        Args:
            n_datasets (int): Number of unique proportional datasets to create.
            dataset_size (int): Number of samples in each proportional dataset.
        """
        if n_datasets * dataset_size > len(self.data):
            msg = "Not enough data to create the requested number of unique datasets with the specified size."
            raise ValueError(
                msg,
            )

        # Shuffle data to ensure randomness
        shuffled_data = self.data.sample(frac=1, random_state=123).reset_index(drop=True)

        all_datasets = []
        for i in range(n_datasets):
            sampled_dataset = pd.DataFrame()

            original_total = len(shuffled_data)
            true_subset = shuffled_data[shuffled_data[self.target]]
            false_subset = shuffled_data[~shuffled_data[self.target]]

            true_count_original = len(true_subset)
            true_sample_size = int((true_count_original / original_total) * dataset_size)
            false_sample_size = dataset_size - true_sample_size

            true_sample = resample(
                true_subset,
                replace=False,
                n_samples=true_sample_size,
                random_state=123 + i,
            )

            false_sample = resample(
                false_subset,
                replace=False,
                n_samples=false_sample_size,
                random_state=123 + i,
            )

            sampled_dataset = pd.concat([true_sample, false_sample])
            sampled_dataset = sampled_dataset.sample(frac=1, random_state=123).reset_index(drop=True)
            all_datasets.append(sampled_dataset)

            # Reset the index of sampled_dataset to ensure alignment with shuffled_data
            sampled_dataset_index = sampled_dataset.index

            # Drop sampled data points by matching their indices with shuffled_data
            shuffled_data = shuffled_data.drop(sampled_dataset_index).reset_index(drop=True)

            sampled_dataset.to_csv(self.data_dir / f"multi_proportional_{i+1}.csv", index=False)

        if not shuffled_data.empty:
            shuffled_data.to_csv(self.data_dir / "remaining_data.csv", index=False)
