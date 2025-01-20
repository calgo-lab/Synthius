from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sdv.metadata import SingleTableMetadata

if TYPE_CHECKING:
    from pathlib import Path
import numpy as np
import pandas as pd


def load_data(data_path: Path) -> pd.DataFrame:
    """Loads data from a given path.

    Args:
        data_path (Path): The file path to load data from.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(data_path, low_memory=False)


def generate_metadata(data: pd.DataFrame) -> dict:
    """Generates metadata from a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to generate metadata for.

    Returns:
        dict: A dictionary representation of the metadata.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    return metadata.to_dict()


def clean_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the column names in the DataFrame by removing unwanted characters.

    Args:
        data (pd.DataFrame): DataFrame whose columns need cleaning.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    data.columns = data.columns.str.replace("[-./]", "", regex=True)
    return data


def preprocess_data(data: pd.DataFrame, *, need_clean_columns: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Preprocesses the DataFrame and handles missing values.

    Args:
        data (pd.DataFrame): DataFrame to preprocess.
        need_clean_columns (bool | None): Whether to clean the column name or not.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
        dict: Dictionary of fill values used for each column.
    """
    fill_values = {}
    if need_clean_columns:
        data = clean_columns(data)

    for column in data.columns:
        if data[column].dtype == "bool":
            fill_value = data[column].mode()[0]
        elif np.issubdtype(data[column].dtype, np.number):
            fill_value = data[column].median()
        else:
            fill_value = data[column].mode()[0]
        data[column] = data[column].fillna(fill_value).infer_objects()
        fill_values[column] = fill_value

    return data, fill_values


def apply_preprocessing(
    data_path: Path,
    fill_values: dict[str, Any],
    *,
    need_clean_columns: bool = False,
) -> pd.DataFrame:
    """Applies the preprocessing using known fill values.

    Args:
        data_path (Path): Path to the data file.
        fill_values (dict[str, Any]): Dictionary of fill values used for each column.
        need_clean_columns (bool | None): Whether to clean the column name or not.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data = load_data(data_path).copy()

    if need_clean_columns:
        data = clean_columns(data)

    for column in data.columns:
        if column in fill_values:
            data[column] = data[column].fillna(fill_values[column]).infer_objects()
    return data


def format_value(x: float) -> str:
    """Format the floating-point number to 5 decimal places or convert to integer if the number is whole.

    Args:
    x (float): The value to format.

    Returns:
    str: The formatted string of the number.
    """
    if isinstance(x, float) and not x.is_integer():
        return f"{x:.5f}"
    if isinstance(x, float) and x.is_integer():
        return f"{int(x)}"
    return str(x)
