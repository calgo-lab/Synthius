from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

if TYPE_CHECKING:
    import pandas as pd


class DataImputationPreprocessor:
    """Preprocess data by filling missing values and scaling features between 0 and 1.

    This class handles missing data by filling boolean or categorical
    columns with the mode and numeric columns with the median.
    It also provides the ability to reverse the imputation process,
    reintroducing missing values based on original proportions.

    Usage Example:
    ----------------------

    ## Encoding:

    ```python
    data_preprocessor = DataImputationPreprocessor('/data/as/dataframe')```

    ```python
    processed_data = data_preprocessor.fit_transform()
    ```

    ## Decoding:

    ```python
    decoded_data = data_preprocessor.inverse_transform(processed_data/to/inverse)
    ```
    """

    def __init__(self: DataImputationPreprocessor, data: pd.DataFrame) -> None:
        """Initialize the DataImputationPreprocessor with the given DataFrame.

        Args:
            data (pd.DataFrame): The input data to preprocess.
        """
        self.data: pd.DataFrame = data
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scalers: dict[str, QuantileTransformer] = {}
        self.imputers: dict[str, SimpleImputer] = {}
        self.col_types: pd.Series = data.dtypes
        self.bool_cols: pd.Index = data.select_dtypes(include="bool").columns
        self.int_cols: pd.Index = data.select_dtypes(include=["int64"]).columns
        self.float_cols: pd.Index = data.select_dtypes(include=["float64"]).columns
        self.missing_value_proportions: pd.Series = self.data.isna().mean()
        self.fill_values: dict[str, float] = {}

        self.random_generator = np.random.default_rng()

    def fit_transform(self: DataImputationPreprocessor) -> pd.DataFrame:
        """Fit the preprocessor to the data and transform it by imputing missing values and scaling features.

        Returns:
            pd.DataFrame: The transformed DataFrame with missing values filled and features scaled between 0 and 1.
        """
        processed_data: pd.DataFrame = self.data.copy()

        # Convert boolean columns to integers
        processed_data[self.bool_cols] = processed_data[self.bool_cols].astype(int)

        # Handling missing values
        for col in processed_data.columns:
            if self.col_types[col] in ["float64", "int64", "bool"]:
                self.imputers[col] = SimpleImputer(strategy="median")
            else:
                self.imputers[col] = SimpleImputer(strategy="most_frequent")
            processed_data[col] = self.imputers[col].fit_transform(processed_data[[col]]).ravel()
            self.fill_values[col] = self.imputers[col].statistics_[0]

        # Encoding categorical data and transforming numerical data
        for col in processed_data.columns:
            if col in self.float_cols or col in self.int_cols:
                self.scalers[col] = QuantileTransformer(output_distribution="uniform")
                processed_data[col] = self.scalers[col].fit_transform(processed_data[[col]])
            elif col not in self.bool_cols:
                self.label_encoders[col] = LabelEncoder()
                processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col])
                # Normalizing encoded categorical data
                processed_data[col] = processed_data[col] / processed_data[col].max()

        return processed_data

    def inverse_transform(self: DataImputationPreprocessor, processed_data: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformation by scaling back and reintroducing missing values.

        Args:
            processed_data (pd.DataFrame): The transformed DataFrame to inverse transform.

        Returns:
            pd.DataFrame: The original DataFrame with imputed values and reintroduced missing values.
        """
        original_data: pd.DataFrame = processed_data.copy()

        # Inverse transforming numerical data
        for col in original_data.columns:
            if col in self.float_cols or col in self.int_cols:
                original_data[col] = self.scalers[col].inverse_transform(original_data[[col]])
            elif col not in self.bool_cols:
                original_data[col] = (
                    (original_data[col] * (self.label_encoders[col].classes_.size - 1)).round().astype(int)
                )
                original_data[col] = self.label_encoders[col].inverse_transform(original_data[col])

        # Convert boolean columns back to booleans
        original_data[self.bool_cols] = original_data[self.bool_cols].round().astype(bool)

        # Convert integer columns back to integers
        original_data[self.int_cols] = original_data[self.int_cols].round().astype(int)

        # Reintroducing missing values based on proportions and fill values
        for col in original_data.columns:
            missing_proportion = self.missing_value_proportions[col]
            if missing_proportion > 0:
                filled_value = self.fill_values[col]
                missing_mask = (original_data[col] == filled_value) & (
                    self.random_generator.random(len(original_data)) < missing_proportion
                )
                original_data.loc[missing_mask, col] = np.nan

        return original_data
