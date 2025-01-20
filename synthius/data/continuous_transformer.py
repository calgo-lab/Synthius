from __future__ import annotations

import numpy as np
import pandas as pd
from rdt.transformers.categorical import LabelEncoder
from sklearn.impute import SimpleImputer


class ContinuousDataTransformer:
    """A class to convert dataset values into continuous numerical values and handle missing data.

    This class is designed to prepare data for analysis or machine learning models by transforming
    continuous data with missing values and encoding categorical or integer data. It fills missing
    values using the median for continuous columns and applies label encoding with added noise for
    non-continuous columns. The class also tracks the precision of continuous data to ensure that
    synthetic data retains the same number of decimal places as the original data.

    Attributes:
        data (pd.DataFrame): The original data to be transformed.
        transformers (dict): Stores LabelEncoder objects for each non-continuous column.
        imputers (dict): Stores SimpleImputer objects for each continuous column.
        fill_values (dict): Stores fill values (medians) used for continuous columns.
        missing_value_proportions (dict): Stores proportions of missing values in continuous columns.
        decimal_places (dict): Stores the number of decimal places for each continuous column.
        transformed_data (pd.DataFrame): The transformed version of the original data.

    Usage Example:
    ----------------------

    ## Encoding:

    ```python
    data_preprocessor = ContinuousDataTransformer('/data/as/dataframe')```

    ```python
    processed_data = data_preprocessor.fit_transform()
    ```

    ## Decoding:

    ```python
    decoded_data = data_preprocessor.inverse_transform(processed_data/to/inverse)
    ```

    """

    def __init__(self: ContinuousDataTransformer, data: pd.DataFrame) -> None:
        """Initializes the ContinuousDataTransformer with a dataset.

        Args:
            data (pd.DataFrame): The input data to be transformed.
        """
        self.data = data
        self.transformers: dict = {}
        self.imputers: dict = {}
        self.fill_values: dict = {}
        self.missing_value_proportions: dict = {}
        self.decimal_places: dict = {}
        self.transformed_data: pd.DataFrame = None

    def is_continuous(self: ContinuousDataTransformer, series: pd.Series) -> bool:
        """Check if a series is continuous by examining if any non-zero decimals exist.

        Args:
            series (pd.Series): The series to be checked for continuity.

        Returns:
            bool: True if the series is continuous, otherwise False.
        """
        series = pd.to_numeric(series, errors="coerce").dropna()
        return series.apply(lambda x: x % 1 != 0).any()

    def get_decimal_places(self: ContinuousDataTransformer, series: pd.Series) -> int:
        """Determine the maximum number of decimal places in a continuous series.

        This method examines the string representation of each non-null value in the series,
        splits it at the decimal point, and calculates the length of the fractional part.
        It then returns the maximum length observed among all values, which indicates the
        precision of the original data.

        Args:
            series (pd.Series): The series whose decimal precision needs to be determined.

        Returns:
            int: The maximum number of decimal places found in the series.
        """
        series = series.dropna().astype(str)
        decimals = series.apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
        return int(decimals.max())

    def fit_transform(self: ContinuousDataTransformer) -> pd.DataFrame:
        """Fit the data with appropriate transformers and imputers and apply transformations.

        Continuous columns have missing values filled using the median. Non-continuous
        columns are transformed using label encoding with noise.

        Returns:
            pd.DataFrame: The transformed data with continuous values.
        """
        self.transformed_data = self.data.copy()

        for col in self.data.columns:
            if self.is_continuous(self.data[col]):
                missing_value_proportion = self.data[col].isna().mean()
                self.missing_value_proportions[col] = missing_value_proportion

                self.decimal_places[col] = self.get_decimal_places(self.data[col])

                imputer = SimpleImputer(strategy="median")
                self.transformed_data[col] = imputer.fit_transform(self.data[[col]]).ravel()
                self.fill_values[col] = imputer.statistics_[0]
                self.imputers[col] = imputer
            else:
                transformer = LabelEncoder(add_noise=True)
                self.transformers[col] = transformer
                self.transformed_data[col] = transformer.fit_transform(self.data[[col]], col)

        return self.transformed_data

    def inverse_transform(self: ContinuousDataTransformer, transformed_data: pd.DataFrame) -> pd.DataFrame:
        """Reverse the transformations applied to the transformed data.

        This method attempts to reintroduce the original missing values and revert
        label encoding to the original categorical values.

        Args:
            transformed_data (pd.DataFrame): The transformed data to be inverted.

        Returns:
            pd.DataFrame: The data reverted back to its original form.
        """
        original_transformed = transformed_data.copy()
        random_generator = np.random.default_rng()

        for col, filled_value in self.fill_values.items():
            original_missing_count = int(self.missing_value_proportions[col] * len(self.data))

            filled_indices = original_transformed.index[original_transformed[col] == filled_value].tolist()
            available_fill_count = len(filled_indices)

            if available_fill_count >= original_missing_count:
                reintroduce_indices = random_generator.choice(filled_indices, original_missing_count, replace=False)
            else:
                proportion_to_reintroduce = available_fill_count / original_missing_count
                reintroduce_count = int(proportion_to_reintroduce * available_fill_count)
                reintroduce_indices = random_generator.choice(filled_indices, reintroduce_count, replace=False)

            original_transformed.loc[reintroduce_indices, col] = np.nan

        for col, transformer in self.transformers.items():
            original_transformed[col] = transformer.reverse_transform(transformed_data[[col]])

        for col, decimals in self.decimal_places.items():
            original_transformed[col] = original_transformed[col].round(decimals)

        return original_transformed
