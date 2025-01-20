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
    reintroducing missing values based on original proportions while maintaining decimal precision.


    Attributes:
    -----------
        data (pd.DataFrame): The input data to be processed.
        id_column (str | None): The name of the ID column, if any.
        label_encoders (dict): Dictionary for label encoders for categorical columns.
        scalers (dict): Dictionary for scalers for numerical columns.
        imputers (dict): Dictionary for imputers for each column.
        col_types (pd.Series): Series representing the data types of columns.
        bool_cols (pd.Index): Index of boolean columns.
        int_cols (pd.Index): Index of integer columns.
        float_cols (pd.Index): Index of float columns.
        missing_value_proportions (pd.Series): Series representing the proportion of missing values per column.
        decimal_places (dict): Dictionary storing the number of decimal places for float columns.
        original_id_values (pd.Series | None): Series storing original ID column values, if applicable.
        id_column_index (int | None): Index position of the ID column in the original data.


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

    def __init__(self: DataImputationPreprocessor, data: pd.DataFrame, id_column: str | None = None) -> None:
        """Initialize the DataImputationPreprocessor with the given DataFrame.

        Args:
            data (pd.DataFrame): The input data to preprocess.
            id_column (str | None): The name of the ID column, if any.

        """
        self.data: pd.DataFrame = data
        self.id_column: str | None = id_column
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scalers: dict[str, QuantileTransformer] = {}
        self.imputers: dict[str, SimpleImputer] = {}
        self.col_types: pd.Series = data.dtypes
        self.bool_cols: pd.Index = data.select_dtypes(include="bool").columns
        self.int_cols: pd.Index = data.select_dtypes(include=["int64"]).columns
        self.float_cols: pd.Index = data.select_dtypes(include=["float64"]).columns
        self.missing_value_proportions: pd.Series = self.data.isna().mean()
        self.decimal_places: dict[str, int] = {}
        self.fill_values: dict[str, float] = {}
        self.original_id_values: pd.Series | None = None
        self.id_column_index: int | None = None

        self.random_generator = np.random.default_rng()

        if self.id_column:
            if self.id_column in self.data.columns:
                self.original_id_values = self.data[self.id_column].copy()
                self.id_column_index = self.data.columns.get_loc(self.id_column)
                self.data = self.data.drop(self.id_column, axis=1)
            else:
                msg = f"The ID column '{self.id_column}' does not exist in the dataset."
                raise ValueError(msg)

    def get_decimal_places(self: DataImputationPreprocessor, series: pd.Series) -> int:
        """Calculate the number of decimal places in a given series.

        Args:
            series (pd.Series): The series to check.

        Returns:
            int: The maximum number of decimal places in the series.
        """
        series = series.dropna().astype(str)
        decimals = series.apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
        return int(decimals.max())

    def fit_transform(self: DataImputationPreprocessor) -> pd.DataFrame:
        """Fit the preprocessor to the data and transform it by imputing missing values and scaling features.

        Returns:
            pd.DataFrame: The transformed DataFrame with missing values filled and features scaled between 0 and 1.
        """
        processed_data: pd.DataFrame = self.data.copy()

        # Calculate and store decimal places for float columns
        for col in self.float_cols:
            self.decimal_places[col] = self.get_decimal_places(processed_data[col])

        # Convert boolean columns to integers
        processed_data[self.bool_cols] = processed_data[self.bool_cols].astype(int)

        for col in processed_data.columns:
            if self.col_types[col] in ["float64", "int64", "bool"]:
                self.imputers[col] = SimpleImputer(strategy="median")
            else:
                self.imputers[col] = SimpleImputer(strategy="most_frequent")
            processed_data[col] = self.imputers[col].fit_transform(processed_data[[col]]).ravel()
            self.fill_values[col] = self.imputers[col].statistics_[0]

        for col in processed_data.columns:
            if col in self.float_cols or col in self.int_cols:
                self.scalers[col] = QuantileTransformer(output_distribution="uniform")
                processed_data[col] = self.scalers[col].fit_transform(processed_data[[col]])
            elif col not in self.bool_cols:
                self.label_encoders[col] = LabelEncoder()
                processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col])
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

        # Inverse transform numerical data
        for col in original_data.columns:
            if col in self.float_cols or col in self.int_cols:
                original_data[col] = self.scalers[col].inverse_transform(original_data[[col]])
                # Round to original decimal places for float columns
                if col in self.float_cols:
                    original_data[col] = original_data[col].round(self.decimal_places[col])
            elif col not in self.bool_cols:
                original_data[col] = (original_data[col] * (self.label_encoders[col].classes_.size - 1)).round().astype(int)
                original_data[col] = self.label_encoders[col].inverse_transform(original_data[col])

        # Convert boolean columns back to booleans
        original_data[self.bool_cols] = original_data[self.bool_cols].round().astype(bool)

        # Convert integer columns back to integers
        original_data[self.int_cols] = original_data[self.int_cols].round().astype(int)

        # Reintroduce missing values based on proportions
        for col in original_data.columns:
            missing_proportion = self.missing_value_proportions[col]
            if missing_proportion > 0:
                filled_value = self.fill_values[col]
                missing_mask = (original_data[col] == filled_value) & (self.random_generator.random(len(original_data)) < missing_proportion)
                original_data.loc[missing_mask, col] = np.nan

        # Reintroduce the original ID column if applicable
        if self.id_column_index is not None:
            rng = np.random.default_rng()
            random_ids = [f"ID-{rng.integers(10000000, 99999999)}" for _ in range(len(original_data))]
            original_data.insert(self.id_column_index, self.id_column, random_ids)

        return original_data
