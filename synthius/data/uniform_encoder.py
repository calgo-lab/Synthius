from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniformDataEncoderError(Exception):
    """Custom exception class for UniformDataEncoder errors."""


class MissingFrequencyError(UniformDataEncoderError):
    """Exception raised for missing frequency of a category."""


class SuitableValuesNotFoundError(UniformDataEncoderError):
    """Exception raised when suitable values to replace NaNs are not found."""


class UniformDataEncoder:
    """Provides uniform encoding for both continuous and categorical data in a DataFrame.

    This class leverages statistical distributions to map original data into uniform
    intervals, ensuring that encoded data reflects the underlying patterns without exposing
    raw values. It supports handling NaN values and can work with both numerical and
    categorical data.

    Attributes:
        frequencies (dict[str, dict[Any, float]]): Tracks the normalized frequency of
            each unique value within each column.
        intervals (dict[str, dict[Any, list[float]]]): Records the intervals each value
            is mapped to during encoding.
        dtypes (dict[str, Any]): Saves the original data types of each column from the DataFrame.
        nan_value (str): Designated string to represent NaN values in encoding.

    Usage Example:
    ----------------------

    ## Encoding:

    ```python
    encoder = UniformDataEncoder()
    encoder.fit(/data/as/dataframe)
    transformed_data = encoder.transform(/data/as/dataframe)
    ```

    ## Decoding:

    ```python
    decoded_data = encoder.reverse_transform(transformed_data, nan_filling=True)
    ```
    """

    MAX_DISPLAY_CATEGORIES = 3

    def __init__(self: UniformDataEncoder) -> None:
        """Constructs a new instance of UniformDataEncoder with initial empty settings.

        This initialization sets up empty dictionaries for storing value frequencies,
        intervals, and data types after encoding. It also establishes a default string
        to represent NaN values across all operations.
        """
        self.frequencies: dict[str, dict[Any, float]] = {}
        self.intervals: dict[str, dict[Any, list[float]]] = {}
        self.dtypes: dict[str, Any] = {}
        self.nan_value: str = "NaN_Category"

    @staticmethod
    def _compute_frequencies_intervals(
        categories: list[Any],
        freq: np.ndarray,
    ) -> tuple[dict[Any, float], dict[Any, list[float]]]:
        """Computes the frequencies and intervals for given categories.

        This method is used internally to compute the statistical representation
        of data values as frequencies and interval ranges.

        Args:
            categories (list[Any]): List of unique data values.
            freq (np.ndarray): Array of frequencies corresponding to the categories.

        Returns:
            tuple[dict[Any, float], dict[Any, list[float]]]: A tuple containing
                two dictionaries, one for the frequencies and one for the intervals
                of the categories.
        """
        frequencies = dict(zip(categories, freq))
        shift = np.cumsum(np.hstack([0, freq]))
        shift[-1] = 1
        list_int = [[shift[i], shift[i + 1]] for i in range(len(shift) - 1)]
        intervals = dict(zip(categories, list_int))

        return frequencies, intervals

    def fit(self: UniformDataEncoder, data: pd.DataFrame, *, ordinal_number: bool = True) -> None:
        """Fits the encoder to the data by calculating value frequencies and intervals.

        This method prepares the encoder to transform new data based on the distribution
        learned from the provided DataFrame. It handles both numerical and categorical data.

        Args:
            data (pd.DataFrame): The DataFrame to fit the encoder.
            ordinal_number (bool, optional): Specifies if the numerical data should be
                treated as ordinal. Defaults to True.

        Raises:
            ValueError: If data contains NaN values that cannot be handled.
        """
        for column in data.columns:
            col_data = data[column]
            self.dtypes[column] = col_data.dtypes

            if ordinal_number and pd.api.types.is_numeric_dtype(col_data):
                labels = sorted(pd.unique(col_data.dropna()))
                if col_data.isna().any():
                    labels.append(self.nan_value)
                col_data = self.fill_nan_with_value(col_data, self.nan_value)
            else:
                col_data = self.fill_nan_with_value(col_data, self.nan_value)
                labels = pd.unique(col_data)

            if ordinal_number and pd.api.types.is_numeric_dtype(col_data):
                labels = sorted(pd.unique(col_data))

            freq = col_data.value_counts(normalize=True, dropna=False)
            nan_value_freq = freq.get(self.nan_value, 0)
            freq = freq.reindex(labels, fill_value=nan_value_freq).array

            self.frequencies[column], self.intervals[column] = self._compute_frequencies_intervals(labels, freq)

    def transform(self: UniformDataEncoder, data: pd.DataFrame) -> pd.DataFrame:
        """Transforms the data using the fitted model, assigning interval-based values.

        This method transforms the DataFrame to a new format where original data values
        are replaced by uniform random values derived from the intervals computed during
        fitting. It ensures that the transformed data maintains a similar statistical
        distribution.

        Args:
            data (pd.DataFrame): The data to transform using the encoder.

        Returns:
            pd.DataFrame: The transformed DataFrame with interval-based numeric representations.

        Raises:
            UserWarning: If the data contains categories that were not seen during fitting.
        """
        transformed_data = data.copy()
        rng = np.random.default_rng()
        for column in data.columns:
            col_data = transformed_data[column]
            col_data = self.fill_nan_with_value(col_data, self.nan_value)
            unseen_indexes = ~(col_data.isin(self.frequencies[column]))
            if unseen_indexes.any():
                unseen_categories = list(col_data.loc[unseen_indexes].unique())
                categories_to_print = self._get_message_unseen_categories(unseen_categories)
                warnings.warn(
                    f"The data in column '{column}' contains new categories "
                    f"that did not appear during 'fit' ({categories_to_print}). Assigning "
                    "them random values. If you want to model new categories, "
                    "please fit the data again.",
                    category=UserWarning,
                    stacklevel=2,
                )

                choices = list(self.frequencies[column].keys())
                size = unseen_indexes.sum()
                col_data.loc[unseen_indexes] = rng.choice(choices, size=size)

            transformed_data[column] = col_data.map(
                lambda label, col=column: self._map_labels(label, col, rng),
            ).astype(float)

        return transformed_data

    def _map_labels(self: UniformDataEncoder, label: str | float, column: str, rng: np.random.Generator) -> float:
        """Maps a label to its corresponding interval using a uniform distribution.

        This helper method is called during the transform process to assign each original
        data value a new value based on its predefined interval.

        Args:
            label (str | float): The label to map.
            column (str): The column from which the label originates.
            rng (np.random.Generator): An instance of a random number generator.

        Returns:
            float: A random float within the interval assigned to the label.
        """
        if label == self.nan_value:
            return rng.uniform(
                self.intervals[column][self.nan_value][0],
                self.intervals[column][self.nan_value][1],
            )
        return rng.uniform(self.intervals[column][label][0], self.intervals[column][label][1])

    def reverse_transform(self: UniformDataEncoder, data: pd.DataFrame, *, nan_filling: bool = False) -> pd.DataFrame:
        """Reverses the transformation, attempting to map interval-based values back to original categories.

        This method attempts to reconstruct the original data from the interval-based
        values generated during the `transform` process. It ensures that the output
        data maintains the original data types and handles NaN values if specified.

        Args:
        data (pd.DataFrame): The transformed data to reverse.
        nan_filling (bool, optional): Specifies if NaN intervals should be filled.
            Defaults to False.

        Returns:
        pd.DataFrame: The DataFrame with values mapped back to their original categories.
        """
        reversed_data = data.copy()
        for column in data.columns:
            col_data = reversed_data[column]
            self.check_nan_in_transform(col_data, self.dtypes[column])
            col_data = col_data.clip(0, 1)

            bins, labels = self._get_bins_and_labels(column)

            result = pd.cut(col_data, bins=bins, labels=labels, include_lowest=True)
            if self.nan_value in labels:
                result = result.astype(str).replace(self.nan_value, np.nan)

            if nan_filling:
                result = self._replace_nan_intervals(result, col_data, column)

            result = self._convert_dtype(result, column)
            reversed_data[column] = result.replace("nan", np.nan)

        return reversed_data

    def _get_bins_and_labels(self: UniformDataEncoder, column: str) -> tuple[list[float], list[str]]:
        """Gets bins and labels for a given column based on the computed intervals.

        This helper method generates the bin edges and corresponding labels for
        mapping interval-based values back to the original categories.

        Args:
        column (str): The column name to get bins and labels for.

        Returns:
        tuple[list[float], list[str]]: A tuple containing a list of bin edges and a list of labels.
        """
        bins = [0.0]
        labels = []
        for key, interval in self.intervals[column].items():
            bins.append(interval[1])
            labels.append(key)
        bins = sorted(set(bins))
        return bins, labels

    def _replace_nan_intervals(
        self: UniformDataEncoder,
        result: pd.Series,
        col_data: pd.Series,
        column: str,
    ) -> pd.Series:
        """Replaces NaN intervals in the result with suitable values based on frequencies.

        This method fills NaN values in the transformed data by selecting appropriate
        values based on the frequency distribution of the original data.

        Args:
        result (pd.Series): The result series with interval-based values.
        col_data (pd.Series): The original column data.
        column (str): The column name being processed.

        Returns:
        pd.Series: The series with NaN values replaced by suitable values.
        """
        nan_mask = result.isna()
        if nan_mask.any():
            value_intervals = self.intervals[column]
            value_frequencies = self.frequencies[column]

            nan_freq, _, half_range = self._get_nan_freq_and_ranges(column, value_intervals)

            sorted_values = sorted(value_frequencies.items(), key=lambda x: x[1])

            lower_value, upper_value = self._find_suitable_values(sorted_values, nan_freq)

            for i in range(len(result)):
                if nan_mask.iloc[i]:
                    if col_data.iloc[i] < half_range:
                        result.iloc[i] = lower_value
                    else:
                        result.iloc[i] = upper_value

        return result

    def _get_nan_freq_and_ranges(
        self: UniformDataEncoder,
        column: str,
        value_intervals: dict,
    ) -> tuple[float, float, float]:
        """Gets the frequency and ranges for NaN intervals in the given column.

        This method retrieves the frequency, range, and half-range of the NaN interval
        for a specified column, which are used to appropriately handle NaN values.

        Args:
        column (str): The column name being processed.
        value_intervals (dict): The dictionary of value intervals.

        Returns:
        tuple[float, float, float]: A tuple containing the frequency, range, and half-range of the NaN interval.

        Raises:
        MissingFrequencyError: If the NaN category is not found in the value frequencies.
        """
        nan_interval = value_intervals[self.nan_value]
        nan_range = nan_interval[1] - nan_interval[0]
        half_range = nan_interval[0] + nan_range / 2

        nan_freq = None
        for val, freq in self.frequencies[column].items():
            if val == self.nan_value:
                nan_freq = freq
                break

        if nan_freq is None:
            msg = "NaN_Category not found in the value frequencies."
            raise MissingFrequencyError(msg)

        return nan_freq, nan_range, half_range

    def _find_suitable_values(self: UniformDataEncoder, sorted_values: list, nan_freq: float) -> tuple[str, str]:
        lower_value, upper_value = None, None
        lower_candidates, upper_candidates, equal_freqs = self._categorize_values_by_freq(sorted_values, nan_freq)

        if equal_freqs:
            lower_value, upper_value = self._handle_equal_frequencies(equal_freqs, lower_candidates)
        else:
            if lower_candidates:
                lower_value = lower_candidates[-1][0]
            if upper_candidates:
                upper_value = upper_candidates[0][0]

        if lower_value is None or upper_value is None:
            msg = "Unable to find suitable values to replace NaNs"
            raise SuitableValuesNotFoundError(msg)

        return lower_value, upper_value

    def _categorize_values_by_freq(
        self: UniformDataEncoder,
        sorted_values: list,
        nan_freq: float,
    ) -> tuple[list, list, list]:
        """Finds suitable values to replace NaN intervals based on frequency.

        This method identifies two suitable values to replace NaN intervals,
        selecting them from values with frequencies close to the NaN frequency.

        Args:
        sorted_values (list): The list of values sorted by frequency.
        nan_freq (float): The frequency of the NaN interval.

        Returns:
        tuple[str, str]: A tuple containing the lower and upper values to replace NaNs.

        Raises:
        SuitableValuesNotFoundError: If suitable values cannot be found.
        """
        lower_candidates, upper_candidates, equal_freqs = [], [], []
        for val, freq in sorted_values:
            if val != self.nan_value:
                if freq < nan_freq:
                    lower_candidates.append((val, freq))
                elif freq > nan_freq:
                    upper_candidates.append((val, freq))
                else:
                    equal_freqs.append(val)
        return lower_candidates, upper_candidates, equal_freqs

    def _handle_equal_frequencies(
        self: UniformDataEncoder,
        equal_freqs: list,
        lower_candidates: list,
    ) -> tuple[str, str]:
        """Handles equal frequencies by selecting appropriate values for NaN replacement.

        This method selects the best values to replace NaN intervals when there are
        multiple values with frequencies equal to the NaN frequency.

        Args:
        equal_freqs (list): The list of values with equal frequencies.
        lower_candidates (list): The list of lower frequency candidates.

        Returns:
        tuple[str, str]: A tuple containing the lower and upper values for NaN replacement.
        """
        lower_value, upper_value = None, None
        if len(equal_freqs) > 1:
            lower_value = equal_freqs[0]
            upper_value = equal_freqs[1]
        else:
            upper_value = equal_freqs[0]
            if lower_candidates:
                lower_value = lower_candidates[-1][0]
        return str(lower_value), str(upper_value)

    def _convert_dtype(self: UniformDataEncoder, result: pd.Series, column: str) -> pd.Series:
        """Converts the result series to the original data type of the column.

        This method ensures that the data type of the result series matches the
        original data type of the column.

        Args:
        result (pd.Series): The result series to convert.
        column (str): The column name being processed.

        Returns:
        pd.Series: The converted result series.
        """
        if pd.api.types.is_numeric_dtype(self.dtypes[column]):
            result = pd.to_numeric(result, errors="coerce")
        else:
            result = self.try_convert_to_dtype(result, self.dtypes[column])
        return result

    def get_intervals(self: UniformDataEncoder) -> None:
        """Logs the intervals for each category in each column.

        This method is used to display the intervals calculated for each category
        in each column. It is mainly useful for debugging or analysis purposes.
        """
        for column, intervals in self.intervals.items():
            logger.info("Column: %s", column)
            for value, interval in intervals.items():
                logger.info("  Value: %s, Interval: %s", value, interval)

    def get_frequencies(self: UniformDataEncoder) -> None:
        """Logs the frequencies of each category in each column.

        This method is used to display the frequencies calculated for each category
        in each column. It is primarily intended for debugging or analysis purposes.
        """
        for column, frequencies in self.frequencies.items():
            logger.info("Column: %s", column)
            for value, frequency in frequencies.items():
                logger.info("  Value: %s, Frequency: %s", value, frequency)

    @staticmethod
    def fill_nan_with_value(data: pd.Series, value: str | float) -> pd.Series:
        """Replaces NaN values in the series with a specified value.

        This method is a utility that helps in handling missing values by replacing them
        with a specified value, which could be a placeholder string or a numeric value.

        Args:
            data (pd.Series): The series in which NaN values need to be replaced.
            value (str | float): The value to use as a replacement for NaNs.

        Returns:
            pd.Series: The series with NaNs replaced by the specified value.
        """
        return data.where(pd.notna(data), value)

    @staticmethod
    def check_nan_in_transform(data: pd.Series, dtype: str | type) -> None:
        """Checks for NaN values in the data and raises an error if found.

        This method ensures that the data passed to the transform function does not contain
        NaN values, which are not allowed during the transformation process.

        Args:
            data (pd.Series): The data series to check.
            dtype (str | type): The data type of the column from which the series is derived.

        Raises:
            ValueError: If NaN values are found in the data.
        """
        if pd.isna(data).any():
            error_message = f"Cannot transform data with NaNs. Column dtype: {dtype}"
            raise ValueError(error_message)

    @staticmethod
    def try_convert_to_dtype(data: pd.Series, dtype: str | type) -> pd.Series:
        """Tries to convert the series to a specified data type.

        This method attempts to convert the series to the specified data type. If the conversion
        fails due to a type mismatch or other issues, the original series is returned.

        Args:
            data (pd.Series): The series to convert.
            dtype (str | type): The target data type.

        Returns:
            pd.Series: The converted series, or the original series if conversion fails.
        """
        try:
            return data.astype(dtype)
        except ValueError:
            return data

    @classmethod
    def _get_message_unseen_categories(cls: type[UniformDataEncoder], unseen_categories: list[Any]) -> str:
        """Generates a message for categories unseen during the fitting process.

        This class method is used to generate a warning message that lists categories
        not seen during the fitting process, which are encountered in the transform method.

        Args:
            unseen_categories (list[Any]): The list of unseen categories.

        Returns:
            str: A formatted string listing up to the first three unseen categories,
                and the total number of such categories if there are more.
        """
        categories_to_print = ", ".join(str(x) for x in unseen_categories[: cls.MAX_DISPLAY_CATEGORIES])
        if len(unseen_categories) > cls.MAX_DISPLAY_CATEGORIES:
            categories_to_print = f"{categories_to_print}, +{len(unseen_categories) - cls.MAX_DISPLAY_CATEGORIES} more"

        return categories_to_print
