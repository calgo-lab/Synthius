from __future__ import annotations

import json
import random
import string
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd

logger = getLogger()


class NumericalLabelEncoder:
    """Handles encoding and decoding of data for machine learning models.

    This class provides methods to encode categorical data into `numerical format`, decode numerical data
    back to categorical format, and manage metadata related to the encoding and decoding processes.

    Usage Example:
    ----------------------

    ## Encoding:

    ```python
    dataset_path = Path('path/to/your/dataset.csv')
    encoder = NumericalLabelEncoder(dataset_path)
    encoded_data, metadata = encoder.encode()
    metadata_path = Path('path/to/save/metadata.json')
    encoder.save_metadata_to_file(metadata_path)
    ```

    ## Decoding:

    ```python
    decoded_data = encoder.decode(encoded/data/as/dataframe)
    ```

    ## Decoding with loaded metadata:

    ```python
    encoder = NumericalLabelEncoder(dataset_path)
    encoder.load_metadata_from_file('path/to/saved/metadata.json')
    encoded_data_path = Path('path/to/encoded/data.csv')
    encoded_data = pd.read_csv(encoded_data_path)
    decoded_data = encoder.decode(encoded_data)
    ```
    """

    def __init__(self: NumericalLabelEncoder, data_path: Path | pd.DataFrame, id_column: str | None = None) -> None:
        """Initializes the NumericalLabelEncoder with a path to the dataset.

        Loads the dataset from the given path and initializes metadata to None.

        Args:
            data_path: (Path | pd.DataFrame): A Path object specifying the path to the dataset. or data as pd.DataFrame.
            id_column (str | None): The name of the ID column to be dropped from the datasets.
        """
        self.id_column = id_column
        self.data: pd.DataFrame = self.load_data(data_path)
        self.metadata: dict = {}

    def load_data(self: NumericalLabelEncoder, data_path: Path | pd.DataFrame) -> pd.DataFrame:
        """Loads the dataset from the specified path, checking for the ID column.

        Args:
            data_path (Path): The path to the dataset file.

        Returns:
            pd.DataFrame: The loaded dataset, with the ID column dropped if it exists.
        """
        if isinstance(data_path, Path):
            data = pd.read_csv(data_path, low_memory=False)
        elif isinstance(data_path, pd.DataFrame):
            data = data_path.copy()
        else:
            msg = "real_data_path must be either a pathlib.Path object pointing to a file or a pandas DataFrame."
            raise TypeError(
                msg,
            )

        if self.id_column:
            if self.id_column in data.columns:
                data = data.drop(columns=[self.id_column])
            else:
                logger.warning("The ID column %s does not exist in the dataset.", self.id_column)

        return data

    def encode(self: NumericalLabelEncoder) -> tuple[pd.DataFrame, dict]:
        """Encodes the dataset, updating metadata with encoding information.

        Returns:
            A tuple containing the encoded dataset and the metadata dictionary.
        """
        self.data, self.metadata = self._create_label_encoding(self.data)
        return self.data, self.metadata

    @staticmethod
    def _create_label_encoding(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Creates label encodings for categorical and boolean columns in the dataframe.

        Args:
            dataframe: The pandas DataFrame to encode.

        Returns:
            A tuple of the encoded dataframe and a dictionary containing encoding metadata.

        Raises:
            ValueError: If metadata has not been set.
        """
        encoding_metadata: dict[str, dict] = {}

        for column in dataframe.columns:
            if dataframe[column].dtype in ["object", "bool"]:
                unique_values = list(dataframe[column].dropna().unique())
                label_mapping = {value if not isinstance(value, np.bool_) else bool(value): i for i, value in enumerate(unique_values)}
                nan_label = len(unique_values)
                label_mapping[str(np.nan)] = nan_label
                encoding_metadata[column] = label_mapping
                dataframe[column] = dataframe[column].map(label_mapping).fillna(nan_label)
            elif dataframe[column].dtype in ["int64", "float64"]:
                if dataframe[column].isna().any():
                    nan_label = -1
                    dataframe[column] = dataframe[column].fillna(nan_label)
                    encoding_metadata[column] = {"nan_label": nan_label}
                else:
                    encoding_metadata[column] = {"nan_label": None}

        return dataframe, encoding_metadata

    def encode_additional_data(self: NumericalLabelEncoder, new_data_path: Path) -> pd.DataFrame:
        """Encodes additional data using existing metadata from a previous encoding.

        Args:
            new_data_path: A Path object specifying the path to the new dataset to encode.

        Returns:
            A pandas DataFrame containing the encoded additional data.
        """
        if self.metadata is None:
            message = "Metadata is not set. Please encode data first."
            raise ValueError(message)

        new_data = self.load_data(new_data_path)
        new_data_encoded, _ = self._encode_new_data(new_data, self.metadata)
        return new_data_encoded

    def _encode_new_data(
        self: NumericalLabelEncoder,
        new_dataframe: pd.DataFrame,
        encoding_metadata: dict,
    ) -> tuple[pd.DataFrame, dict]:
        """Encodes a new dataframe using existing encoding metadata.

        Args:
            new_dataframe: The new pandas DataFrame to encode.
            encoding_metadata: A dictionary containing existing encoding metadata.

        Returns:
            A tuple of the encoded new dataframe and the encoding metadata dictionary.
        """
        for column, mapping in encoding_metadata.items():
            if "nan_label" not in mapping:
                nan_label = mapping.get(str(np.nan))
                unique_new_values = [x for x in new_dataframe[column].unique() if pd.notna(x)]

                max_label = max(mapping.values(), default=-1)
                for value in unique_new_values:
                    if value not in mapping:
                        max_label += 1
                        mapping[value] = max_label
                new_dataframe[column] = new_dataframe[column].map(mapping).fillna(nan_label)
            else:
                nan_label = mapping["nan_label"]
                if nan_label is None:
                    nan_label = -1
                new_dataframe[column] = new_dataframe[column].fillna(nan_label)

        return new_dataframe, encoding_metadata

    def decode(self: NumericalLabelEncoder, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Decodes previously encoded data back to its original form using metadata.

        Args:
            encoded_data: The encoded pandas DataFrame to decode.

        Returns:
            A pandas DataFrame of the decoded data.

        Raises:
            ValueError: If metadata has not been set.
        """
        if not self.metadata:
            message = "Metadata is not set. Please encode data first."
            raise ValueError(message)
        return self._decode_data(encoded_data, self.metadata)

    @staticmethod
    def _decode_data(encoded_dataframe: pd.DataFrame, encoding_metadata: dict) -> pd.DataFrame:
        """Decodes an encoded dataframe back to its original form using provided metadata.

        Args:
            encoded_dataframe: The encoded pandas DataFrame to decode.
            encoding_metadata: A dictionary containing encoding metadata.

        Returns:
            A pandas DataFrame of the decoded data.
        """
        decoded_dataframe = encoded_dataframe.copy()
        for column, mapping in encoding_metadata.items():
            if "nan_label" not in mapping:
                inverted_mapping = {v: k for k, v in mapping.items() if k != str(np.nan)}
                decoded_dataframe[column] = decoded_dataframe[column].map(inverted_mapping)
                if str(np.nan) in mapping:
                    decoded_dataframe[column] = decoded_dataframe[column].replace(mapping[str(np.nan)], np.nan)
            else:
                nan_label = mapping["nan_label"]
                if nan_label is not None:
                    decoded_dataframe[column] = decoded_dataframe[column].apply(
                        lambda x, bound_nan_label=nan_label: np.nan if x == bound_nan_label else x,
                    )

            # Check for and correct -0.0 values
            if decoded_dataframe[column].dtype in ["float64", "float32"]:
                decoded_dataframe[column] = decoded_dataframe[column].apply(
                    lambda x: 0.0 if np.signbit(x) and x == 0.0 else x,  # noqa: PLR2004
                )
        return decoded_dataframe

    def save_metadata_to_file(self: NumericalLabelEncoder, filepath: Path) -> None:
        """Saves the encoding metadata to a file.

        Args:
            filepath: A string or path specifying the file path to save the metadata.
        """
        with filepath.open("w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=4)

    def load_metadata_from_file(self: NumericalLabelEncoder, filepath: Path) -> None:
        """Loads the encoding metadata from a file.

        This method updates the encoder's metadata attribute with the data loaded from the specified file.

        Args:
            filepath: The file path from which to load the metadata.
        """
        with filepath.open(encoding="utf-8") as file:
            self.metadata = json.load(file)
            # Ensure numerical keys are correctly converted back to their original types
            for column, mappings in self.metadata.items():
                if "nan_label" not in mappings:
                    self.metadata[column] = {float(k) if k.replace(".", "", 1).isdigit() else k: v for k, v in mappings.items()}


class CategoricalEncoder:
    """Encodes data values to words with dynamically adjusted length based on unique values and decodes them back.

    This class is designed to work with both categorical and numerical data, including handling missing values.
    It reads the data from a specified path, creates a unique encoding for each unique value in the dataset
    (excluding any specified label column), and provides functionalities to encode the dataset, decode it back to
    its original form, and save/load the encoding metadata to/from a file. The length of the encoding is dynamically
    adjusted based on the number of unique values in each column, optimizing the encoding size.

    Usage Example:
    ----------------------

    ## Encoding:

    ```python
    dataset_path = Path('path/to/your/dataset.csv')
    encoder = CategoricalEncoder(dataset_path)
    encoded_data, metadata = encoder.encode()
    metadata_path = Path('path/to/save/metadata.json')
    encoder.save_metadata_to_file(metadata_path)
    ```

    ## Decoding:

    ```python
    decoded_data = encoder.decode(encoded/data/as/dataframe)
    ```

    ## Decoding with loaded metadata:

    ```python
    encoder = CategoricalEncoder(dataset_path)
    encoder.load_metadata_from_file('path/to/saved/metadata.json')
    encoded_data_path = Path('path/to/encoded/data.csv')
    encoded_data = pd.read_csv(encoded_data_path)
    decoded_data = encoder.decode(encoded_data)
    ```
    """

    def __init__(self: CategoricalEncoder, data_source: Path | pd.DataFrame, label_column: str | None = None) -> None:
        """Initializes the CategoricalEncoder with and an optional label column to exclude.

        Args:
            data_source (Path | pd.DataFrame): A Path object specifying the path to the dataset file, or a pandas
                                               DataFrame containing the data.
            label_column (str | None): Optional; the name of the label column to exclude from encoding and decoding.
        """
        self.label_column = label_column
        self.metadata: dict = {}

        if isinstance(data_source, Path):
            self.data_path = data_source
            self.data = pd.read_csv(data_source, low_memory=False)
        else:
            self.data = data_source

    def _calculate_code_length(self: CategoricalEncoder, num_unique_values: int) -> int:
        """Calculates the minimum code length required to uniquely represent a given number of unique values.

        Args:
            num_unique_values (int): The number of unique values for which to calculate the code length.

        Returns:
            int: The minimum code length needed to uniquely represent all unique values.
        """
        alphabet_length = len(string.ascii_uppercase)
        code_length = 1
        while alphabet_length**code_length < num_unique_values:
            code_length += 1
        return code_length

    def _generate_word_code(self: CategoricalEncoder, existing_codes: set[str], code_length: int) -> str:
        """Generates a unique word code of a specified length not present in existing_codes.

        Args:
            existing_codes (set[str]): A set of codes that have already been generated to ensure uniqueness.
            code_length (int): The length of the code to generate.

        Returns:
            str: A unique string code of the specified length.
        """
        while True:
            code = "".join(random.choices(string.ascii_uppercase, k=code_length))  # noqa: S311
            if code not in existing_codes:
                return code

    def _create_word_encoding(self: CategoricalEncoder, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Creates word encodings for all columns in the dataframe, excluding the specified label column.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to encode.

        Returns:
            A tuple containing the encoded pandas DataFrame and a dictionary of encoding metadata.
        """
        encoding_metadata = {}
        columns_to_encode = dataframe.columns.difference([self.label_column]) if self.label_column else dataframe.columns

        for column in columns_to_encode:
            unique_values = dataframe[column].unique()
            num_unique_values = len(unique_values) + 1  # +1 for NaN
            code_length = self._calculate_code_length(num_unique_values)

            word_mapping = {}
            existing_codes: set = set()

            nan_code = self._generate_word_code(existing_codes, code_length)
            word_mapping["nan"] = nan_code
            existing_codes.add(nan_code)

            for value in unique_values:
                if pd.isna(value):
                    continue
                word_mapping[value] = self._generate_word_code(existing_codes, code_length)
                existing_codes.add(word_mapping[value])

            encoding_metadata[column] = word_mapping
            dataframe[column] = dataframe[column].apply(lambda x, wm=word_mapping: wm["nan"] if pd.isna(x) else wm[x])

        return dataframe, encoding_metadata

    def encode(self: CategoricalEncoder) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
        """Encodes the dataset to 3-letter word codes and updates metadata.

        Returns:
            A tuple of the encoded dataset pandas DataFrame and the metadata dictionary.
        """
        data_copy = self.data.copy()
        encoded_data, self.metadata = self._create_word_encoding(data_copy)
        return encoded_data, self.metadata

    def decode(self: CategoricalEncoder, encoded_data: pd.DataFrame) -> pd.DataFrame:
        """Decodes the dataset from 3-letter word codes back to original data values.

        Args:
            encoded_data (pd.DataFrame): The encoded pandas DataFrame to decode.

        Returns:
            A pandas DataFrame of the decoded data.
        """
        if not self.metadata:
            message = "Metadata is not set. Please encode data first."
            raise ValueError(message)

        decoded_data = encoded_data.copy()
        for column, word_mapping in self.metadata.items():
            inverted_mapping = {v: k for k, v in word_mapping.items()}
            # Convert 'nan' back to np.nan
            inverted_mapping[word_mapping["nan"]] = np.nan
            decoded_data[column] = decoded_data[column].map(inverted_mapping)
        return decoded_data

    def save_metadata_to_file(self: CategoricalEncoder, filepath: Path) -> None:
        """Saves the encoding metadata to a file ensuring all keys are converted to strings to be JSON serializable.

        Args:
            filepath (Path): Specifying the file path where to save the metadata.

        Returns:
            None
        """
        # Convert metadata keys and sub-keys to strings
        serializable_metadata = {}
        for column, mappings in self.metadata.items():
            serializable_metadata[column] = {str(key): value for key, value in mappings.items()}

        with Path.open(filepath, "w", encoding="utf-8") as file:
            json.dump(serializable_metadata, file, ensure_ascii=False, indent=4)

    def load_metadata_from_file(self: CategoricalEncoder, filepath: Path) -> None:
        """Loads the encoding metadata from a file and updates the encoder's metadata attribute.

        Args:
            filepath (Path): Specifying the file path from where to load the metadata.

        Returns:
            None
        """
        with Path.open(filepath, "r", encoding="utf-8") as file:
            loaded_metadata = json.load(file)
            # Convert numeric strings back to their original types where necessary
            self.metadata = {}
            for column, mappings in loaded_metadata.items():
                self.metadata[column] = {float(k) if k.replace(".", "", 1).isdigit() else k: v for k, v in mappings.items()}
