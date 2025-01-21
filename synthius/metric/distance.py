from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from synthius.data import NumericalLabelEncoder

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.base import TransformerMixin


import pandas as pd
from IPython.display import display

from synthius.metric.utils import BaseMetric

logger = getLogger()

# flake8: noqa:N806, N803


class DistanceMetrics(BaseMetric):
    """Evaluates the distance metrics between real and synthetic datasets.

    This class computes various distance metrics to assess the similarity between real
    and synthetic datasets. It supports different scaling methods for data normalization,
    including `QuantileTransformer`, `StandardScaler`, and `MinMaxScaler`. The class evaluates
    synthetic datasets by calculating Distance to Closest Record (DCR) and Nearest Neighbor Distance Ratio (NNDR).

    This implementation is adapted from https://github.com/Team-TUD/CTAB-GAN/tree/main/model/eval

    Attributes:
        real_data_path: The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (List[Path]): List of paths to synthetic datasets.
        results (List[dict]): List to store evaluation results.
        selected_metrics: A list of metrics to evaluate. If None, all metrics are evaluated.
        scaler (TransformerMixin): Scaler instance for data normalization.
        encoder (NumericalLabelEncoder): Encoder for numerical data.
        encoded_real (np.ndarray): Encoded real dataset.
        real_data_scaled (np.ndarray): Scaled real dataset.
        display_result (bool): Flag to determine if results should be displayed.
    """

    def __init__(  # noqa: PLR0913
        self: DistanceMetrics,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list[Path],
        scaler_choice: str = "MinMaxScaler",
        id_column: str | None = None,
        selected_metrics: list[str] | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the DistanceMetrics class with paths to datasets and the choice of scaler.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths (List[Path]): A list of file paths to synthetic datasets.
            scaler_choice (str, optional): The choice of scaler for data normalization. Defaults to "MinMaxScaler".
            id_column (str | None): The name of the ID column to be dropped from the datasets.
            selected_metrics (list[str] | None): Optional list of metrics to evaluate. If None,
                                        all metrics are evaluated.
            display_result (bool): Whether to display the results after evaluation.
        """
        if id_column is None:
            logger.warning("No ID column selected; all columns will be used for analysis.")

        self.synthetic_data_paths = synthetic_data_paths

        self.results: list = []
        self.scaler = self.select_scaler(scaler_choice)

        self.encoder = NumericalLabelEncoder(real_data_path, id_column)
        self.encoded_real, _ = self.encoder.encode()
        self.real_data_scaled = self.scaler.fit_transform(self.encoded_real)

        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

        DistanceMetrics.__name__ = "Distance"

        self.evaluate_all()

    def select_scaler(self: DistanceMetrics, scaler_choice: str) -> TransformerMixin:
        """Selects and returns a scaler based on the provided choice.

        Args:
            scaler_choice (str): The scaler choice.

        Returns:
            TransformerMixin: An instance of a scaler from sklearn.preprocessing.
        """
        if scaler_choice == "QuantileTransformer":
            return QuantileTransformer()
        if scaler_choice == "StandardScaler":
            return StandardScaler()
        return MinMaxScaler()

    def remove_self_distances(self: DistanceMetrics, dist_matrix: np.ndarray) -> np.ndarray:
        """Removes self distances from the distance matrix.

        Args:
            dist_matrix (np.ndarray): The distance matrix.

        Returns:
            np.ndarray: The distance matrix with self distances removed.
        """
        return dist_matrix[~np.eye(dist_matrix.shape[0], dtype=bool)].reshape(dist_matrix.shape[0], -1)

    def remove_zero_distances(
        self: DistanceMetrics,
        dist_matrix: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Removes zero distances from the distance matrix and logs the number of removed points.

        Args:
            dist_matrix (np.ndarray): The distance matrix.

        Returns:
            tuple[np.ndarray, int]: The distance matrix with zero distances removed and the count of zero distances.
        """
        zero_distances = np.isclose(dist_matrix, 0)
        zero_count = int(np.sum(zero_distances))
        if zero_count > 0:
            dist_matrix = dist_matrix[~zero_distances.any(axis=1)]

        return dist_matrix, zero_count

    def chunked_pairwise_distances(
        self: DistanceMetrics,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        metric: str = "euclidean",
        chunk_size: int = 1000,
    ) -> np.ndarray:
        """Calculates pairwise distances in chunks to reduce memory usage.

        This function divides the input data into smaller chunks and calculates the pairwise
        distances between these chunks. This approach reduces peak memory usage compared to
        calculating the distances for the entire dataset at once.

        Args:
            X (np.ndarray): The first dataset for which pairwise distances are to be calculated.
            Y (np.ndarray, optional): The second dataset for which pairwise distances are to be calculated.
                                    If None, pairwise distances within X are calculated. Defaults to None.
            metric (str, optional): The distance metric to use. Defaults to 'euclidean'.
            chunk_size (int, optional): The size of each chunk. Defaults to 1000.

        Returns:
            np.ndarray: The complete distance matrix containing pairwise distances between all points in X and Y.
        """
        if Y is None:
            Y = X

        num_chunks_X = int(np.ceil(X.shape[0] / chunk_size))
        num_chunks_Y = int(np.ceil(Y.shape[0] / chunk_size))

        dist_matrix = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)

        for i in range(num_chunks_X):
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, X.shape[0])
            for j in range(num_chunks_Y):
                start_j = j * chunk_size
                end_j = min((j + 1) * chunk_size, Y.shape[0])
                dist_matrix[start_i:end_i, start_j:end_j] = pairwise_distances(
                    X[start_i:end_i],
                    Y[start_j:end_j],
                    metric=metric,
                )

        return dist_matrix

    def rr_distance(
        self: DistanceMetrics,
    ) -> tuple[float, float, float, float, int]:
        """Calculates and returns distance metrics for the real dataset.

        This function calculates the pairwise Euclidean distances between all records in the real dataset.
        It removes self-distances from the distance matrix and then computes various statistical measures:

        - Minimum Distance (DCR): The minimum distance of each real data point to its closest real data point.
        - Maximum Distance: The maximum distance of each real data point to its furthest real data point.
        - 5th Percentile of DCR: The 5th percentile of the minimum distances.
        - Nearest Neighbor Distance Ratio (NNDR): The ratio of the distance to the nearest neighbor to the distance to
                                                  the second nearest neighbor.
        - 5th Percentile of NNDR: The 5th percentile of the NNDR values.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - Mean of minimum distances between real data points.
                - Mean of maximum distances between real data points.
                - 5th percentile of DCR.
                - 5th percentile of NNDR.
                - Count of zero distances removed.

        """
        dist_rr = self.chunked_pairwise_distances(
            self.real_data_scaled,
            metric="euclidean",
        )

        dist_rr = self.remove_self_distances(dist_rr)
        dist_rr, zero_count = self.remove_zero_distances(dist_rr)

        min_dist_rr = np.min(dist_rr, axis=1)

        max_dist_rr = np.max(dist_rr, axis=1)

        fifth_perc_rr = np.percentile(min_dist_rr, 5)

        nndr_rr = np.sort(dist_rr, axis=1)[:, :2]

        nn_fifth_perc_rr = np.percentile(nndr_rr[:, 0] / nndr_rr[:, 1], 5)

        mean_min_dist_rr = np.mean(min_dist_rr)
        mean_max_dist_rr = np.mean(max_dist_rr)

        return mean_min_dist_rr, mean_max_dist_rr, fifth_perc_rr, nn_fifth_perc_rr, zero_count

    def ss_distance(
        self: DistanceMetrics,
        synthetic_data_scaled: np.ndarray,
    ) -> tuple[float, float, float, float, int]:
        """Calculates and returns distance metrics for the synthetic dataset.

        This function calculates the pairwise Euclidean distances between all records in the synthetic dataset.
        It removes self-distances from the distance matrix and then computes various statistical measures:

        - Minimum Distance (DCR): The minimum distance of each synthetic data point to its closest synthetic data point.
        - Maximum Distance: The maximum distance of each synthetic data point to its furthest synthetic data point.
        - 5th Percentile of DCR: The 5th percentile of the minimum distances.
        - Nearest Neighbor Distance Ratio (NNDR): The ratio of the distance to the nearest neighbor to the distance
                                                  to the second nearest neighbor.
        - 5th Percentile of NNDR: The 5th percentile of the NNDR values.

        Args:
            synthetic_data_scaled (np.ndarray): The scaled synthetic dataset.

        Returns:
            tuple[float, float, float, float]: A tuple containing:
                - Mean of minimum distances between synthetic data points.
                - Mean of maximum distances between synthetic data points.
                - 5th percentile of DCR.
                - 5th percentile of NNDR.
                - Count of zero distances removed.

        """
        dist_ss = self.chunked_pairwise_distances(
            synthetic_data_scaled,
            metric="euclidean",
        )

        dist_ss = self.remove_self_distances(dist_ss)
        dist_ss, zero_count = self.remove_zero_distances(dist_ss)

        min_dist_ss = np.min(dist_ss, axis=1)
        max_dist_ss = np.max(dist_ss, axis=1)

        fifth_perc_ss = np.percentile(min_dist_ss, 5)

        nndr_ss = np.sort(dist_ss, axis=1)[:, :2]

        nn_fifth_perc_ss = np.percentile(nndr_ss[:, 0] / nndr_ss[:, 1], 5)

        mean_min_dist_ss = np.mean(min_dist_ss)
        mean_max_dist_ss = np.mean(max_dist_ss)

        return mean_min_dist_ss, mean_max_dist_ss, fifth_perc_ss, nn_fifth_perc_ss, zero_count

    def rs_distance(
        self: DistanceMetrics,
        synthetic_data_scaled: np.ndarray,
    ) -> tuple[float, float, float, float, int]:
        """Calculates and returns distance metrics between real and synthetic datasets.

        This function calculates the pairwise Euclidean distances between all records in the real dataset
        and all records in the synthetic dataset. It then computes various statistical measures:

        - Minimum Distance (DCR): The minimum distance of each real data point to its closest synthetic data point.
        - Maximum Distance: The maximum distance of each real data point to its furthest synthetic data point.
        - 5th Percentile of DCR: The 5th percentile of the minimum distances.
        - Nearest Neighbor Distance Ratio (NNDR): The ratio of the distance to the nearest neighbor to the distance
                                                to the second nearest neighbor.
        - 5th Percentile NNDR: The 5th percentile of the NNDR values.

        Args:
            synthetic_data_scaled (np.ndarray): The scaled synthetic dataset.

        Returns:
            tuple[float, float, float, float, int]: A tuple containing:
                - Mean of minimum distances between real and synthetic data points.
                - Mean of maximum distances between real and synthetic data points.
                - 5th percentile of DCR.
                - 5th percentile of NNDR.
                - Count of zero distances removed.
        """
        dist_rs = self.chunked_pairwise_distances(
            self.real_data_scaled,
            synthetic_data_scaled,
            metric="euclidean",
        )

        dist_rs, zero_count = self.remove_zero_distances(dist_rs)

        min_dist_rs = np.min(dist_rs, axis=1)
        max_dist_rs = np.max(dist_rs, axis=1)

        fifth_perc_rs = np.percentile(min_dist_rs, 5)

        nndr_rs = np.sort(dist_rs, axis=1)[:, :2]

        nn_fifth_perc_rs = np.percentile(nndr_rs[:, 0] / nndr_rs[:, 1], 5)

        mean_min_dist_rs = np.mean(min_dist_rs)
        mean_max_dist_rs = np.mean(max_dist_rs)

        return mean_min_dist_rs, mean_max_dist_rs, fifth_perc_rs, nn_fifth_perc_rs, zero_count

    def evaluate(
        self: DistanceMetrics,
        synthetic_data_path: Path,
    ) -> pd.DataFrame:
        """Evaluates a synthetic dataset against the real dataset using various distances.

        For each synthetic dataset, the method encodes and scales synthetic data,
        calculates evaluation metrics, and appends the results to the results list.

        Args:
            synthetic_data_path (Path): The file path to the synthetic dataset.

        Returns:
            pd.DataFrame: A pd.DataFrame containing evaluation metrics for the synthetic dataset.
        """
        encoded_synthetic = self.encoder.encode_additional_data(synthetic_data_path).copy()
        synthetic_data_scaled = self.scaler.transform(encoded_synthetic)

        model_name = synthetic_data_path.stem
        temp_results: dict[str, str | float | int] = {"Model Name": model_name}

        rr_metrics = {
            "5th Percentile | DCR | RR": "fifth_perc_rr",
            "5th Percentile | NNDR | RR": "nn_fifth_perc_rr",
            "Mean | DCR | RR": "mean_min_dist_rr",
            "Removed DataPoint | RR": "zero_count_rr",
        }
        ss_metrics = {
            "5th Percentile | DCR | SS": "fifth_perc_ss",
            "5th Percentile | NNDR | SS": "nn_fifth_perc_ss",
            "Removed DataPoint | SS": "zero_count_ss",
        }
        rs_metrics = {
            "5th Percentile | DCR | R&S": "fifth_perc_rs",
            "5th Percentile | NNDR | R&S": "nn_fifth_perc_rs",
            "Mean | DCR | R&S": "mean_min_dist_rs",
            "Score": "score",
            "Removed DataPoint | R&S": "zero_count_rs",
        }

        selected_metrics = self.selected_metrics if self.selected_metrics is not None else []

        # Calculate RR Metrics
        if any(metric in selected_metrics for metric in rr_metrics) or not selected_metrics:
            mean_min_dist_rr, mean_max_dist_rr, fifth_perc_rr, nn_fifth_perc_rr, zero_count_rr = self.rr_distance()
            temp_results.update(
                {
                    "5th Percentile | DCR | RR": float(fifth_perc_rr),
                    "5th Percentile | NNDR | RR": float(nn_fifth_perc_rr),
                    "Mean | DCR | RR": float(mean_min_dist_rr),
                    "Removed DataPoint | RR": int(zero_count_rr),
                },
            )

        # Calculate SS Metrics
        if any(metric in selected_metrics for metric in ss_metrics) or not selected_metrics:
            _, __, fifth_perc_ss, nn_fifth_perc_ss, zero_count_ss = self.ss_distance(synthetic_data_scaled)
            temp_results.update(
                {
                    "5th Percentile | DCR | SS": float(fifth_perc_ss),
                    "5th Percentile | NNDR | SS": float(nn_fifth_perc_ss),
                    "Removed DataPoint | SS": int(zero_count_ss),
                },
            )

        # Calculate RS Metrics
        if any(metric in selected_metrics for metric in rs_metrics) or not selected_metrics:
            mean_min_dist_rs, ___, fifth_perc_rs, nn_fifth_perc_rs, zero_count_rs = self.rs_distance(
                synthetic_data_scaled,
            )
            score = mean_min_dist_rs / mean_max_dist_rr if mean_max_dist_rr != 0 else float("nan")
            temp_results.update(
                {
                    "5th Percentile | DCR | R&S": float(fifth_perc_rs),
                    "5th Percentile | NNDR | R&S": float(nn_fifth_perc_rs),
                    "Mean | DCR | R&S": float(mean_min_dist_rs),
                    "Score": float(score),
                    "Removed DataPoint | R&S": int(zero_count_rs),
                },
            )

        # Filter only explicitly selected metrics
        if selected_metrics:
            filtered_results = {
                "Model Name": model_name,
                **{metric: temp_results[metric] for metric in selected_metrics if metric in temp_results},
            }
        else:
            filtered_results = temp_results

        logger.info("Distance evaluation for %s completed.", model_name)
        self.results.append(filtered_results)
        return pd.DataFrame([filtered_results])

    def pivot_results(self: DistanceMetrics) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(self.results)

        available_metrics = [
            "5th Percentile | DCR | R&S",
            "5th Percentile | DCR | RR",
            "5th Percentile | DCR | SS",
            "5th Percentile | NNDR | R&S",
            "5th Percentile | NNDR | RR",
            "5th Percentile | NNDR | SS",
            "Mean | DCR | RR",
            "Mean | DCR | R&S",
            "Score",
            "Removed DataPoint | R&S",
            "Removed DataPoint | RR",
            "Removed DataPoint | SS",
        ]
        present_metrics = [metric for metric in available_metrics if metric in df_results.columns]

        if not present_metrics:
            msg = "No valid metrics found in the results. Check the selected metrics."
            raise ValueError(msg)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=present_metrics,
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def evaluate_all(self: DistanceMetrics) -> None:
        """Evaluates all synthetic datasets against the real dataset and stores the results."""
        for synthetic_data_path in self.synthetic_data_paths:
            self.evaluate(synthetic_data_path)

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: DistanceMetrics) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
