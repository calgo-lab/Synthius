from __future__ import annotations

import functools
import logging
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import ContinuousKLDivergence, CSTest, DiscreteKLDivergence

from synthetic_data.metric.utils import BaseMetric, generate_metadata, load_data

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
pd.options.mode.copy_on_write = True


def handle_errors(log_message: str) -> Callable:
    """Decorator to handle errors in metric computation functions.

    Args:
        log_message (str): The message to log if an exception is raised.

    Returns:
        Callable: The decorated function.
    """

    def decorator_handle_errors(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper_handle_errors(*args: tuple[Any], **kwargs: dict[str, Any]) -> float:
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError) as e:
                logger.error(log_message, e)  # noqa: TRY400
                return float("nan")

        return wrapper_handle_errors

    return decorator_handle_errors


class AdvancedQualityMetrics(BaseMetric):
    """A class to compute advanced quality metrics for synthetic data compared to real data.

    This class uses DiscreteKLDivergence, ContinuousKLDivergence, and CSTest (Chi-squared test) from SDMetrics:
    https://docs.sdv.dev/sdmetrics

    - `DiscreteKLDivergence` measures the divergence in the distribution of discrete data types across column pairs
    in real and synthetic datasets. This metric is specifically designed for discrete data types and assesses the
    similarity in distribution across column pairs. It is applied to all relevant pairs of discrete columns in the
    dataset, and a higher value indicates better fidelity of the synthetic data.

    - `ContinuousKLDivergence` evaluates the divergence in distribution for continuous data in column pairs. It works
    by first binning the continuous values to turn them into categorical values and then computing the relative
    entropy (Kullback-Leibler Divergence).

    - `CSTest` assesses the similarity between real and synthetic columns in terms of their distribution shapes,
    specifically for discrete data. It uses the Chi-squared test to evaluate if the synthetic data comes from the
    same distribution as the real data, providing a p-value as the score.

    Attributes:
        real_data_path: The path to the real dataset.
        synthetic_data_paths: A list of paths to the synthetic datasets.
        results: A list to store the computed metrics results.
        real_data: The loaded real dataset.
        metadata: Metadata generated from the real dataset.
        display_result: A boolean indicating whether to display the results.
    """

    def __init__(
        self: AdvancedQualityMetrics,
        real_data_path: Path,
        synthetic_data_paths: list[Path],
        metadata: dict | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the AdvancedQualityMetrics with paths to the real and synthetic datasets.

        Args:
            real_data_path (Path): The file path to the real dataset.
            synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
            metadata (dict | None): Optional metadata for the real dataset.
            display_result (bool): Whether to display the results. The default is True.
        """
        self.real_data_path: Path = real_data_path
        self.synthetic_data_paths: list[Path] = synthetic_data_paths
        self.results: list[dict[str, str | float]] = []

        self.real_data: pd.DataFrame = load_data(real_data_path)
        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)

        self.display_result = display_result
        self.pivoted_results = None

        AdvancedQualityMetrics.__name__ = "Advanced Quality"

        self.evaluate_all()

    @handle_errors("Skipping 'Discrete KL Divergence' due to error: %s")
    def compute_discrete_kl_divergence(self: AdvancedQualityMetrics, synthetic_data: pd.DataFrame) -> float:
        """Computes the Discrete KL Divergence.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data for comparison.

        Returns:
            float: The computed Discrete KL Divergence or NaN if an error occurs.
        """
        return DiscreteKLDivergence.compute(self.real_data, synthetic_data, self.metadata)

    @handle_errors("Skipping 'Continuous KL Divergence' due to error: %s")
    def compute_continuous_kl_divergence(self: AdvancedQualityMetrics, synthetic_data: pd.DataFrame) -> float:
        """Compute the Continuous KL Divergence.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data for comparison.

        Returns:
            float: The computed Continuous KL Divergence or NaN if an error occurs.
        """
        return ContinuousKLDivergence.compute(self.real_data, synthetic_data, self.metadata)

    @handle_errors("Skipping 'CS Test' due to error: %s")
    def compute_cs_test(self: AdvancedQualityMetrics, synthetic_data: pd.DataFrame) -> float:
        """Compute the Chi-Squared Test.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data for comparison.

        Returns:
            float: The computed Chi-Squared Test or NaN if an error occurs.
        """
        return CSTest.compute(self.real_data, synthetic_data, self.metadata)

    def evaluate(self: AdvancedQualityMetrics, synthetic_data_path: Path) -> pd.DataFrame:
        """Evaluates a synthetic dataset against the real dataset using advanced quality metrics.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            pd.DataFrame: Evaluation results for the model.
        """
        synthetic_data = load_data(synthetic_data_path).copy()

        discrete_kl_divergence_score = self.compute_discrete_kl_divergence(synthetic_data)
        continuous_kl_divergence_score = self.compute_continuous_kl_divergence(synthetic_data)
        cs_test_score = self.compute_cs_test(synthetic_data)

        model_name = synthetic_data_path.stem

        logger.info("Advanced Quality for %s Done.", model_name)

        return {
            "Model Name": model_name,
            "Discrete KL Divergence": discrete_kl_divergence_score,
            "Continuous KL Divergence": continuous_kl_divergence_score,
            "CS Test": cs_test_score,
        }

    def pivot_results(self: AdvancedQualityMetrics) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(self.results)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=["Discrete KL Divergence", "Continuous KL Divergence", "CS Test"],
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def evaluate_all(self: AdvancedQualityMetrics) -> None:
        """Evaluates all synthetic datasets against the real dataset and displays the results.

        Evaluations are performed in parallel using multiple cores.
        """
        with ProcessPoolExecutor() as executor:
            # Create a dictionary to map futures to paths
            futures_to_paths: dict[Future, Path] = {
                executor.submit(self.evaluate, path): path for path in self.synthetic_data_paths
            }

            for future in as_completed(futures_to_paths):
                path = futures_to_paths[future]
                if future.exception():
                    logger.error("Error processing %s: %s", path.stem, future.exception())
                else:
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Unexpected error processing %s: %s", path.stem, exc)  # noqa: TRY400

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: AdvancedQualityMetrics) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
