from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from typing import TYPE_CHECKING, Any

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import (
    BNLikelihood,
    BNLogLikelihood,
    GMLogLikelihood,
)

if TYPE_CHECKING:
    from pathlib import Path

from synthetic_data.metric.utils import BaseMetric, apply_preprocessing, generate_metadata, load_data, preprocess_data

logger = getLogger()


class LikelihoodMetrics(BaseMetric):
    """A class to compute likelihood metrics for synthetic data compared to real data.

    This class uses BNLikelihood, BNLogLikelihood, and GMLikelihood from SDMetrics:
    https://docs.sdv.dev/sdmetrics
    -`BNLikelihood` uses a Bayesian Network to calculate the likelihood of the synthetic
    data belonging to the real data.

    -`BNLogLikelihood` uses log of Bayesian Network to calculate the likelihood of the synthetic
    data belonging to the real data.

    -`GMLogLikelihood` operates by fitting multiple GaussianMixture models to the real data.
    It then evaluates the likelihood of the synthetic data conforming to these models.

    Attributes:
        real_data_path: The path to the real dataset.
        synthetic_data_paths: A list of paths to the synthetic datasets.
        results: A list to store the computed metrics results.
        real_data: The loaded real dataset.
        metadata: Metadata generated from the real dataset.
        display_result: A boolean indicating whether to display the results.

    """

    def __init__(
        self: LikelihoodMetrics,
        real_data_path: Path,
        synthetic_data_paths: list[Path],
        metadata: dict | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the LikelihoodMetrics with paths to the real and synthetic datasets.

        Args:
            real_data_path (Path): The file path to the real dataset.
            synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
            metadata (dict | None): Optional metadata for the real dataset.
            display_result (bool): Whether to display the results. The default is True.
        """
        self.real_data_path: Path = real_data_path
        self.synthetic_data_paths: list[Path] = synthetic_data_paths
        self.results: list[dict[str, Any]] = []

        self.real_data: pd.DataFrame = load_data(real_data_path)
        self.real_data, self.fill_values = preprocess_data(self.real_data)
        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)

        self.display_result = display_result
        self.pivoted_results = None

        LikelihoodMetrics.__name__ = "Likelihood"

        self.evaluate_all()

    def compute_gm_log_likelihood(self: LikelihoodMetrics, synthetic_data: pd.DataFrame) -> float:
        """Compute the GMLogLikelihood.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data for comparison.

        Returns:
            float: The computed GMLogLikelihood.
        """
        return GMLogLikelihood.compute(self.real_data, synthetic_data, self.metadata)

    def compute_bn_likelihood(self: LikelihoodMetrics, synthetic_data: pd.DataFrame) -> float:
        """Compute the BNLikelihood.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data for comparison.

        Returns:
            float: The computed BNLikelihood.
        """
        return BNLikelihood.compute(self.real_data, synthetic_data, self.metadata)

    def compute_bn_log_likelihood(self: LikelihoodMetrics, synthetic_data: pd.DataFrame) -> float:
        """Compute the BNLogLikelihood.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data for comparison.

        Returns:
            float: The computed BNLogLikelihood.
        """
        return BNLogLikelihood.compute(self.real_data, synthetic_data, self.metadata)

    def evaluate(self: LikelihoodMetrics, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates a synthetic dataset against the real dataset using likelihood metrics.

        Args:
            synthetic_data_path: The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, Any]: Evaluation results for the model.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem

        with ProcessPoolExecutor() as executor:
            futures: dict[Future, str] = {
                executor.submit(self.compute_bn_likelihood, synthetic_data): "BN Likelihood",
                executor.submit(self.compute_bn_log_likelihood, synthetic_data): "BN Log Likelihood",
                executor.submit(self.compute_gm_log_likelihood, synthetic_data): "GM Log Likelihood",
            }

            results: dict[str, Any] = {
                "Model Name": model_name,
                "GM Log Likelihood": None,
                "BN Likelihood": None,
                "BN Log Likelihood": None,
            }

            for future in as_completed(futures):
                metric_name = futures[future]
                try:
                    results[metric_name] = future.result()
                    logger.info("%s for %s Done.", metric_name, model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error computing %s for %s: %s", metric_name, model_name, exc)  # noqa: TRY400
                    results[metric_name] = None

        return results

    def pivot_results(self: LikelihoodMetrics) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(self.results)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=["GM Log Likelihood", "BN Likelihood", "BN Log Likelihood"],
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def evaluate_all(self: LikelihoodMetrics) -> None:
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

    def display_results(self: LikelihoodMetrics) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
