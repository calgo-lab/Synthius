from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import (
    BNLikelihood,
    BNLogLikelihood,
    GMLogLikelihood,
)

from synthius.metric.utils import BaseMetric, apply_preprocessing, generate_metadata, load_data, preprocess_data

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
        real_data_path (Path): The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
        results (List[dict]): A list to store the computed metrics results.
        real_data (pd.DataFrame): The loaded real dataset.
        metadata: Metadata generated from the real dataset.
        selected_metrics (list[str]): A list of metrics to evaluate. If None, all metrics are evaluated.
        want_parallel (bool): A boolean indicating whether to use parallel processing.
        display_result (bool): A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: LikelihoodMetrics,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list[Path],
        metadata: dict | None = None,
        selected_metrics: list[str] | None = None,
        *,
        want_parallel: bool = False,
        display_result: bool = True,
    ) -> None:
        """Initializes the LikelihoodMetrics with paths to the real and synthetic datasets.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
            metadata (dict | None): Optional metadata for the real dataset.
            selected_metrics (list[str] | None): Optional list of metrics to evaluate. If None,
                                                 all metrics are evaluated.
            want_parallel (bool): Whether to use parallel processing. The default is False.
            display_result (bool): Whether to display the results. The default is True.
        """
        if isinstance(real_data_path, Path):
            self.real_data_path: Path = real_data_path
            self.real_data = load_data(real_data_path)
        elif isinstance(real_data_path, pd.DataFrame):
            self.real_data = real_data_path
        else:
            msg = "real_data_path must be either a pathlib.Path object pointing to a file or a pandas DataFrame."
            raise TypeError(
                msg,
            )

        self.synthetic_data_paths: list[Path] = synthetic_data_paths
        self.results: list[dict[str, Any]] = []

        self.real_data, self.fill_values = preprocess_data(self.real_data)
        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)

        self.want_parallel = want_parallel
        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

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

        results: dict[str, str | float] = {"Model Name": model_name}

        metric_dispatch = {
            "BN Likelihood": self.compute_bn_likelihood,
            "BN Log Likelihood": self.compute_bn_log_likelihood,
            "GM Log Likelihood": self.compute_gm_log_likelihood,
        }

        for metric in self.selected_metrics or metric_dispatch.keys():
            if metric in metric_dispatch:
                results[metric] = metric_dispatch[metric](synthetic_data)
            else:
                logger.warning("Metric %s is not supported and will be skipped.", metric)

        logger.info("Likelihood for %s Done.", model_name)
        return results

    def pivot_results(self: LikelihoodMetrics) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(self.results)

        available_metrics = [
            "BN Likelihood",
            "BN Log Likelihood",
            "GM Log Likelihood",
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

    def evaluate_all(self: LikelihoodMetrics) -> None:
        """Evaluates all synthetic datasets against the real dataset and displays the results.

        Evaluations are performed in parallel using multiple cores.
        """
        if self.want_parallel:
            with ProcessPoolExecutor() as executor:
                # Create a dictionary to map futures to paths
                futures_to_paths: dict[Future, Path] = {executor.submit(self.evaluate, path): path for path in self.synthetic_data_paths}

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

        else:
            for path in self.synthetic_data_paths:
                try:
                    result = self.evaluate(path)
                    self.results.append(result)
                except Exception:  # noqa: PERF203
                    logger.exception("Evaluation failed for %s", path)

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: LikelihoodMetrics) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
