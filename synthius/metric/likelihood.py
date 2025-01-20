from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import (
    BNLikelihood,
    BNLogLikelihood,
    GMLogLikelihood,
)

from synthius.metric.utils import BaseMetric, apply_preprocessing, generate_metadata, load_data, preprocess_data

if TYPE_CHECKING:
    from typing import Any, Callable


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

    def get_metric_dispatch(self: LikelihoodMetrics) -> dict[str, Callable]:
        """Returns a dictionary mapping metric names to their corresponding computation methods."""
        return {
            "GM Log Likelihood": self.compute_gm_log_likelihood,
            "BN Likelihood": self.compute_bn_likelihood,
            "BN Log Likelihood": self.compute_bn_log_likelihood,
        }

    def evaluate_all_metrics_in_parallel(self: LikelihoodMetrics, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates all likelihood metrics for a synthetic dataset in parallel."""
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem
        metric_dispatch = self.get_metric_dispatch()

        results: dict[str, Any] = {"Model Name": model_name}

        with ProcessPoolExecutor() as executor:
            futures: dict[Future, str] = {executor.submit(metric_dispatch[metric], synthetic_data): metric for metric in metric_dispatch}

            for future in as_completed(futures):
                metric_name = futures[future]
                try:
                    results[metric_name] = future.result()
                    logger.info("%s for %s Done.", metric_name, model_name)
                except Exception:
                    logger.exception("Error computing %s for %s", metric_name, model_name)
                    results[metric_name] = float("nan")

        return results

    def evaluate_all_metrics_in_sequential(self: LikelihoodMetrics, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates all likelihood metrics for a synthetic dataset sequentially."""
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem
        metric_dispatch = self.get_metric_dispatch()

        results: dict[str, Any] = {"Model Name": model_name}

        for metric, metric_func in metric_dispatch.items():
            try:
                results[metric] = metric_func(synthetic_data)
                logger.info("%s for %s Done.", metric, model_name)
            except Exception:  # noqa: PERF203
                logger.exception("Error computing %s for %s", metric, model_name)
                results[metric] = float("nan")

        return results

    def evaluate_selected_metrics(self: LikelihoodMetrics, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates only selected likelihood metrics sequentially."""
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem
        metric_dispatch = self.get_metric_dispatch()

        results: dict[str, Any] = {"Model Name": model_name}
        selected_metrics = self.selected_metrics if self.selected_metrics is not None else []

        if not selected_metrics:
            logger.warning("No metrics selected for evaluation in Likelihood Metrics.")
            return results

        for metric in selected_metrics:
            if metric in metric_dispatch:
                try:
                    results[metric] = metric_dispatch[metric](synthetic_data)
                    logger.info("%s for %s Done.", metric, model_name)
                except Exception:
                    logger.exception("Error computing %s for %s", metric, model_name)
                    results[metric] = float("nan")
            else:
                logger.warning("Metric %s is not supported.", metric)

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
            "GM Log Likelihood",
            "BN Likelihood",
            "BN Log Likelihood",
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

    def _evaluate_sequential(self: LikelihoodMetrics) -> None:
        """Evaluates all synthetic datasets sequentially."""
        if self.selected_metrics is None:
            for path in self.synthetic_data_paths:
                result = self.evaluate_all_metrics_in_sequential(path)
                self.results.append(result)
        else:
            for path in self.synthetic_data_paths:
                result = self.evaluate_selected_metrics(path)
                self.results.append(result)

    def _evaluate_parallel(self: LikelihoodMetrics) -> None:
        """Evaluates all synthetic datasets in parallel."""
        if self.selected_metrics is None:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.evaluate_all_metrics_in_parallel, path): path for path in self.synthetic_data_paths}
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                    except Exception:
                        logger.exception("Error processing %s", path)
        else:
            logger.warning(
                "Parallel execution is disabled for selected metrics in LikelihoodMetrics. Running sequentially.",
            )

            for path in self.synthetic_data_paths:
                result = self.evaluate_selected_metrics(path)
                self.results.append(result)

    def evaluate_all(self: LikelihoodMetrics) -> None:
        """Evaluates all synthetic datasets and stores the results."""
        if self.want_parallel:
            self._evaluate_parallel()
        else:
            self._evaluate_sequential()

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: LikelihoodMetrics) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
