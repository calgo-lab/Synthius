from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import (
    CategoricalCAP,
    CategoricalGeneralizedCAP,
    CategoricalKNN,
    CategoricalNB,
    CategoricalRF,
    CategoricalZeroCAP,
)

from synthius.metric.utils import BaseMetric, apply_preprocessing, generate_metadata, load_data, preprocess_data

logger = getLogger()


class PrivacyAgainstInference(BaseMetric):
    """A class to compute Privacy Against Inference for synthetic data compared to real data.

    Privacy Against Inference describes a set of metrics that calculate the risk of an attacker
    being able to infer real, sensitive values. We assume that an attacker already possess a
    few columns of real data; they will combine it with the synthetic data to make educated guesses.

    This class uses `CategoricalKNN`, `CategoricalNB`, `CategoricalRF`,
    `CategoricalCAP`, `CategoricalZeroCAP` and `CategoricalGeneralizedCAP` from SDMetrics:
    https://docs.sdv.dev/sdmetrics

    - `CategoricalKNN` Uses k-nearest neighbors to determine inference risk.
    - `CategoricalNB` Assesses inference risk using Naive Bayes algorithm.
    - `CategoricalRF` Evaluates inference risk using a random forest classifier.
    - `CategoricalCAP` Quantifies risk of Correct Attribution Probability (CAP) attacks.
    - `CategoricalZeroCAP` Measures privacy risk when the synthetic data's equivalence class is empty.
    - `CategoricalGeneralizedCAP` Considers nearest matches using hamming distance when no exact matches exist.

    ### Important Note:
    The `key_fields` and `sensitive_fields` must all be of the same type.

    Attributes:
        real_data_path (Path): The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
        key_fields: A list of key fields for the privacy metrics.
        sensitive_fields: A list of sensitive fields for the privacy metrics.
        results (List[dict]): A list to store the computed metrics results.
        real_data (pd.DataFrame): The loaded real dataset.
        metadata: Metadata generated from the real dataset.
        selected_metrics (list[str]): A list of metrics to evaluate. If None, all metrics are evaluated.
        want_parallel (bool): A boolean indicating whether to use parallel processing.
        display_result (bool): A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: PrivacyAgainstInference,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list,
        key_fields: list[str],
        sensitive_fields: list[str],
        metadata: dict | None = None,
        selected_metrics: list[str] | None = None,
        *,
        want_parallel: bool = False,
        display_result: bool = True,
    ) -> None:
        """Initializes the PrivacyAgainstInference with paths to the real and synthetic datasets.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths: A list of paths to the synthetic datasets.
            key_fields: A list of key fields for the privacy metrics.
            sensitive_fields: A list of sensitive fields for the privacy metrics.
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

        self.key_fields: list = key_fields
        self.sensitive_fields: list = sensitive_fields

        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)

        self.want_parallel = want_parallel
        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

        PrivacyAgainstInference.__name__ = "Privacy Against Inference"

        self.evaluate_all()

    def compute_categorical_knn(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalKNN metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalKNN score.
        """
        return CategoricalKNN.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def compute_categorical_nb(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalNB metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalNB score.
        """
        return CategoricalNB.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def compute_categorical_rf(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalRF metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalRF score.
        """
        return CategoricalRF.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def compute_categorical_cap(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalCAP metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalCAP score.
        """
        return CategoricalCAP.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def compute_categorical_zero_cap(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalZeroCAP metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalZeroCAP score.
        """
        return CategoricalZeroCAP.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def compute_categorical_generalized_cap(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalGeneralizedCAP metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalGeneralizedCAP score.
        """
        return CategoricalGeneralizedCAP.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def evaluate(self: PrivacyAgainstInference, synthetic_data_path: Path) -> pd.DataFrame:
        """Evaluates a synthetic dataset against the real dataset using advanced quality metrics.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            pd.DataFrame: Evaluation results for the model.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem

        results: dict[str, str | float] = {"Model Name": model_name}

        metric_dispatch = {
            "CategoricalKNN": self.compute_categorical_knn,
            "CategoricalNB": self.compute_categorical_nb,
            "CategoricalRF": self.compute_categorical_rf,
            "CategoricalCAP": self.compute_categorical_cap,
            "CategoricalZeroCAP": self.compute_categorical_zero_cap,
            "CategoricalGeneralizedCAP": self.compute_categorical_generalized_cap,
        }

        for metric in self.selected_metrics or metric_dispatch.keys():
            if metric in metric_dispatch:
                try:
                    results[metric] = metric_dispatch[metric](synthetic_data)
                except Exception as e:  # noqa: BLE001
                    logger.warning("Could not compute metric %s for model %s. Skipping.", metric, model_name)
                    logger.warning(e)
                logger.warning("%s for %s Done.", metric, model_name)
            else:
                logger.warning("Metric %s is not supported and will be skipped.", metric)

        logger.info("Privacy Against Inference for %s Done.", model_name)
        return results

    def pivot_results(self: PrivacyAgainstInference) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        try:
            df_results = pd.DataFrame(self.results)

            available_metrics = [
                "CategoricalKNN",
                "CategoricalNB",
                "CategoricalRF",
                "CategoricalCAP",
                "CategoricalZeroCAP",
                "CategoricalGeneralizedCAP",
            ]

            if self.selected_metrics:
                available_metrics = [metric for metric in available_metrics if metric in self.selected_metrics]

            df_melted = df_results.melt(
                id_vars=["Model Name"],
                value_vars=available_metrics,
                var_name="Metric",
                value_name="Value",
            )

            return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

        except Exception as e:
            logger.exception("Error while pivoting the DataFrame: %s", e)  # noqa: TRY401
            return pd.DataFrame()

    def evaluate_all(self: PrivacyAgainstInference) -> None:
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

    def display_results(self: PrivacyAgainstInference) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
