from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from typing import TYPE_CHECKING, Any

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import (
    CategoricalCAP,
    CategoricalEnsemble,
    CategoricalGeneralizedCAP,
    CategoricalKNN,
    CategoricalNB,
    CategoricalRF,
    CategoricalSVM,
    CategoricalZeroCAP,
)

if TYPE_CHECKING:
    from pathlib import Path
from synthetic_data.metric.utils import BaseMetric, apply_preprocessing, generate_metadata, load_data, preprocess_data

logger = getLogger()


class PrivacyAgainstInference(BaseMetric):
    """A class to compute Privacy Against Inference for synthetic data compared to real data.

    Privacy Against Inference describes a set of metrics that calculate the risk of an attacker
    being able to infer real, sensitive values. We assume that an attacker already possess a
    few columns of real data; they will combine it with the synthetic data to make educated guesses.

    This class uses `CategoricalKNN`, `CategoricalNB`, `CategoricalRF`, `CategoricalEnsemble`,
    `CategoricalCAP`, `CategoricalZeroCAP` and `CategoricalGeneralizedCAP` from SDMetrics:
    https://docs.sdv.dev/sdmetrics

    - `CategoricalKNN` Uses k-nearest neighbors to determine inference risk.
    - `CategoricalNB` Assesses inference risk using Naive Bayes algorithm.
    - `CategoricalRF` Evaluates inference risk using a random forest classifier.
    - `CategoricalEnsemble` Uses an ensemble of classifiers to estimate inference risk.
    - `CategoricalCAP` Quantifies risk of Correct Attribution Probability (CAP) attacks.
    - `CategoricalZeroCAP` Measures privacy risk when the synthetic data's equivalence class is empty.
    - `CategoricalGeneralizedCAP` Considers nearest matches using hamming distance when no exact matches exist.

    ### Important Note:
    The `key_fields` and `sensitive_fields` must all be of the same type.

    Attributes:
        real_data_path: The path to the real dataset.
        synthetic_data_paths: A list of paths to the synthetic datasets.
        key_fields: A list of key fields for the privacy metrics.
        sensitive_fields: A list of sensitive fields for the privacy metrics.
        results: A list to store the computed metrics results.
        real_data: The loaded real dataset.
        metadata: Metadata generated from the real dataset.
        display_result: A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: PrivacyAgainstInference,
        real_data_path: Path,
        synthetic_data_paths: list,
        key_fields: list[str],
        sensitive_fields: list[str],
        metadata: dict | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the PrivacyAgainstInference with paths to the real and synthetic datasets.

        Args:
            real_data_path: The path to the real dataset.
            synthetic_data_paths: A list of paths to the synthetic datasets.
            key_fields: A list of key fields for the privacy metrics.
            sensitive_fields: A list of sensitive fields for the privacy metrics.
            metadata (dict | None): Optional metadata for the real dataset.
            display_result (bool): Whether to display the results. The default is True.
        """
        self.real_data_path: Path = real_data_path
        self.synthetic_data_paths: list[Path] = synthetic_data_paths
        self.results: list[dict[str, Any]] = []
        self.real_data: pd.DataFrame = load_data(real_data_path)

        self.real_data, self.fill_values = preprocess_data(self.real_data)

        self.key_fields: list = key_fields
        self.sensitive_fields: list = sensitive_fields

        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)
        self.display_result = display_result
        self.pivoted_results = None

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

    def compute_categorical_svm(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalSVM metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalSVM score.
        """
        return CategoricalSVM.compute(
            self.real_data,
            synthetic_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
        )

    def compute_categorical_ensemble(self: PrivacyAgainstInference, synthetic_data: pd.DataFrame) -> float:
        """Computes the CategoricalEnsemble metric for the given synthetic data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic data to evaluate.

        Returns:
            float: The computed CategoricalEnsemble score.
        """
        model_kwargs = {
            "attackers": [
                CategoricalCAP.MODEL,
                CategoricalZeroCAP.MODEL,
                CategoricalGeneralizedCAP.MODEL,
                CategoricalNB.MODEL,
                CategoricalKNN.MODEL,
                CategoricalRF.MODEL,
                CategoricalSVM.MODEL,
            ],
        }
        return CategoricalEnsemble.compute(
            self.real_data,
            synthetic_data,
            self.metadata,
            self.key_fields,
            self.sensitive_fields,
            model_kwargs=model_kwargs,
        )

    def evaluate(self: PrivacyAgainstInference, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates a synthetic dataset against the real dataset using privacy metrics.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            pd.DataFrame: A DataFrame with the computed metrics for the synthetic dataset.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem

        with ProcessPoolExecutor() as executor:
            futures: dict[Future, str] = {
                executor.submit(self.compute_categorical_nb, synthetic_data): "CategoricalNB",
                executor.submit(self.compute_categorical_rf, synthetic_data): "CategoricalRF",
                executor.submit(self.compute_categorical_cap, synthetic_data): "CategoricalCAP",
                executor.submit(self.compute_categorical_zero_cap, synthetic_data): "CategoricalZeroCAP",
                executor.submit(self.compute_categorical_generalized_cap, synthetic_data): "CategoricalGeneralizedCAP",
                executor.submit(self.compute_categorical_svm, synthetic_data): "CategoricalSVM",
                executor.submit(self.compute_categorical_ensemble, synthetic_data): "CategoricalEnsemble",
            }

            results: dict[str, Any] = {
                "Model Name": model_name,
                "CategoricalNB": None,
                "CategoricalRF": None,
                "CategoricalCAP": None,
                "CategoricalZeroCAP": None,
                "CategoricalGeneralizedCAP": None,
                "CategoricalSVM": None,
                "CategoricalEnsemble": None,
            }

            for future in as_completed(futures):
                metric_name = futures[future]
                try:
                    results[metric_name] = future.result()
                    logger.info("%s for %s Done.", metric_name, model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error computing %s for %s: %s", metric_name, model_name, exc)  # noqa: TRY400
                    results[metric_name] = None

        try:
            results["CategoricalKNN"] = self.compute_categorical_knn(synthetic_data)
            logger.info("CategoricalKNN for %s Done.", model_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error computing CategoricalKNN for %s: %s", model_name, exc)  # noqa: TRY400
            results["CategoricalKNN"] = None

        return results

    def pivot_results(self: PrivacyAgainstInference) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(self.results)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=[
                "CategoricalKNN",
                "CategoricalNB",
                "CategoricalRF",
                "CategoricalCAP",
                "CategoricalZeroCAP",
                "CategoricalGeneralizedCAP",
                "CategoricalSVM",
                "CategoricalEnsemble",
            ],
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def evaluate_all(self: PrivacyAgainstInference) -> None:
        """Evaluates all synthetic datasets in parallel and stores the results."""
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.evaluate, path): path for path in self.synthetic_data_paths}
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except RuntimeError as ex:
                    logger.exception("Evaluation failed for %s: %s", path, ex)  # noqa: TRY401
                except Exception as ex:
                    logger.exception("An unexpected error occurred for %s: %s", path, ex)  # noqa: TRY401

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: PrivacyAgainstInference) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
