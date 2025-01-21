from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING

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

from synthius.metric.utils import BaseMetric, apply_preprocessing, generate_metadata, load_data, preprocess_data

if TYPE_CHECKING:
    from typing import Any, Callable


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

    def get_metric_dispatch(self: PrivacyAgainstInference) -> dict[str, Callable]:
        """Returns a dictionary mapping metric names to their corresponding computation methods.

        Returns:
            dict[str, Callable]: A dictionary where keys are metric names and values are methods to compute them.
        """
        return {
            "CategoricalNB": self.compute_categorical_nb,
            "CategoricalRF": self.compute_categorical_rf,
            "CategoricalCAP": self.compute_categorical_cap,
            "CategoricalZeroCAP": self.compute_categorical_zero_cap,
            "CategoricalGeneralizedCAP": self.compute_categorical_generalized_cap,
            "CategoricalSVM": self.compute_categorical_svm,
            "CategoricalEnsemble": self.compute_categorical_ensemble,
            "CategoricalKNN": self.compute_categorical_knn,
        }

    def evaluate_all_metrics_in_parallel(self: PrivacyAgainstInference, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates all privacy metrics for a synthetic dataset in parallel.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, Any]: A dictionary with the computed metrics for the synthetic dataset.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem

        results: dict[str, Any] = {
            "Model Name": model_name,
            "CategoricalNB": float("nan"),
            "CategoricalRF": float("nan"),
            "CategoricalCAP": float("nan"),
            "CategoricalZeroCAP": float("nan"),
            "CategoricalGeneralizedCAP": float("nan"),
            "CategoricalSVM": float("nan"),
            "CategoricalEnsemble": float("nan"),
            "CategoricalKNN": float("nan"),
        }

        metric_dispatch = self.get_metric_dispatch()

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

        try:
            results["CategoricalKNN"] = self.compute_categorical_knn(synthetic_data)
            logger.info("CategoricalKNN for %s Done.", model_name)
        except Exception:
            logger.exception("Error computing CategoricalKNN for %s", model_name)
            results["CategoricalKNN"] = float("nan")

        return results

    def evaluate_all_metrics_in_sequential(self: PrivacyAgainstInference, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates only selected privacy metrics sequentially for a synthetic dataset.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, Any]: A dictionary with the computed selected metrics for the synthetic dataset.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem

        metric_dispatch = self.get_metric_dispatch()

        results: dict[str, Any] = {"Model Name": model_name}

        for metric in metric_dispatch:
            try:
                results[metric] = metric_dispatch[metric](synthetic_data)
                logger.info("%s for %s Done.", metric, model_name)
            except Exception:  # noqa: PERF203
                logger.exception("Error computing %s for %s", metric, model_name)
                results[metric] = float("nan")

        return results

    def evaluate_selected_metrics(self: PrivacyAgainstInference, synthetic_data_path: Path) -> dict[str, Any]:
        """Evaluates only selected privacy metrics sequentially.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, Any]: A dictionary with the computed selected metrics for the synthetic dataset.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values).copy()
        model_name = synthetic_data_path.stem

        metric_dispatch = self.get_metric_dispatch()

        results: dict[str, Any] = {"Model Name": model_name}

        selected_metrics = self.selected_metrics if self.selected_metrics is not None else []

        if not selected_metrics:
            logger.warning("No metrics selected for evaluation In Privacy Against Inference.")
            return results

        for metric in selected_metrics:
            if metric in metric_dispatch:
                try:
                    results[metric] = metric_dispatch[metric](synthetic_data)
                    logger.info("%s for %s Done.", metric, model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error computing %s for %s: %s", metric, model_name, exc)  # noqa: TRY400
                    results[metric] = None
            else:
                logger.warning("Metric %s is not supported.", metric)

        self.results.append(results)
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
                "CategoricalSVM",
                "CategoricalEnsemble",
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

    def _submit_jobs(self: PrivacyAgainstInference, executor: ProcessPoolExecutor) -> dict[Future, Path]:
        """Submits jobs to the executor based on whether all or selected metrics are being evaluated."""
        if self.selected_metrics is None:
            return {executor.submit(self.evaluate_all_metrics_in_parallel, path): path for path in self.synthetic_data_paths}

        return {executor.submit(self.evaluate_selected_metrics, path): path for path in self.synthetic_data_paths}

    def _evaluate_parallel(self: PrivacyAgainstInference) -> None:
        """Evaluates all synthetic datasets in parallel."""
        with ProcessPoolExecutor() as executor:
            futures = self._submit_jobs(executor)
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except RuntimeError:
                    logger.exception("Evaluation failed for %s", path)
                except Exception:
                    logger.exception("An unexpected error occurred for %s", path)

    def _evaluate_sequential(self: PrivacyAgainstInference) -> None:
        """Evaluates all synthetic datasets sequentially."""
        if self.selected_metrics is None:
            for path in self.synthetic_data_paths:
                result = self.evaluate_all_metrics_in_sequential(path)
                self.results.append(result)
        else:
            for path in self.synthetic_data_paths:
                result = self.evaluate_selected_metrics(path)
                self.results.append(result)

    def evaluate_all(self: PrivacyAgainstInference) -> None:
        """Evaluates all synthetic datasets and stores the results."""
        if self.want_parallel:
            self._evaluate_parallel()
        else:
            self._evaluate_sequential()

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: PrivacyAgainstInference) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
