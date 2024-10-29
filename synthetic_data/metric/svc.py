from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from logging import getLogger
from typing import TYPE_CHECKING

import pandas as pd
from IPython.display import display
from sdmetrics.single_table import (
    SVCDetection,
)

if TYPE_CHECKING:
    from pathlib import Path
from synthetic_data.metric.utils import BaseMetric, generate_metadata, load_data

logger = getLogger()


class SVCEvaluator(BaseMetric):
    """A class to compute SVCDetection for synthetic data compared to real data.

    This class uses SVCDetection from SDMetrics:
    https://docs.sdv.dev/sdmetrics

    `SVCDetection` uses a Support Vector Classifier to distinguish between real and synthetic data.

    Attributes:
        real_data_path: The path to the real dataset.
        synthetic_data_paths: A list of paths to the synthetic datasets.
        results: A list to store the computed metrics results.
        real_data: The loaded real dataset.
        metadata: Metadata generated from the real dataset.
        display_result: Boolean indicating whether to display the results.
    """

    def __init__(
        self: SVCEvaluator,
        real_data_path: Path,
        synthetic_data_paths: list,
        metadata: dict | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the SVCEvaluator with paths to the real and synthetic datasets.

        Args:
            real_data_path: The path to the real dataset.
            synthetic_data_paths: A list of paths to the synthetic datasets.
            metadata (dict | None): Optional metadata for the real dataset.
            display_result (bool): Whether to display the results. The default is True.
        """
        self.real_data_path: Path = real_data_path
        self.synthetic_data_paths: list[Path] = synthetic_data_paths
        self.results: list = []

        self.real_data: pd.DataFrame = load_data(real_data_path)
        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)

        self.display_result = display_result
        self.pivoted_results = None

        SVCDetection.__name__ = "SVC Detection"

        self.evaluate_all()

    def compute_svc_detection(
        self: SVCEvaluator,
        synthetic_data: pd.DataFrame,
        model_name: str,
    ) -> float:
        """Computes the SVCDetection metric for synthetic data compared to real data.

        Args:
            synthetic_data: The synthetic dataset to evaluate.
            model_name: Name of the model.

        Returns:
            float: The computed SVCDetection metric score.
        """
        try:
            svc_detection_score = SVCDetection.compute(
                self.real_data,
                synthetic_data,
                self.metadata,
            )
            return 1 - svc_detection_score
        except (ValueError, TypeError) as e:
            logger.error("SVCDetection metric computation failed for %s: %s", model_name, e)  # noqa: TRY400
            return float("nan")

    def evaluate(self: SVCEvaluator, synthetic_data_path: Path) -> pd.DataFrame:
        """Evaluates a synthetic dataset against the real dataset using SVCDetection metric.

        Args:
            synthetic_data_path: The path to the synthetic dataset to evaluate.

        Returns:
            dict: A dictionary containing the model name and its SVCDetection score.
        """
        synthetic_data = load_data(synthetic_data_path).copy()

        model_name = synthetic_data_path.stem

        svc_detection_score = self.compute_svc_detection(synthetic_data, model_name)
        logger.info("SVC for %s Done.", model_name)

        return {
            "Model Name": model_name,
            "SVCDetection": svc_detection_score,
        }

    def evaluate_all(self: SVCEvaluator) -> None:
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

    def pivot_results(self: SVCEvaluator) -> pd.DataFrame:
        """Pivots the accumulated results to organize models as columns and metrics as rows.

        Returns:
            pd.DataFrame: A pivoted DataFrame of the evaluation results.
        """
        df_results = pd.DataFrame(self.results)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=["SVCDetection"],
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def display_results(self: SVCEvaluator) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
