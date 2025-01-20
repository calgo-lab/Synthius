from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from logging import getLogger
from pathlib import Path

import pandas as pd
from IPython.display import display
from sdmetrics.reports.single_table import DiagnosticReport, QualityReport
from sdmetrics.single_table import (
    NewRowSynthesis,
)

from synthius.metric.utils import BaseMetric, generate_metadata, load_data

logger = getLogger()


class BasicQualityMetrics(BaseMetric):
    """A class to evaluate basic quality metrics of synthetic data against real data.

    This class uses Quality Report, Diagnostic Report, and NewRowSynthesis from SDMetrics:
    https://docs.sdv.dev/sdmetrics

    Main Metrics:
    - Quality Report: Evaluates how well synthetic data captures mathematical properties from real data.
                      This is also known as synthetic data fidelity.
      - Column Shapes: Describes the overall distribution of a column.
      - Column Pair Trends: Describes the trend between two columns, e.g., correlation.

    - Diagnostic Report: Checks for basic data validity and data structure issues.
      - Validity: Checks if each column in the data contains valid data.
      - Structure: Checks if each table has the same overall structure as the real data, including column names.

    - NewRowSynthesis: Measures whether each row in the synthetic data is novel or matches an original row in
                       the real data.

    Attributes:
        real_data_path: The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
        results (list[dict[str, str | float]]): A list to store evaluation results.
        real_data (pd.DataFrame): The loaded real dataset.
        metadata (dict): Metadata generated from the real dataset.
        selected_metrics: A list of metrics to evaluate. If None, all metrics are evaluated.
        want_parallel: A boolean indicating whether to use parallel processing.
        display_result: A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: BasicQualityMetrics,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list[Path],
        metadata: dict | None = None,
        selected_metrics: list[str] | None = None,
        *,
        want_parallel: bool = False,
        display_result: bool = True,
    ) -> None:
        """Initializes the BasicQualityMetrics with real and synthetic data paths.

        Args:
            real_data_path (Path): The file path to the real dataset.
            synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
            metadata (dict | None): Optional metadata for the real dataset.
            selected_metrics (list[str] | None): Optional list of metrics to evaluate. If None,
                                                 all metrics are evaluated
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
        self.results: list[dict[str, str | float]] = []

        self.metadata = metadata if metadata is not None else generate_metadata(self.real_data)

        self.want_parallel = want_parallel
        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

        BasicQualityMetrics.__name__ = "Basic Quality"

        self.evaluate_all()

    def evaluate_quality(self: BasicQualityMetrics, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """Evaluates the quality of synthetic data against real data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic dataset to evaluate.

        Returns:
            dict[str, float]: A dictionary containing the overall quality score and sub-metric scores.
        """
        quality_report = QualityReport()
        quality_report.generate(self.real_data, synthetic_data, self.metadata, verbose=False)
        quality_properties = quality_report.get_properties()
        return {
            "Overall Quality": quality_report.get_score(),
            "Column Shapes": self.get_score_from_properties(quality_properties, "Column Shapes"),
            "Column Pair Trends": self.get_score_from_properties(quality_properties, "Column Pair Trends"),
        }

    def evaluate_diagnostics(self: BasicQualityMetrics, synthetic_data: pd.DataFrame) -> dict[str, float]:
        """Evaluates the diagnostics of synthetic data against real data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic dataset to evaluate.

        Returns:
            dict[str, float]: A dictionary containing the overall diagnostic score and sub-metric scores.
        """
        diagnostic_report = DiagnosticReport()
        diagnostic_report.generate(self.real_data, synthetic_data, self.metadata, verbose=False)
        diagnostic_properties = diagnostic_report.get_properties()
        return {
            "Overall Diagnostic": diagnostic_report.get_score(),
            "Data Validity": self.get_score_from_properties(diagnostic_properties, "Data Validity"),
            "Data Structure": self.get_score_from_properties(diagnostic_properties, "Data Structure"),
        }

    @staticmethod
    def compute_chunk(chunk: pd.DataFrame, real_data: pd.DataFrame, metadata: dict, tolerance: float) -> float:
        """Calculates NewRowSynthesis in chunks to speed up the process."""
        return NewRowSynthesis.compute(
            real_data=real_data,
            synthetic_data=chunk,
            metadata=metadata,
            numerical_match_tolerance=tolerance,
        )

    def evaluate_new_row(self: BasicQualityMetrics, synthetic_data: pd.DataFrame) -> float:
        """Evaluates the novelty of rows in synthetic data against real data.

        Args:
            synthetic_data (pd.DataFrame): The synthetic dataset to evaluate.

        Returns:
            float: The score indicating the novelty of the synthetic rows.
        """
        """Evaluates the novelty of rows in synthetic data against real data in parallel."""
        num_workers = 10
        chunk_size = len(synthetic_data) // num_workers

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    self.compute_chunk,
                    synthetic_data.iloc[i : i + chunk_size],
                    self.real_data,
                    self.metadata,
                    0.01,
                )
                for i in range(0, len(synthetic_data), chunk_size)
            ]

        scores = [future.result() for future in futures]
        return sum(scores) / len(scores)

    def get_score_from_properties(self: BasicQualityMetrics, properties: pd.DataFrame, property_name: str) -> float:
        """Extracts the score for a given property from the properties DataFrame.

        Args:
            properties (pd.DataFrame): The DataFrame containing properties and scores.
            property_name (str): The name of the property to extract the score for.

        Returns:
            float: The score of the specified property.
        """
        return properties[properties["Property"] == property_name]["Score"].iloc[0]

    def evaluate(self: BasicQualityMetrics, synthetic_data_path: Path) -> dict[str, str | float]:
        """Evaluates a synthetic dataset against the real dataset and returns the results.

        Args:
            synthetic_data_path (Path): The file path to the synthetic dataset to evaluate.

        Returns:
            dict: Evaluation results for the model.
        """
        synthetic_data = load_data(synthetic_data_path).copy()

        model_name = synthetic_data_path.stem

        results: dict[str, str | float] = {"Model Name": model_name}

        metric_dispatch = {
            "Overall Quality": self.evaluate_quality,
            "Column Shapes": self.evaluate_quality,
            "Column Pair Trends": self.evaluate_quality,
            "Overall Diagnostic": self.evaluate_diagnostics,
            "Data Validity": self.evaluate_diagnostics,
            "Data Structure": self.evaluate_diagnostics,
            "New Row Synthesis": self.evaluate_new_row,
        }

        for metric in self.selected_metrics or metric_dispatch.keys():
            if metric in metric_dispatch:
                if metric in ["Overall Quality", "Column Shapes", "Column Pair Trends"]:
                    quality_scores = self.evaluate_quality(synthetic_data)
                    results.update({k: v for k, v in quality_scores.items() if k == metric})
                elif metric in ["Overall Diagnostic", "Data Validity", "Data Structure"]:
                    diagnostic_scores = self.evaluate_diagnostics(synthetic_data)
                    results.update({k: v for k, v in diagnostic_scores.items() if k == metric})
                elif metric == "New Row Synthesis":
                    results["New Row Synthesis"] = self.evaluate_new_row(synthetic_data)
            else:
                logger.warning("Metric %s is not supported and will be skipped.", metric)

        logger.info("Basic Quality for %s Done.", model_name)
        return results

    def evaluate_all(self: BasicQualityMetrics) -> None:
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

    def pivot_results(self: BasicQualityMetrics) -> pd.DataFrame:
        """Pivots the accumulated results to organize models as columns and metrics as rows.

        Returns:
            pd.DataFrame: A pivoted DataFrame of the evaluation results.
        """
        df_results = pd.DataFrame(self.results)

        available_metrics = [
            "Overall Quality",
            "Column Shapes",
            "Column Pair Trends",
            "Overall Diagnostic",
            "Data Validity",
            "Data Structure",
            "New Row Synthesis",
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

    def display_results(self: BasicQualityMetrics) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
