from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from IPython.display import display

from synthius.metric.utils import BaseMetric

logger = logging.getLogger("anonymeter")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


class AnonymeterMetric(BaseMetric):
    """Base class for anonymeter metric classes."""

    def __init__(self):
        self.results = None
        self.selected_metrics = None
        self.pivoted_results = None
        self.want_parallel = False
        self.display_result = False
        self.synthetic_data_paths = []

    def evaluate(
            self: AnonymeterMetric,
            synthetic_data_path: Path,
    ) -> dict[str, str | float]:
        raise NotImplementedError("Evaluation not implemented")

    def pivot_results(self: AnonymeterMetric) -> pd.DataFrame:
        """Pivots the accumulated results to organize models as columns and metrics as rows.

        Returns:
            pd.DataFrame: A pivoted DataFrame of the evaluation results.
        """
        try:
            df_results = pd.DataFrame(self.results)

            all_numeric_metrics = [
                "Privacy Risk",
                "Main Attack Success Rate",
                "Main Attack Marginal Error ±",
                "Baseline Attack Success Rate",
                "Baseline Attack Error ±",
                "Control Attack Success Rate",
                "Control Attack Error ±",
            ]
            all_non_numeric_metrics = ["CI(95%)"]

            numeric_metrics = [metric for metric in all_numeric_metrics if metric in df_results.columns]
            non_numeric_metrics = [metric for metric in all_non_numeric_metrics if metric in df_results.columns]

            # If selected_metrics is specified, filter again
            if self.selected_metrics:
                numeric_metrics = [metric for metric in numeric_metrics if metric in self.selected_metrics]
                non_numeric_metrics = [metric for metric in non_numeric_metrics if metric in self.selected_metrics]

            # Handle numeric metrics
            if numeric_metrics:
                df_results[numeric_metrics] = df_results[numeric_metrics].apply(pd.to_numeric, errors="coerce")

                df_melted_numeric = df_results.melt(
                    id_vars=["Model Name"],
                    value_vars=numeric_metrics,
                    var_name="Metric",
                    value_name="Value",
                )

                pivoted_df_numeric = df_melted_numeric.pivot_table(
                    index="Metric",
                    columns="Model Name",
                    values="Value",
                    aggfunc="mean",  # Handle NaN gracefully
                )
            else:
                pivoted_df_numeric = pd.DataFrame()

            # Handle non-numeric metrics
            if non_numeric_metrics:
                df_melted_non_numeric = df_results.melt(
                    id_vars=["Model Name"],
                    value_vars=non_numeric_metrics,
                    var_name="Metric",
                    value_name="Value",
                )

                pivoted_df_non_numeric = df_melted_non_numeric.pivot_table(
                    index="Metric",
                    columns="Model Name",
                    values="Value",
                    aggfunc="first",  # First is okay for non-numeric
                )
            else:
                pivoted_df_non_numeric = pd.DataFrame()

            pivoted_df = pd.concat([pivoted_df_numeric, pivoted_df_non_numeric])

            desired_order = [
                "Privacy Risk",
                "CI(95%)",
                "Main Attack Success Rate",
                "Main Attack Marginal Error ±",
                "Baseline Attack Success Rate",
                "Baseline Attack Error ±",
                "Control Attack Success Rate",
                "Control Attack Error ±",
            ]
            selected_order = [metric for metric in desired_order if metric in pivoted_df.index]

            return pivoted_df.reindex(selected_order)

        except Exception as e:
            logger.exception("Error while pivoting the DataFrame: %s", e)  # noqa: TRY401
            return pd.DataFrame()

    def evaluate_all(self: AnonymeterMetric) -> None:
        """Evaluates all synthetic datasets in parallel and stores the results."""
        if self.want_parallel:
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(self.evaluate, path): path for path in self.synthetic_data_paths}
                for future in as_completed(futures):
                    path = futures[future]
                    model_name = path.stem

                    try:
                        result = future.result()
                        if result:
                            self.results.append(result)
                            logger.info("%s for %s Done.", self.__class__.__name__, model_name)

                    except RuntimeError as ex:
                        logger.exception("Evaluation failed for %s: %s", path, ex)  # noqa: TRY401
                    except Exception as ex:
                        logger.exception("An unexpected error occurred for %s: %s", path, ex)  # noqa: TRY401

        else:
            for path in self.synthetic_data_paths:
                try:
                    result = self.evaluate(path)
                    self.results.append(result)
                    logger.info("%s for %s Done.", self.__class__.__name__, path.stem)
                except Exception:  # noqa: PERF203
                    logger.exception("Evaluation failed for %s", path)

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: AnonymeterMetric) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")