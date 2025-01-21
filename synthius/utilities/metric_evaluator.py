from __future__ import annotations

import io
import logging
import pickle
import warnings
from functools import wraps
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from typing import Any, Callable

    from synthius.metric.utils import BaseMetric


from pathlib import Path
from typing import TypeVar

from synthius.metric import (
    AdvancedQualityMetrics,
    BasicQualityMetrics,
    DistanceMetrics,
    LikelihoodMetrics,
    LinkabilityMetric,
    PrivacyAgainstInference,
    PropensityScore,
    SinglingOutMetric,
)
from synthius.metric.utils import format_value, generate_metadata, load_data
from synthius.model import ModelLoader

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

R = TypeVar("R")

METRIC_CLASSES = {
    "AdvancedQualityMetrics": AdvancedQualityMetrics,
    "BasicQualityMetrics": BasicQualityMetrics,
    "DistanceMetrics": DistanceMetrics,
    "LikelihoodMetrics": LikelihoodMetrics,
    "LinkabilityMetric": LinkabilityMetric,
    "PrivacyAgainstInference": PrivacyAgainstInference,
    "PropensityScore": PropensityScore,
    "SinglingOutMetric": SinglingOutMetric,
}


def handle_errors(func: Callable[..., R]) -> Callable[..., R | None]:
    """Decorator to handle errors in metric functions. Logs an error message and skips the metric if exceptions occurs.

    Args:
        func (Callable[..., R]): The metric function to be wrapped by the decorator.

    Returns:
        Callable[..., Optional[R]]: The wrapped function that handles errors.
    """

    @wraps(func)
    def wrapper_handle_errors(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> R | None:
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            metric_name = func.__name__
            logging.error("%s skipped due to %s", metric_name, e)  # noqa: TRY400
            return None

    return wrapper_handle_errors


class MetricsAggregator:
    """Aggregates various metrics for synthetic data evaluation, comparing them against realdata.

    Attributes:
        real_data_path (Path): Path to the real dataset file.
        synthetic_data_paths (list[Path]): List of paths to synthetic dataset files.
        control_data (Optional[Path]): Path to the control dataset file, if applicable.
        key_fields (list[str]): List of key field names used in certain metrics.
        sensitive_fields (list[str]): List of sensitive field names considered in privacy metrics.
        distance_scaler (str): Choice of scaling method for distance metrics.
        singlingout_mode (str): Mode of operation for singling out metric.
        singlingout_n_attacks (int): Number of attack iterations for singling out metric.
        singlingout_n_cols (Optional[int]): Number of columns considered in singling out attacks.
        linkability_n_neighbors (int): Number of neighbors considered in linkability metric.
        linkability_n_attacks (Optional[int]): Number of attack iterations for linkability metric.
        linkability_aux_cols (list[list[str]]): Auxiliary columns for linkability evaluation.
        id_column (Optional[str]): Identifier column in the datasets.
        utility_test_path (Path): Path to the utility test dataset file.
        utility_models_path (Path): Path to the directory containing the utility models.
        label_column (str): The name of the target variable in the dataset.
        want_parallel (Optional[bool]): A flag indicating whether to run metrics in parallel.
        pos_label (bool | str): The label of the positive class. Default is True.
        need_split (bool): A flag indicating whether the dataset should be split into training and testing sets.
        all_results (DataFrame): Accumulated results from all metrics.
        metadata (dict): Metadata generated from the real dataset.

    Usage Example:
    ----------------------
    ```python
    metrics_result = MetricsAggregator(
        real_data_path=train_data,
        synthetic_data_paths=synthetic_data_paths,
        control_data=test_data,
        key_fields=key_fields,
        sensitive_fields=sensitive_fields,
        distance_scaler="MinMaxScaler",
        singlingout_mode="multivariate",
        singlingout_n_attacks=200,
        singlingout_n_cols=5,
        linkability_n_neighbors=50,
        linkability_n_attacks=None,
        linkability_aux_cols=aux_cols,
        id_column=ID,
        utility_test_path=test_data,
        utility_models_path=models_path,
        label_column=TARGET,
        pos_label=POS_LABEL,
        need_split=False,
        want_parallel=False,
    )
    ```

    ## For running metrics for models and original dataset:

    ```python
    metrics_result.run_all_with_original()
    display(metrics_result.all_results)
    ```

    ## For running metrics for just models:

    ```python
    metrics_result.run_metrics_for_models()
    display(metrics_result.all_results)
    ```

    ## For running metrics for just original dataset:

    ```python
    metrics_result.run_metrics_for_original()
    display(metrics_result.all_results)
    ```

    ## If you want to update the model's results with the original dataset's results, you can do the following:

    ```python
    metrics_result = MetricsAggregator.load_results(Path("res.pkl"))
    metrics_result.run_metrics_for_original()
    metrics_result.run_or_update_metric("Utility")
    display(metrics_result.all_results)
    """

    def __init__(  # noqa: PLR0913
        self: MetricsAggregator,
        real_data_path: Path,
        synthetic_data_paths: list[Path],
        control_data: Path | None,
        key_fields: list[str],
        sensitive_fields: list[str],
        distance_scaler: str,
        singlingout_mode: str,
        singlingout_n_attacks: int,
        singlingout_n_cols: int | None,
        linkability_n_neighbors: int,
        linkability_n_attacks: int | None,
        linkability_aux_cols: list[list[str]],
        id_column: str | None,
        utility_test_path: Path,
        utility_models_path: Path,
        label_column: str,
        *,
        want_parallel: bool | None = None,
        pos_label: bool | str = True,
        need_split: bool = True,
        load_data_now: bool = True,
    ) -> None:
        """Initializes the MetricsAggregator with dataset paths, fields, and configuration for metrics."""
        self.real_data_path = real_data_path
        self.synthetic_data_paths = synthetic_data_paths
        self.control_data = control_data

        self.key_fields = key_fields
        self.sensitive_fields = sensitive_fields

        self.distance_scaler = distance_scaler

        self.singlingout_mode = singlingout_mode
        self.singlingout_n_attacks = singlingout_n_attacks
        self.singlingout_n_cols = singlingout_n_cols

        self.linkability_n_neighbors = linkability_n_neighbors
        self.linkability_n_attacks = linkability_n_attacks
        self.linkability_aux_cols = linkability_aux_cols

        self.id_column = id_column
        self.all_results = pd.DataFrame()

        if load_data_now:
            self.metadata = generate_metadata(load_data(self.real_data_path))
        else:
            self.metadata = {}

        self.utility_test_path = utility_test_path
        self.utility_models_path = utility_models_path
        self.label_column = label_column

        if want_parallel is None:
            self.want_parallel = False
        else:
            self.want_parallel = want_parallel

        self.pos_label = pos_label
        self.need_split = need_split

        self.drop_original_for_utility = True

    def add_metrics(self: MetricsAggregator, metric_class: BaseMetric) -> None:
        """Adds results from a metric evaluation to the aggregated results.

        Args:
            metric_class (BaseMetric): An instance of a metric class that has executed its evaluation.
        """
        if hasattr(metric_class, "pivoted_results") and metric_class.pivoted_results is not None:
            df_results = metric_class.pivoted_results.copy()
            metric_type = type(metric_class).__name__

            # Add 'Metric Type' as a second level of the index
            df_results = df_results.copy()
            df_results["Metric Type"] = metric_type
            df_results = df_results.set_index("Metric Type", append=True)
            df_results = df_results.reorder_levels(["Metric Type", "Metric"], axis=0)

            if self.all_results.empty:
                self.all_results = df_results
            else:
                self.all_results = pd.concat([self.all_results, df_results], axis=0)

    def add_utility_metrics(self: MetricsAggregator, df_results: pd.DataFrame, metric_name: str) -> None:
        """Adds utility metric results to the aggregated results.

        Args:
            df_results (DataFrame): DataFrame containing the results of the utility metric.
            metric_name (str): Name of the metric to label the DataFrame.
        """
        if self.drop_original_for_utility and "Original" in df_results.columns:
            df_results = df_results.drop(columns=["Original"])

        df_results["Metric Type"] = metric_name
        df_results = df_results.set_index("Metric Type", append=True)
        df_results = df_results.reorder_levels(["Metric Type", "Metric"], axis=0)

        if self.all_results.empty:
            self.all_results = df_results
        else:
            self.all_results = pd.concat([self.all_results, df_results], axis=0)

    @handle_errors
    def run_basic_quality_metrics(self: MetricsAggregator) -> None:
        """Executes the Basic Quality Metrics, adding their results to the aggregated output."""
        basic_quality_metrics = BasicQualityMetrics(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            metadata=self.metadata,
            want_parallel=self.want_parallel,
            display_result=False,
        )
        self.add_metrics(basic_quality_metrics)

    @handle_errors
    def run_advanced_quality_metrics(self: MetricsAggregator) -> None:
        """Executes the Advanced Quality Metrics, adding their results to the aggregated output."""
        advanced_quality_metrics = AdvancedQualityMetrics(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            metadata=self.metadata,
            want_parallel=self.want_parallel,
            display_result=False,
        )
        self.add_metrics(advanced_quality_metrics)

    @handle_errors
    def run_likelihood_metrics(self: MetricsAggregator) -> None:
        """Executes the Likelihood Metrics, adding their results to the aggregated output."""
        likelihood_metrics = LikelihoodMetrics(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            metadata=self.metadata,
            want_parallel=self.want_parallel,
            display_result=False,
        )
        self.add_metrics(likelihood_metrics)

    @handle_errors
    def run_privacy_against_inference(self: MetricsAggregator) -> None:
        """Executes the Privacy Against Inference Metric, adding results to the aggregated output."""
        privacy_against_inference = PrivacyAgainstInference(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
            metadata=self.metadata,
            want_parallel=self.want_parallel,
            display_result=False,
        )
        self.add_metrics(privacy_against_inference)

    @handle_errors
    def run_propensity_score(self: MetricsAggregator) -> None:
        """Executes the Propensity Score Metric, adding its results to the aggregated output."""
        propensity_score = PropensityScore(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            id_column=self.id_column,
            display_result=False,
        )
        self.add_metrics(propensity_score)

    @handle_errors
    def run_distance_metrics(self: MetricsAggregator) -> None:
        """Executes the Distance Metrics, adding their results to the aggregated output."""
        distance_metrics = DistanceMetrics(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            scaler_choice=self.distance_scaler,
            id_column=self.id_column,
            display_result=False,
        )
        self.add_metrics(distance_metrics)

    @handle_errors
    def run_singling_out_metric(self: MetricsAggregator) -> None:
        """Executes the Singling Out Metric, adding its results to the aggregated output."""
        singling_out_metric = SinglingOutMetric(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            control_data_path=self.control_data,
            mode=self.singlingout_mode,
            n_attacks=self.singlingout_n_attacks,
            n_cols=self.singlingout_n_cols,
            want_parallel=self.want_parallel,
            display_result=False,
        )
        self.add_metrics(singling_out_metric)

    @handle_errors
    def run_linkability_metric(self: MetricsAggregator) -> None:
        """Executes the Linkability Metric, adding its results to the aggregated output."""
        linkability_metric = LinkabilityMetric(
            real_data_path=self.real_data_path,
            synthetic_data_paths=self.synthetic_data_paths,
            control_data_path=self.control_data,
            aux_cols=self.linkability_aux_cols,
            n_neighbors=self.linkability_n_neighbors,
            n_attacks=self.linkability_n_attacks,
            want_parallel=self.want_parallel,
            display_result=False,
        )
        self.add_metrics(linkability_metric)

    def run_utility_metric(self: MetricsAggregator) -> None:
        """Runs the utility metric evaluation."""
        ModelLoader.results_list.clear()
        experiment_names = [path.stem for path in self.synthetic_data_paths] + ["Original"]

        for exp in experiment_names:
            ModelLoader(
                data_path=self.utility_test_path,
                label_column=self.label_column,
                experiment_name=exp,
                models_base_path=self.utility_models_path / exp,
                pos_label=self.pos_label,
                need_split=self.need_split,
            )

        fig = ModelLoader.plot_metrics(pos_label=self.pos_label)

        self.saved_plot = fig

        result = ModelLoader.pivoted_results

        metric_name = "Utility"

        if not self.all_results.empty and metric_name in self.all_results.index.get_level_values("Metric Type").unique():
            for index, row in result.iterrows():
                self.all_results.loc[(metric_name, index), :] = row
        else:
            self.add_utility_metrics(result, metric_name=metric_name)

    def run_metrics_for_original(self: MetricsAggregator) -> None:
        """Run metrics for original dataset.

        This function splits the real dataset into two halves, treating one half as 'Original'
        and the other half as 'Synthetic', and then runs all metrics except utility.
        """
        train_data = load_data(self.real_data_path)
        train1, train2 = train_test_split(train_data, test_size=0.5, random_state=42)

        # Save the split datasets to temporary files
        original_path = Path("Real.csv")
        synthetic_path = Path("Original.csv")
        train1.to_csv(original_path, index=False)
        train2.to_csv(synthetic_path, index=False)

        temp_aggregator = MetricsAggregator(
            real_data_path=original_path,
            synthetic_data_paths=[synthetic_path],
            control_data=self.control_data,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
            distance_scaler=self.distance_scaler,
            singlingout_mode=self.singlingout_mode,
            singlingout_n_attacks=self.singlingout_n_attacks // 2,  # Half the attacks
            singlingout_n_cols=self.singlingout_n_cols,
            linkability_n_neighbors=self.linkability_n_neighbors,
            linkability_n_attacks=self.linkability_n_attacks,
            linkability_aux_cols=self.linkability_aux_cols,
            id_column=self.id_column,
            utility_test_path=self.utility_test_path,
            utility_models_path=self.utility_models_path,
            label_column=self.label_column,
            want_parallel=self.want_parallel,
            need_split=self.need_split,
        )

        # Skip running utility metrics
        metrics_to_run = [
            temp_aggregator.run_basic_quality_metrics,
            temp_aggregator.run_advanced_quality_metrics,
            temp_aggregator.run_likelihood_metrics,
            temp_aggregator.run_privacy_against_inference,
            temp_aggregator.run_propensity_score,
            temp_aggregator.run_distance_metrics,
            temp_aggregator.run_singling_out_metric,
            temp_aggregator.run_linkability_metric,
        ]

        for metric_fn in metrics_to_run:
            metric_fn()

        temp_results = temp_aggregator.all_results.copy()
        temp_results.columns = ["Original"]  # Rename columns to align with the "Original" dataset

        if self.all_results.empty:
            self.all_results = temp_results
        else:
            for metric_type in temp_results.index.get_level_values("Metric Type").unique():
                for metric in temp_results.loc[metric_type].index:
                    self.all_results.loc[(metric_type, metric), "Original"] = temp_results.loc[
                        (metric_type, metric),
                        "Original",
                    ]

        # Clean up temporary files
        original_path.unlink()
        synthetic_path.unlink()

    def run_metrics_for_models(self: MetricsAggregator) -> pd.DataFrame:
        """Runs all metrics and aggregates the results into a single table output.

        Returns:
            DataFrame: A pandas DataFrame containing the aggregated results from all metrics.
        """
        self.run_utility_metric()
        logging.info("Utility Done")

        self.run_basic_quality_metrics()
        logging.info("Basic Done")

        self.run_advanced_quality_metrics()
        logging.info("Advance Done")

        self.run_likelihood_metrics()
        logging.info("Likelihood Done")

        self.run_privacy_against_inference()
        logging.info("Privacy Done")

        self.run_propensity_score()
        logging.info("Propensity Done")

        self.run_distance_metrics()
        logging.info("Distance Done")

        self.run_singling_out_metric()
        logging.info("SinglingOut Done")

        self.run_linkability_metric()
        logging.info("Linkability Done")

        return self.all_results.apply(lambda x: x.apply(format_value))

    def run_all_with_original(self: MetricsAggregator) -> None:
        """Runs all metrics including those for the original dataset.

        This method first runs all metrics for the synthetic datasets and then runs metrics
        for the original dataset by splitting it into two halves.
        """
        self.drop_original_for_utility = False
        self.run_metrics_for_models()
        self.run_metrics_for_original()

    def reorder_metrics(self: MetricsAggregator) -> pd.DataFrame:
        """Reorder the DataFrame blocks according to a predefined primary metric order."""
        primary_metric_order = [
            "Utility",
            "Basic Quality",
            "Advanced Quality",
            "Likelihood",
            "Privacy Against Inference",
            "Propensity Score",
            "Distance",
            "Singling Out",
            "Linkability",
        ]

        sorted_results = pd.DataFrame()

        grouped = self.all_results.groupby(level="Metric Type")
        for metric in primary_metric_order:
            if metric in grouped.groups:
                sorted_results = pd.concat([sorted_results, grouped.get_group(metric)])

        self.all_results = sorted_results

    def save_results(self: MetricsAggregator, file_path: Path) -> None:
        """Saves the aggregated results to a specified file path using pickle serialization.

        Args:
            file_path (Path): The file path where the results will be saved.


        Usage Example:
        ----------------------
        ```python
        metrics_result.save_results(Path("res.pkl"))
        ```
        """
        self.reorder_metrics()

        data_to_save = {
            "results": self.all_results,
            "config": {
                "real_data_path": self.real_data_path,
                "synthetic_data_paths": self.synthetic_data_paths,
                "control_data": self.control_data,
                "key_fields": self.key_fields,
                "sensitive_fields": self.sensitive_fields,
                "distance_scaler": self.distance_scaler,
                "singlingout_mode": self.singlingout_mode,
                "singlingout_n_attacks": self.singlingout_n_attacks,
                "singlingout_n_cols": self.singlingout_n_cols,
                "linkability_n_neighbors": self.linkability_n_neighbors,
                "linkability_n_attacks": self.linkability_n_attacks,
                "linkability_aux_cols": self.linkability_aux_cols,
                "id_column": self.id_column,
                "utility_test_path": self.utility_test_path,
                "utility_models_path": self.utility_models_path,
                "label_column": self.label_column,
                "pos_label": self.pos_label,
                "need_split": self.need_split,
                "want_parallel": self.want_parallel,
            },
        }

        buf = io.BytesIO()
        if hasattr(self, "saved_plot"):
            self.saved_plot.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            data_to_save["plot"] = buf.read()

        with file_path.open("wb") as f:
            pickle.dump(data_to_save, f)
        logging.info("Results and all configuration data saved to %s", file_path)

    @classmethod
    def load_results(cls: type[MetricsAggregator], file_path: Path, *, show_plot: bool = True) -> MetricsAggregator:
        """Saves the aggregated results to a specified file path using pickle serialization.

        Args:
            file_path (Path): The file path where the results will be saved.
            show_plot (bool): Display the plot or not


        Usage Example:
        ----------------------
        ```python
        metrics_result = MetricsAggregator.load_results(Path("res.pkl"))
        metrics_result.all_results
        ```
        """
        with file_path.open("rb") as f:
            loaded_data = pickle.load(f)  # noqa: S301

        config = loaded_data["config"]
        instance = cls(
            real_data_path=config["real_data_path"],
            synthetic_data_paths=config["synthetic_data_paths"],
            control_data=config["control_data"],
            key_fields=config["key_fields"],
            sensitive_fields=config["sensitive_fields"],
            distance_scaler=config["distance_scaler"],
            singlingout_mode=config["singlingout_mode"],
            singlingout_n_attacks=config["singlingout_n_attacks"],
            singlingout_n_cols=config["singlingout_n_cols"],
            linkability_n_neighbors=config["linkability_n_neighbors"],
            linkability_n_attacks=config["linkability_n_attacks"],
            linkability_aux_cols=config["linkability_aux_cols"],
            id_column=config["id_column"],
            utility_test_path=config["utility_test_path"],
            utility_models_path=config["utility_models_path"],
            label_column=config["label_column"],
            pos_label=config["pos_label"],
            need_split=config["need_split"],
            load_data_now=False,
        )
        instance.all_results = loaded_data["results"]

        instance.reorder_metrics()

        if show_plot and "plot" in loaded_data:
            plot_data = io.BytesIO(loaded_data["plot"])
            loaded_plot = Image.open(plot_data)
            plt.figure(figsize=(10, 7))
            plt.imshow(loaded_plot)
            plt.axis("off")
            plt.show()

        return instance

    def run_or_update_metric(self: MetricsAggregator, metric_class_name: str, **kwargs: dict[str, Any]) -> None:
        """Run a metric and update or append its results in the aggregated DataFrame.

        Instantiate the provided metric class with given arguments, run it, and update
        or append its results in the `all_results` DataFrame.

        Args:
            metric_class_name (str): The name of the metric class to be instantiated and run.
            **kwargs (Dict[str, Any]): Keyword arguments to pass to the metric class constructor.

        Returns:
            None

        Usage Example:
        ----------------------
        ```python
        metrics_result = MetricsAggregator.load_results(Path("results.pkl"))
        metrics_result.run_or_update_metric(
            "LikelihoodMetrics",
            real_data_path=metrics_result.real_data_path,
            synthetic_data_paths=metrics_result.synthetic_data_paths,
            display_result=False,
        )
        ````
        """
        try:
            if metric_class_name == "Utility":
                self.run_utility_metric()
                return

            metric_class = METRIC_CLASSES.get(metric_class_name)
            if not metric_class:
                msg = f"Metric class '{metric_class_name}' is not recognized."
                raise ValueError(msg)  # noqa: TRY301

            if self.metadata is None:
                self.metadata = generate_metadata(load_data(self.real_data_path))

            metric_instance = metric_class(**kwargs)

            if not hasattr(metric_instance, "pivoted_results"):
                msg = "%s does not have attribute 'pivoted_results'"
                raise AttributeError(msg, metric_class.__name__)  # noqa: TRY301

            metric_name = type(metric_instance).__name__

            if metric_name in self.all_results.index.get_level_values("Metric Type").unique():
                for index, result in metric_instance.pivoted_results.iterrows():
                    self.all_results.loc[(metric_name, index), :] = result
            else:
                self.add_metrics(metric_instance)

            logging.info("%s results updated or added.", metric_name)
        except Exception as e:  # noqa: BLE001
            logging.error("Failed to run or update metric %s: %s", metric_class_name, e)  # noqa: TRY400
