from __future__ import annotations

import logging
from logging import getLogger
from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import GridSampler
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sklearn.model_selection import train_test_split

from synthius.model import ModelFitter

from .metrics_map import DEFAULT_METRICS, METRIC_CLASS_MAP, METRIC_REQUIRED_PARAMS, METRICS_MAP
from .models import MODEL_RUNNERS, run_model

logger = getLogger()


class SyntheticModelFinder:
    """A class to optimize synthetic data generation models and evaluate them using specified metrics.

    Attributes:
        selected_metrics (list[str]): Metrics to evaluate models.
        distance_scaler (str): Scaler for distance metrics.
        singlingout_mode (str): Mode for singling-out metrics.
        singlingout_n_attacks (int): Number of singling-out attacks.
        singlingout_n_cols (int): Number of columns for singling-out metrics.
        linkability_n_neighbors (int): Neighbors for linkability metrics.
        linkability_n_attacks (int): Attacks for linkability metrics.
        linkability_aux_cols (list[list[str]]): Auxiliary columns for linkability metrics.
        key_fields (list[str]): Key fields for privacy evaluation.
        sensitive_fields (list[str]): Sensitive fields for privacy evaluation.

        Usage Example:
        ----------------------
        ```python
        best_trial = SyntheticModelFinder(
            key_fields=key_fields,
            sensitive_fields=sensitive_fields,
            selected_metrics=["F1", "CategoricalZeroCAP"],
        )

        best_trial.run_synthetic_pipeline(
            real_data_path=data_path,
            label_column=LABEL,
            id_column=ID,
            output_path=synt_path,
            num_sample=NUM_SAMPLE,
            positive_condition_value=True,
            negative_condition_value=False,
        )

    """

    def __init__(  # noqa: PLR0913
        self: SyntheticModelFinder,
        selected_metrics: list[str] | None = None,
        distance_scaler: str | None = None,
        singlingout_mode: str | None = None,
        singlingout_n_attacks: int | None = None,
        singlingout_n_cols: int | None = None,
        linkability_n_neighbors: int | None = None,
        linkability_n_attacks: int | None = None,
        linkability_aux_cols: list[list[str]] | None = None,
        key_fields: list[str] | None = None,
        sensitive_fields: list[str] | None = None,
    ) -> None:
        """Initializes SyntheticModelFinder with metrics and parameter configurations."""
        self.selected_metrics = selected_metrics if selected_metrics else DEFAULT_METRICS
        self.distance_scaler = distance_scaler
        self.singlingout_mode = singlingout_mode
        self.singlingout_n_attacks = singlingout_n_attacks
        self.singlingout_n_cols = singlingout_n_cols
        self.linkability_n_neighbors = linkability_n_neighbors
        self.linkability_n_attacks = linkability_n_attacks
        self.linkability_aux_cols = linkability_aux_cols
        self.key_fields = key_fields
        self.sensitive_fields = sensitive_fields

        self.train_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.metadata: SingleTableMetadata | None = None
        self.conditions: list[Condition] = []

        self.utility_op: bool = False

    def validate_metric_params(self: SyntheticModelFinder) -> None:
        """Validates whether required parameters for selected metrics are provided."""
        selected_classes = set()
        for metric_class, all_possible_metrics in METRICS_MAP.items():
            if any(m in self.selected_metrics for m in all_possible_metrics):
                selected_classes.add(metric_class)

        if "Utility" in selected_classes:
            self.utility_op = True

        for mc in selected_classes:
            required = METRIC_REQUIRED_PARAMS.get(mc, [])

            for param in required:
                if not hasattr(self, param) or getattr(self, param) is None:
                    if self.selected_metrics == DEFAULT_METRICS:
                        msg = (
                            f"Metric class '{mc}' requires '{param}' but it was not provided. "
                            "Since you are using the default metrics, please specify this parameter."
                        )
                        raise ValueError(
                            msg,
                        )

                    msg = f"Metric class '{mc}' requires '{param}' but it was not provided."
                    raise ValueError(
                        msg,
                    )

    def evaluate_utility_metrics(self: SyntheticModelFinder, synthetic_data_path: Path) -> dict:
        """Evaluate utility metrics using ModelFitter."""
        _ = ModelFitter(
            data_path=synthetic_data_path,
            label_column=self.label_column,
            experiment_name=synthetic_data_path.stem,
            models_base_path=self.output_path / "models",
            test_data_path=self.output_path / "test.csv",
            pos_label=True,
        )

        # Fetch the latest results from ModelFitter
        latest_results = ModelFitter.results_list[-1] if ModelFitter.results_list else {}
        return {
            "F1": latest_results.get("f1", 0),
            "F1_Weighted": latest_results.get("f1_weighted", 0),
            "F1_Macro": latest_results.get("f1_macro", 0),
            "Precision_Macro": latest_results.get("precision_macro", 0),
            "Recall_Macro": latest_results.get("recall_macro", 0),
            "Accuracy": latest_results.get("accuracy", 0),
        }

    def objective(self: SyntheticModelFinder, trial: optuna.Trial) -> list[float]:
        """Objective function for model optimization using Optuna.

        Args:
            trial (optuna.Trial): The optimization trial object.

        Returns:
            List[float]: Metric scores for selected metrics.
        """
        # Pick a model
        model_name: str = trial.suggest_categorical("model", list(MODEL_RUNNERS.keys()))

        # Run chosen model
        run_model(
            model_name=model_name,
            train_data=self.train_data,
            metadata=self.metadata,
            conditions=self.conditions,
            id_column=self.id_column,
            num_sample=self.num_sample,
            save_path=self.output_path,
        )

        # Evaluate
        results: dict[str, float] = {}
        synthetic_data_path: Path = self.output_path / f"{model_name}.csv"

        for metric_class, all_possible_metrics in METRICS_MAP.items():
            selected = [m for m in self.selected_metrics if m in all_possible_metrics]
            if not selected:
                continue

            if metric_class == "PrivacyAgainstInference":
                metric_instance = METRIC_CLASS_MAP[metric_class](
                    real_data_path=self.train_data,
                    synthetic_data_paths=[synthetic_data_path],
                    key_fields=self.key_fields,
                    sensitive_fields=self.sensitive_fields,
                    metadata=None,
                    selected_metrics=selected,
                    display_result=False,
                )

            elif metric_class == "LinkabilityMetric":
                metric_instance = METRIC_CLASS_MAP[metric_class](
                    real_data_path=self.train_data,
                    synthetic_data_paths=[synthetic_data_path],
                    aux_cols=self.linkability_aux_cols,
                    n_neighbors=self.linkability_n_neighbors,
                    n_attacks=self.linkability_n_attacks,
                    control_data_path=self.test_data,
                    selected_metrics=selected,
                    display_result=False,
                )

            elif metric_class == "SinglingOutMetric":
                metric_instance = METRIC_CLASS_MAP[metric_class](
                    real_data_path=self.train_data,
                    synthetic_data_paths=[synthetic_data_path],
                    mode=self.singlingout_mode,
                    n_attacks=self.singlingout_n_attacks,
                    n_cols=self.singlingout_n_cols,
                    control_data_path=self.test_data,
                    selected_metrics=selected,
                    display_result=False,
                )

            elif metric_class == "DistanceMetrics":
                metric_instance = METRIC_CLASS_MAP[metric_class](
                    real_data_path=self.train_data,
                    synthetic_data_paths=[synthetic_data_path],
                    scaler_choice=self.distance_scaler,
                    id_column=self.id_column,
                    selected_metrics=selected,
                    display_result=False,
                )

            elif metric_class == "PropensityScore":
                metric_instance = METRIC_CLASS_MAP[metric_class](
                    real_data_path=self.train_data,
                    synthetic_data_paths=[synthetic_data_path],
                    id_column=self.id_column,
                    selected_metrics=selected,
                    display_result=False,
                )

            elif metric_class == "Utility":
                utility_metrics = self.evaluate_utility_metrics(synthetic_data_path)
                metric_results = {metric: utility_metrics.get(metric, 0) for metric in selected}

            else:
                metric_instance = METRIC_CLASS_MAP[metric_class](
                    real_data_path=self.train_data,
                    synthetic_data_paths=[synthetic_data_path],
                    metadata=None,
                    selected_metrics=selected,
                    display_result=False,
                )

            if metric_class != "Utility":
                metric_results = metric_instance.results[0] if metric_instance.results else {}

            results.update(metric_results)

        # For any metric that wasn't returned by the metric instance, default to 0
        return [results.get(metric, 0) for metric in self.selected_metrics]

    def optimize_models(self: SyntheticModelFinder) -> optuna.trial.FrozenTrial | None:
        """Optimizes models using Optuna and returns the best trial.

        Returns:
            Optional[optuna.trial.FrozenTrial]: Best trial if found, otherwise None.
        """
        search_space: dict[str, list[str]] = {"model": list(MODEL_RUNNERS.keys())}
        sampler: GridSampler = GridSampler(search_space, seed=42)
        directions: list[str] = ["maximize"] * len(self.selected_metrics)

        study: optuna.Study = optuna.create_study(directions=directions, sampler=sampler)

        study.optimize(
            lambda trial: self.objective(trial),
            n_trials=len(MODEL_RUNNERS),
        )

        best_trials: list[optuna.trial.FrozenTrial] = study.best_trials

        logging.info("Found %s optimal trial(s) on the Pareto front.", len(best_trials))

        for i, trial in enumerate(best_trials):
            logging.info("Trial %s: Model -> %s", i + 1, trial.params["model"])
            for metric, value in zip(self.selected_metrics, trial.values):
                logging.info("%s: %s", metric, value)

        if self.utility_op:
            ModelFitter(
                data_path=self.output_path / "train.csv",
                label_column=self.label_column,
                experiment_name="Original",
                models_base_path=self.output_path / "Original",
                test_data_path=self.output_path / "test.csv",
            )

            ModelFitter.plot_metrics(pos_label=True)

        return best_trials[0] if best_trials else None

    def run_synthetic_pipeline(  # noqa: PLR0913
        self: SyntheticModelFinder,
        real_data_path: str | Path,
        label_column: str,
        id_column: str,
        output_path: str | Path,
        num_sample: int | None = None,
        test_data_path: Path | None = None,
        *,
        need_split: bool = True,
        positive_condition_value: str | bool = True,
        negative_condition_value: str | bool = False,
    ) -> optuna.trial.FrozenTrial | None:
        """Runs the complete synthetic data generation pipeline.

        Args:
            real_data_path (Union[str, Path]): Path to the real dataset.
            label_column (str): Column name containing labels in the dataset.
            id_column (str): Column name used as the identifier.
            output_path (Union[str, Path]): Path to save synthetic datasets.
            num_sample (int): Number of samples to generate. If None, the number od sample is set to the size of
                              the train dataset.
            test_data_path (Optional[Path]): Path to test dataset (if available).
            need_split (bool): Whether to split the dataset into training and testing sets.
            positive_condition_value (Union[str, bool]): Value representing the positive condition.
            negative_condition_value (Union[str, bool]): Value representing the negative condition.

        Returns:
            Optional[optuna.trial.FrozenTrial]: The best trial if found, otherwise None.
        """
        data: pd.DataFrame = pd.read_csv(real_data_path).copy()
        self.label_column: str = label_column

        if need_split:
            self.train_data, self.test_data = train_test_split(
                data,
                test_size=0.2,
                random_state=42,
                stratify=data[self.label_column],
            )
        else:
            self.train_data = data
            self.test_data = pd.read_csv(test_data_path).copy() if test_data_path else None

        self.output_path: Path = Path(output_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.id_column: str = id_column

        self.train_data.to_csv(self.output_path / "train.csv", index=False)
        self.test_data.to_csv(self.output_path / "test.csv", index=False)

        if num_sample is None:
            self.num_sample = len(self.train_data)
        else:
            self.num_sample = num_sample

        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)

        category_counts = self.train_data[self.label_column].value_counts()
        target_a = category_counts.get(positive_condition_value, 0)
        target_b = category_counts.get(negative_condition_value, 0)

        true_condition = Condition(num_rows=target_a, column_values={self.label_column: positive_condition_value})
        false_condition = Condition(num_rows=target_b, column_values={self.label_column: negative_condition_value})
        self.conditions = [true_condition, false_condition]

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.validate_metric_params()
        best_trial = self.optimize_models()

        if best_trial:
            logging.info("Best Model Selected Automatically: %s", best_trial.params["model"])
            return best_trial

        logging.warning("No best trial was found.")
        return None
