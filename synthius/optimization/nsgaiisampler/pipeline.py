from __future__ import annotations

import logging
import warnings
from logging import getLogger
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, TVAESynthesizer
from sklearn.model_selection import train_test_split
from tabulate import tabulate

from synthius.automation import DEFAULT_METRICS, METRIC_CLASS_MAP, METRIC_REQUIRED_PARAMS, METRICS_MAP
from synthius.data import DataImputationPreprocessor
from synthius.model import ARF, WGAN, ModelFitter
from synthius.utilities import MetricsAggregator

logging.getLogger("SingleTableSynthesizer").setLevel(logging.ERROR)
logging.getLogger("sdv").setLevel(logging.ERROR)
logging.getLogger("copulas").setLevel(logging.ERROR)
logging.getLogger("rdt").setLevel(logging.ERROR)


logger = getLogger()

warnings.filterwarnings("ignore")


class PrivacyMetricError(ValueError):
    """Error raised when privacy metrics are selected without enabling required flags."""

    def __init__(self: PrivacyMetricError) -> None:
        """Initialize the PrivacyMetricError with a descriptive message."""
        super().__init__(
            "You selected a privacy-related metric (e.g., 'Privacy Risk'), "
            "but did not enable `linkability_metric=True` or `singlingout_metric=True`. "
            "Please set the correct flag to use these metrics.",
        )


class NSGAIISamplerHPOptimizer:
    """A hyperparameter optimizer using NSGA-II for synthetic data generation models.

    This class implements multi-objective optimization for various synthetic data generation models
    including CTGAN, CopulaGAN, TVAE, WGAN, and ARF. It uses the NSGA-II algorithm to optimize
    multiple competing objectives like utility, privacy, and data quality metrics.

    Usage Example:
    ----------------------
    ```python
    optimizer = NSGAIISamplerHPOptimizer(
        selected_metrics=["CS Test", "CategoricalNB", "Overall Quality"],
        key_fields=key_fields,
        sensitive_fields=sensitive_fields,
        linkability_aux_cols=aux_cols,
        distance_scaler="MinMaxScaler",
        singlingout_mode="multivariate",
        singlingout_n_attacks=2_000,
        singlingout_n_cols=7,
        linkability_n_neighbors=500,
        linkability_n_attacks=None,
    )

    best_trial = optimizer.run_synthetic_pipeline(
        real_data_path=data_path,
        label_column=LABEL,
        id_column=ID,
        output_path=synt_path,
        num_sample=NUM_SAMPLE,
        n_trials=20,
        positive_condition_value=True,
        negative_condition_value=False,
    )

    # For calculating all metrics after optimization is complete
    # You can use the following code to calculate all metrics
    # Make sure you passed all required parameters during initialization
    result = optimizer.evaluate_best_model_metrics()
    display(result.all_results)

    """

    def __init__(  # noqa: PLR0913
        self: NSGAIISamplerHPOptimizer,
        inference_all_columns: list[str],
        selected_metrics: list[str] | None = None,
        distance_scaler: str | None = None,
        singlingout_mode: str | None = None,
        singlingout_n_attacks: int | None = None,
        singlingout_n_cols: int | None = None,
        linkability_n_neighbors: int | None = None,
        linkability_n_attacks: int | None = None,
        linkability_aux_cols: list[list[str]] | None = None,
        inference_n_attacks: int | None = None,
        inference_sample_attacks: bool = False,  # noqa: FBT001, FBT002
        inference_use_custom_model: bool = False,  # noqa: FBT001, FBT002
        key_fields: list[str] | None = None,
        sensitive_fields: list[str] | None = None,
        *,
        linkability_metric: bool = False,
        singlingout_metric: bool = False,
        calculate_all_results: bool = True,
    ) -> None:
        """Initialize the NSGA-II hyperparameter optimizer.

        Args:
            selected_metrics (list[str] | None): List of metrics to evaluate. Defaults to `DEFAULT_METRICS`.
            distance_scaler (str | None): Scaler to use for distance-based metrics. Optional.
            singlingout_mode (str | None): Mode to use for singling-out metrics. Optional.
            singlingout_n_attacks (int | None): Number of attacks for singling-out metrics. Optional.
            singlingout_n_cols (int | None): Number of columns for singling-out metrics. Optional.
            linkability_n_neighbors (int | None): Number of neighbors for linkability metrics. Optional.
            linkability_n_attacks (int | None): Number of attacks for linkability metrics. Optional.
            linkability_aux_cols (list[list[str]] | None): Auxiliary columns for linkability metrics. Optional.
            inference_n_attacks (Optional[int]): Number of attack iterations for inference metric.
            inference_all_columns (List[str]): A list of all possible columns needed for the InferenceMetric.
            inference_sample_attacks (bool): Whether to sample the number of records to be used for the inference attack.
            inference_use_custom_model (bool): Whether to use a custom (XGBoost) model to perform the inference attack.
            key_fields (list[str] | None): List of key fields for metric evaluation. Optional.
            sensitive_fields (list[str] | None): List of sensitive fields for metric evaluation. Optional.
            linkability_metric: Whether to use linkability metrics. Defaults to False.
            singlingout_metric: Whether to use singling-out metrics. Defaults to False.
            calculate_all_results: Whether to calculate all results after optimization is complete. Defaults to True.
        """
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

        self.linkability_metric = linkability_metric
        self.singlingout_metric = singlingout_metric

        self.inference_n_attacks = inference_n_attacks
        self.inference_all_columns = inference_all_columns
        self.inference_sample_attacks = inference_sample_attacks
        self.inference_use_custom_model = inference_use_custom_model

        self.train_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.metadata: SingleTableMetadata | None = None
        self.conditions: list[Condition] = []

        self.calculate_all_results = calculate_all_results

        self.utility_op: bool = False

        self.seen_samples: set[tuple[str, frozenset[tuple[str, Any]]]] = set()
        self.failed_sampling_attempts = 0

    def _check_required_params(self: NSGAIISamplerHPOptimizer, metric_type: str, required_params: list[str]) -> None:
        """Check if required parameters are provided for a metric type."""
        for param in required_params:
            if getattr(self, param) is None:
                msg = f"{metric_type} metric requires '{param}', but it was not provided."
                raise ValueError(msg)

    def _validate_privacy_metrics(self: NSGAIISamplerHPOptimizer) -> None:
        """Validate privacy-related metrics and their parameters."""
        privacy_values = {
            "Privacy Risk",
            "CI(95%)",
            "Main Attack Success Rate",
            "Main Attack Marginal Error ±",
            "Baseline Attack Success Rate",
            "Baseline Attack Error ±",
            "Control Attack Success Rate",
            "Control Attack Error ±",
        }

        selected_privacy_metrics = any(m in privacy_values for m in self.selected_metrics)
        if selected_privacy_metrics and not (self.linkability_metric or self.singlingout_metric):
            raise PrivacyMetricError

        if self.linkability_metric:
            required_params = ["linkability_n_neighbors", "linkability_n_attacks", "linkability_aux_cols"]
            self._check_required_params("Linkability", required_params)

        if self.singlingout_metric:
            required_params = ["singlingout_mode", "singlingout_n_attacks", "singlingout_n_cols"]
            self._check_required_params("Singling-out", required_params)

    def validate_metric_params(self: NSGAIISamplerHPOptimizer) -> None:
        """Validate that all required parameters for selected metrics are provided.

        Raises:
            ValueError: If any required parameter for selected metrics is missing.
        """
        selected_classes = set()
        for metric_class, all_possible_metrics in METRICS_MAP.items():
            if any(m in self.selected_metrics for m in all_possible_metrics):
                selected_classes.add(metric_class)

        if "Utility" in selected_classes:
            self.utility_op = True

        self._validate_privacy_metrics()

        for mc in selected_classes:
            if mc in ["LinkabilityMetric", "SinglingOutMetric"]:
                continue

            required = METRIC_REQUIRED_PARAMS.get(mc, [])
            for param in required:
                if not hasattr(self, param) or getattr(self, param) is None:
                    if self.selected_metrics == DEFAULT_METRICS:
                        msg = (
                            f"Metric class '{mc}' requires '{param}' but it was not provided. "
                            "Since you are using the default metrics, please specify this parameter."
                        )
                        raise ValueError(msg)
                    msg = f"Metric class '{mc}' requires '{param}' but it was not provided."
                    raise ValueError(msg)

    def evaluate_utility_metrics(self: NSGAIISamplerHPOptimizer, synthetic_data_path: Path) -> dict:
        """Evaluates utility metrics for synthetic data using the ModelFitter.

        Args:
            synthetic_data_path (Path): Path to the synthetic data CSV file.

        Returns:
            dict: A dictionary containing evaluated utility metric scores (e.g., F1, Precision, Recall, etc.).
        """
        _ = ModelFitter(
            data_path=synthetic_data_path,
            label_column=self.label_column,
            experiment_name=synthetic_data_path.stem,
            models_base_path=self.output_path / "models",
            test_data_path=self.output_path / "test.csv",
            pos_label=True,
        )

        latest_results = ModelFitter.results_list[-1] if ModelFitter.results_list else {}
        return {
            "F1": latest_results.get("f1", 0),
            "F1_Weighted": latest_results.get("f1_weighted", 0),
            "F1_Macro": latest_results.get("f1_macro", 0),
            "Precision_Macro": latest_results.get("precision_macro", 0),
            "Recall_Macro": latest_results.get("recall_macro", 0),
            "Accuracy": latest_results.get("accuracy", 0),
        }

    def validate_required_params(self: NSGAIISamplerHPOptimizer) -> None:
        """Validate all required parameters if calculate_all_results is True."""
        if not self.calculate_all_results:
            return

        # List of required parameters for full metric calculation
        required_params = {
            "distance_scaler": self.distance_scaler,
            "singlingout_mode": self.singlingout_mode,
            "singlingout_n_attacks": self.singlingout_n_attacks,
            "singlingout_n_cols": self.singlingout_n_cols,
            "linkability_n_neighbors": self.linkability_n_neighbors,
            "linkability_aux_cols": self.linkability_aux_cols,
            "key_fields": self.key_fields,
            "sensitive_fields": self.sensitive_fields,
        }

        missing_params = [param for param, value in required_params.items() if value is None]

        if missing_params:
            logger.warning("\n You have chosen to calculate all metrics, but the following required parameters are missing:")
            for param in missing_params:
                logger.warning(" - %s", param)

            msg = f"Missing required parameters for full metrics evaluation: {', '.join(missing_params)}"
            raise ValueError(msg)

        logger.warning(
            "Please provide all required parameters for full metrics calculation or set calculate_all_results to False to ignore this error."
            "If you not provide all required parameters, metric calculation for the best model will not be performed.\n",
        )

        logger.info("✅ All required parameters for full metric calculation are provided.")

    def run_model(
        self: NSGAIISamplerHPOptimizer,
        model_name: str,
        params: dict,
    ) -> Path:
        """Run a synthetic data generation model with given parameters.

        Args:
            model_name: Name of the model to run (CTGAN, CopulaGAN, TVAE, WGAN, or ARF).
            params: Dictionary of model parameters.

        Returns:
            Path: Path to the generated synthetic data file.

        Raises:
            ValueError: If model_name is not supported.
        """
        self.train_data = pd.read_csv(self.output_path / "train.csv")

        trial_number = params.pop("trial")  # Extract trial number
        trial_suffix = "_".join(f"{k}-{v}" for k, v in params.items())

        # Updated filename to include trial number
        output_file = self.output_path / f"trial-{trial_number}_{model_name}_{trial_suffix}.csv"

        if model_name == "ARF":
            arf_model = ARF(
                x=self.train_data,
                id_column=self.id_column,
                **params,
                early_stop=True,
                verbose=False,
            )
            arf_model.forde()
            synthetic_data = arf_model.forge(n=self.num_sample)
            synthetic_data.to_csv(output_file, index=False)

        elif model_name == "CopulaGAN":
            cpgan_model = CopulaGANSynthesizer(
                metadata=self.metadata,
                **params,
            )

            cpgan_model.fit(self.train_data)
            synthetic_data = cpgan_model.sample_from_conditions(conditions=self.conditions)
            synthetic_data.to_csv(output_file, index=False)

        elif model_name == "CTGAN":
            ctgan_model = CTGANSynthesizer(
                metadata=self.metadata,
                **params,
            )

            ctgan_model.fit(self.train_data)
            synthetic_data = ctgan_model.sample_from_conditions(conditions=self.conditions)
            synthetic_data.to_csv(output_file, index=False)

        elif model_name == "TVAE":
            tvae_model = TVAESynthesizer(
                metadata=self.metadata,
                **params,
            )

            tvae_model.fit(self.train_data)
            synthetic_data = tvae_model.sample_from_conditions(conditions=self.conditions)
            synthetic_data.to_csv(output_file, index=False)

        elif model_name == "WGAN":
            preprocessor = DataImputationPreprocessor(self.train_data, self.id_column)
            processed_train_data = preprocessor.fit_transform()

            n_features = processed_train_data.shape[1]
            wgan_model = WGAN(
                n_features=n_features,
                **params,
            )

            wgan_model.train(
                processed_train_data,
            )

            synthetic_samples = wgan_model.generate_samples(self.num_sample)
            synthetic_data = pd.DataFrame(synthetic_samples, columns=processed_train_data.columns)
            decoded_data = preprocessor.inverse_transform(synthetic_data)
            decoded_data.to_csv(output_file, index=False)

        else:
            msg = f"Unsupported model: {model_name}"
            raise ValueError(msg)

        return output_file

    def _get_model_params(self: NSGAIISamplerHPOptimizer, trial: optuna.Trial) -> tuple[str, dict[str, Any]]:
        """Get model parameters for the trial.

        Args:
            trial: Optuna trial object.

        Returns:
            Tuple[str, Dict[str, Any]]: Model name and parameters dictionary.
        """
        model_name = trial.suggest_categorical("model", ["CTGAN", "CopulaGAN", "TVAE", "WGAN", "ARF"])

        # Define parameter options with proper type hints
        params: dict[str, list[Any]] = {
            "embedding_dim": [64, 128, 256, 512, 1024],
            "generator_dim": [(128, 128), (256, 256), (512, 512), (1024, 1024)],
            "discriminator_dim": [(128, 128), (256, 256), (512, 512), (1024, 1024)],
            "compress_dims": [(128, 128), (256, 256), (512, 512), (1024, 1024)],
            "decompress_dims": [(128, 128), (256, 256), (512, 512), (1024, 1024)],
            "base_nodes": [64, 128, 256, 512, 1024],
            "batch_size": [64, 128, 256, 512, 1024],
            "num_epochs": [10_000, 15_000, 30_000],
            "num_trees": [10, 20],
            "min_node_size": [5, 10],
            "critic_iters": [1, 2, 3, 4, 5],
        }

        sampled_params: dict[str, Any] = {}

        # Sample parameters based on model type
        if model_name in ["CTGAN", "CopulaGAN"]:
            for param in ["embedding_dim", "generator_dim", "discriminator_dim"]:
                sampled_params[param] = trial.suggest_categorical(param, params[param])
        elif model_name == "TVAE":
            for param in ["embedding_dim", "compress_dims", "decompress_dims"]:
                sampled_params[param] = trial.suggest_categorical(param, params[param])
        elif model_name == "WGAN":
            for param in ["base_nodes", "batch_size", "num_epochs", "critic_iters"]:
                sampled_params[param] = trial.suggest_categorical(param, params[param])
        elif model_name == "ARF":
            for param in ["num_trees", "min_node_size"]:
                sampled_params[param] = trial.suggest_categorical(param, params[param])

        sampled_params["trial"] = trial.number
        return model_name, sampled_params

    def _evaluate_metrics(self: NSGAIISamplerHPOptimizer, synthetic_data_path: Path) -> dict[str, float]:
        """Evaluate all selected metrics for the synthetic data.

        Args:
            synthetic_data_path: Path to synthetic data file.

        Returns:
            dict[str, float]: Dictionary of metric results.
        """
        results: dict[str, float] = {}

        for metric_class, all_possible_metrics in METRICS_MAP.items():
            selected = [m for m in self.selected_metrics if m in all_possible_metrics]
            if not selected:
                continue

            if metric_class == "Utility":
                utility_metrics = self.evaluate_utility_metrics(synthetic_data_path)
                results.update({metric: utility_metrics.get(metric, 0) for metric in selected})
                continue

            metric_params: dict[str, Any] = {
                "real_data_path": self.train_data,
                "synthetic_data_paths": [synthetic_data_path],
                "selected_metrics": selected,
                "display_result": False,
            }

            if metric_class == "PrivacyAgainstInference":
                metric_params.update({"key_fields": self.key_fields, "sensitive_fields": self.sensitive_fields})
            elif metric_class == "LinkabilityMetric" and self.linkability_metric:
                metric_params.update(
                    {
                        "aux_cols": self.linkability_aux_cols,
                        "n_neighbors": self.linkability_n_neighbors,
                        "n_attacks": self.linkability_n_attacks,
                        "control_data_path": self.output_path / "test.csv",
                    },
                )
            elif metric_class == "SinglingOutMetric" and self.singlingout_metric:
                metric_params.update(
                    {
                        "mode": self.singlingout_mode,
                        "n_attacks": self.singlingout_n_attacks,
                        "n_cols": self.singlingout_n_cols,
                        "control_data_path": self.output_path / "test.csv",
                    },
                )
            elif metric_class == "DistanceMetrics":
                metric_params.update({"scaler_choice": self.distance_scaler, "id_column": self.id_column})
            elif metric_class == "PropensityScore":
                metric_params.update({"id_column": self.id_column})

            if (
                metric_class not in ["LinkabilityMetric", "SinglingOutMetric"]
                or (metric_class == "LinkabilityMetric" and self.linkability_metric)
                or (metric_class == "SinglingOutMetric" and self.singlingout_metric)
            ):
                metric_instance = METRIC_CLASS_MAP[metric_class](**metric_params)
                metric_results = metric_instance.results[0] if metric_instance.results else {}
                results.update(metric_results)

        return results

    def objective(self: NSGAIISamplerHPOptimizer, trial: optuna.Trial) -> list[float] | None:
        """Objective function for optimization.

        Args:
            trial: Optuna trial object.

        Returns:
            list[float] | None: List of metric values for the trial, or None if trial should be skipped.
        """
        model_name, sampled_params = self._get_model_params(trial)
        sampled_params_tuple = (model_name, frozenset(sampled_params.items()))

        if sampled_params_tuple in self.seen_samples:
            logger.info("Trial %s: Duplicate hyperparameters detected. Skipping...", trial.number)
            return None

        self.seen_samples.add(sampled_params_tuple)
        logger.info("Trial %s is running for: Model %s with parameters: %s", trial.number, model_name, sampled_params)

        synthetic_data_path = self.run_model(model_name=model_name, params=sampled_params)
        results = self._evaluate_metrics(synthetic_data_path)

        return [results.get(metric, 0) for metric in self.selected_metrics]

    def _print_optimization_summary(self: NSGAIISamplerHPOptimizer, best_trials: list[optuna.trial.FrozenTrial]) -> pd.DataFrame:
        """Create and print summary of optimization results."""
        summary_table = []
        for trial in best_trials:
            hyperparams = {k: v for k, v in trial.params.items() if k != "model"}
            trial_data = {
                "Trial": trial.number,
                "Model": trial.params["model"],
                "Hyperparameters": str(hyperparams),
                **dict(zip(self.selected_metrics, trial.values, strict=False)),
            }
            summary_table.append(trial_data)

        summary_df = pd.DataFrame(summary_table)
        logger.info("\nBest Trials on the Pareto front Summary:")
        logger.info("\n%s", tabulate(summary_df, headers="keys", tablefmt="pretty"))
        return summary_df

    def _select_best_trial(self: NSGAIISamplerHPOptimizer, best_trials: list[optuna.trial.FrozenTrial], summary_df: pd.DataFrame) -> optuna.trial.FrozenTrial:
        """Select the best trial based on normalized metrics."""
        metric_columns = self.selected_metrics
        normalized_df = summary_df.copy()

        for metric in metric_columns:
            min_val = summary_df[metric].min()
            max_val = summary_df[metric].max()

            if max_val > min_val:
                if metric in {"Privacy Risk", "Autogluon", "XGBoost", "HistGradientBoosting"}:
                    normalized_df[metric] = 1 - (summary_df[metric] - min_val) / (max_val - min_val)
                else:
                    normalized_df[metric] = (summary_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[metric] = 0.5  # If all values are the same, assign neutral value

        normalized_df["Total Score"] = normalized_df[metric_columns].sum(axis=1)

        best_trial_row = normalized_df.loc[normalized_df["Total Score"].idxmax()]
        best_trial_number = int(best_trial_row["Trial"])
        return next(trial for trial in best_trials if trial.number == best_trial_number)

    def _save_best_model_data(self: NSGAIISamplerHPOptimizer, best_trial: optuna.trial.FrozenTrial) -> None:
        """Copy and rename the best model's output data to a dedicated directory.

        Args:
            best_trial: The best trial selected from optimization.
        """
        # Create best_model directory
        best_model_dir = self.output_path / "best_model"
        best_model_dir.mkdir(exist_ok=True)

        # Construct the original filename pattern
        model_name = best_trial.params["model"]
        trial_params = {k: v for k, v in best_trial.params.items() if k != "model"}
        params_str = "_".join(f"{k}-{v}" for k, v in trial_params.items())
        original_filename = f"trial-{best_trial.number}_{model_name}_{params_str}.csv"

        # Source and destination paths
        source_path = self.output_path / original_filename
        dest_path = best_model_dir / f"{model_name}.csv"

        # Copy and rename the file
        if source_path.exists():
            import shutil

            shutil.copy2(source_path, dest_path)
            logger.info("Best model data saved to: %s", dest_path)
        else:
            logger.error("Could not find best model output file: %s", source_path)

    def optimizer(self: NSGAIISamplerHPOptimizer, n_trials: int) -> optuna.trial.FrozenTrial | None:
        """Run the optimization process for synthetic data generation models."""
        directions = [
            "minimize" if metric in {"Privacy Risk", "Autogluon", "XGBoost", "HistGradientBoosting"} else "maximize" for metric in self.selected_metrics
        ]
        study = optuna.create_study(directions=directions, sampler=NSGAIISampler())

        self.failed_sampling_attempts = 0
        max_failed_trials = 5

        for i in range(n_trials):
            if self.failed_sampling_attempts >= max_failed_trials:
                logger.info("Optimization stopped early after %s trials due to 5 duplicate hyperparameter sets.", i)
                break

            trial = study.ask()
            result = self.objective(trial)

            if result is None:
                self.failed_sampling_attempts += 1
                continue

            self.failed_sampling_attempts = 0
            study.tell(trial, result)

            if study.best_trials:
                best_trial = study.best_trials[0]
                logger.info(
                    "Trial %s finished with value: %s. Best trial so far: %s with values: %s",
                    trial.number,
                    result,
                    best_trial.number,
                    best_trial.values,
                )

        best_trials = study.best_trials
        if not best_trials:
            logger.warning("No optimal trials found.")
            return None

        logger.info("Found %s optimal trial(s) on the Pareto front.", len(best_trials))
        summary_df = self._print_optimization_summary(best_trials)
        best_trial = self._select_best_trial(best_trials, summary_df)

        logger.info("\n🔹 The Best Model Based on Overall Multi-Metric Score:")
        logger.info("Trial %s: %s", best_trial.number, best_trial.params["model"])
        logger.info("Hyperparameters: %s", best_trial.params)
        logger.info("\n🔹 Best Values:")
        for metric, value in zip(self.selected_metrics, best_trial.values, strict=False):
            logger.info("- %s: %s", metric, value)

        if self.utility_op:
            ModelFitter(
                data_path=self.output_path / "train.csv",
                label_column=self.label_column,
                experiment_name="Original",
                models_base_path=self.output_path / "Original",
                test_data_path=self.output_path / "test.csv",
            )
            ModelFitter.plot_metrics(pos_label=True)

        # Save best model data
        self._save_best_model_data(best_trial)

        return best_trial

    def run_synthetic_pipeline(  # noqa: PLR0913
        self: NSGAIISamplerHPOptimizer,
        real_data_path: str | Path,
        label_column: str,
        id_column: str,
        output_path: str | Path,
        num_sample: int | None = None,
        test_data_path: Path | None = None,
        n_trials: int = 15,
        *,
        need_split: bool = True,
        positive_condition_value: str | bool = True,
        negative_condition_value: str | bool = False,
    ) -> optuna.trial.FrozenTrial | None:
        """Run the complete synthetic data generation pipeline.

        Args:
            real_data_path: Path to real data file.
            label_column: Name of the label column.
            id_column: Name of the ID column.
            output_path: Path for output files.
            num_sample: Number of synthetic samples to generate. Defaults to None.
            test_data_path: Path to test data file. Defaults to None.
            n_trials: Number of optimization trials. Defaults to 15.
            need_split: Whether to split data into train/test. Defaults to True.
            positive_condition_value: Value for positive class. Defaults to True.
            negative_condition_value: Value for negative class. Defaults to False.

        Returns:
            optuna.trial.FrozenTrial | None: Best trial found, or None if no valid trials.
        """
        data: pd.DataFrame = pd.read_csv(real_data_path).copy()
        self.label_column: str = label_column
        self.positive_condition_value = positive_condition_value  # Store positive condition value

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
        self.test_data.to_csv(self.output_path / "test.csv", index=False)  # type: ignore[union-attr]

        if num_sample is None:
            self.num_sample = len(self.train_data)
        else:
            self.num_sample = num_sample

        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.train_data)

        category_counts = self.train_data[self.label_column].value_counts()
        target_a = category_counts.get(positive_condition_value, 0)
        target_b = category_counts.get(negative_condition_value, 0)

        true_condition = Condition(num_rows=target_a, column_values={self.label_column: positive_condition_value})
        false_condition = Condition(num_rows=target_b, column_values={self.label_column: negative_condition_value})
        self.conditions = [true_condition, false_condition]

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        self.validate_required_params()
        self.validate_metric_params()

        best_trial = self.optimizer(n_trials=n_trials)

        if best_trial:
            idx = best_trial.number  # The "global trial number"
            logging.info("Best trial index: %s", idx)

            # Save best model data
            self._save_best_model_data(best_trial)

            return best_trial

        logging.warning("No best trial was found.")
        return None

    def evaluate_best_model_metrics(
        self: NSGAIISamplerHPOptimizer,
        *,
        want_parallel: bool = False,
    ) -> pd.DataFrame:
        """Run comprehensive metrics evaluation on the best model.

        Args:
            key_fields: List of key fields for privacy metrics. Required if not set during initialization.
            sensitive_fields: List of sensitive fields for privacy metrics. Required if not set during initialization.
            aux_cols: List of auxiliary column sets for linkability metrics. Required if not set during initialization.
            want_parallel: Whether to run metrics in parallel. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame containing all metric results.

        Raises:
            ValueError: If called before running optimization, if required paths don't exist,
                       or if required fields are not provided.
        """
        # Check if we have necessary paths
        best_model_path = self.output_path / "best_model"
        if not best_model_path.exists():
            msg = "No best model found. Please run optimization first."
            raise ValueError(msg)

        synthetic_data_paths = list(best_model_path.glob("*.csv"))
        if not synthetic_data_paths:
            msg = "No synthetic data found in best_model directory."
            raise ValueError(msg)

        train_path = self.output_path / "train.csv"
        test_path = self.output_path / "test.csv"
        models_path = self.output_path / "models"

        if not all(p.exists() for p in [train_path, test_path]):
            msg = "Training or test data not found in output directory."
            raise ValueError(msg)

        # Create models directory if it doesn't exist
        models_path.mkdir(exist_ok=True)

        # Run ModelFitter for each synthetic dataset
        for synt_path in synthetic_data_paths:
            ModelFitter(
                data_path=synt_path,
                label_column=self.label_column,
                experiment_name=synt_path.stem,
                models_base_path=models_path,
                test_data_path=test_path,
                pos_label=self.positive_condition_value,
            )

        # Run ModelFitter for original data
        ModelFitter(
            data_path=train_path,
            label_column=self.label_column,
            experiment_name="Original",
            models_base_path=models_path,
            test_data_path=test_path,
            pos_label=self.positive_condition_value,
        )

        # We are sure that all required parameters are provided, this is for ruff type checking
        if (
            self.key_fields is None
            or self.sensitive_fields is None
            or self.distance_scaler is None
            or self.singlingout_mode is None
            or self.singlingout_n_attacks is None
            or self.singlingout_n_cols is None
            or self.linkability_n_neighbors is None
            or self.linkability_aux_cols is None
        ):
            msg = "Some metrics parameters are not provided. Please provide all required parameters."
            raise ValueError(msg)

        # Use validated parameters for metrics aggregator
        metrics_aggregator = MetricsAggregator(
            real_data_path=train_path,
            synthetic_data_paths=synthetic_data_paths,
            control_data=test_path,
            key_fields=self.key_fields,
            sensitive_fields=self.sensitive_fields,
            distance_scaler=self.distance_scaler,
            singlingout_mode=self.singlingout_mode,
            singlingout_n_attacks=self.singlingout_n_attacks,
            singlingout_n_cols=self.singlingout_n_cols,
            linkability_n_neighbors=self.linkability_n_neighbors,
            linkability_n_attacks=None,
            linkability_aux_cols=self.linkability_aux_cols,
            inference_all_columns=self.inference_all_columns,
            inference_use_custom_model=self.inference_use_custom_model,
            inference_sample_attacks=self.inference_sample_attacks,
            inference_n_attacks=self.inference_n_attacks,
            id_column=self.id_column,
            utility_test_path=test_path,
            utility_models_path=models_path,
            label_column=self.label_column,
            want_parallel=want_parallel,
            pos_label=self.positive_condition_value,
            need_split=False,
        )

        # Run metrics evaluation
        metrics_aggregator.run_all_with_original()
        return metrics_aggregator
