# High-level API: generates synthetic data and produces metrics.
from pathlib import Path

from synthius.model import Synthesizer

from .generator import _generate
from .metrics import _compute_metrics


def run_synthius(  # noqa: PLR0913
    original_data_filename: str,
    data_dir: str,
    synth_dir: str,
    models_dir: str,
    results_dir: str,
    target_column: str,
    aux_cols: tuple[list[str], list[str]],
    models: list[Synthesizer] | None = None,
    random_seed: int | None = None,
    id_column: str | None = None,
    positive_label: str | bool = True,  # noqa: FBT002
    key_fields: list[str] | None = None,
    sensitive_fields: list[str] | None = None,
    metric_aggregator_mode: str = "onlyoriginal",
    results_checkpoint_path: Path | None = None
) -> None:
    """High-level API: Generates synthetic data and computes evaluation metrics.

    Args:
        original_data_filename: CSV filename containing the original dataset.
        data_dir: Directory containing the original data.
        synth_dir: Directory where synthetic datasets are saved.
        models_dir: Directory containing model outputs or checkpoints.
        results_dir: Directory where final aggregated results are stored.
        target_column: Name of the target column for supervised tasks.
        models: List of instantiated Synthesizer objects.
        random_seed: Random seed for reproducibility.
        id_column: The id column to be set aside from generation (and added artificially)
            and utility evaluation.
        positive_label: Label considered as positive in binary classification.
        key_fields: Fields used as unique identifiers (optional).
        sensitive_fields: Fields used for fairness metrics (optional).
        aux_cols: Auxiliary columns used in metrics computation (optional).
        metric_aggregator_mode: Mode for aggregating metrics.
            Can be "synthetic" for evaluating the synthetic data only,
            "onlyoriginal" for evaluating the original copy-the-data experiment,
            "runall" for evaluating both synthetic data and original copy-the-data experiment.
    """
    key_fields = key_fields or []
    sensitive_fields = sensitive_fields or []
    aux_cols = aux_cols or [[]]

    # Step 1: Generate Synthetic Data
    _generate(
        original_data_filename=original_data_filename,
        data_dir=data_dir,
        synth_dir=synth_dir,
        target_column=target_column,
        models=models,
        random_seed=random_seed,
        positive_label=positive_label,
        id_column=id_column
    )

    # Step 2: Evaluate Synthetic Data
    _compute_metrics(
        data_dir=data_dir,
        synth_dir=synth_dir,
        models_dir=models_dir,
        results_dir=results_dir,
        target_column=target_column,
        positive_label=positive_label,
        key_fields=key_fields,
        sensitive_fields=sensitive_fields,
        aux_cols=aux_cols,
        id_column=id_column,
        metric_aggregator_mode=metric_aggregator_mode,
        results_checkpoint_path=results_checkpoint_path,
    )
