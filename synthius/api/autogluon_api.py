"""API for the generation of synthetic datasets.

This module provides a function to fit and evaluate models on both real and
synthetic datasets, storing the resulting metrics and trained models.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from synthius.model import ModelFitter, ModelLoader


def fit_models(
    metrics_dir: str | Path,
    data_dir: str | Path,
    synth_dir: str | Path,
    models_dir: str | Path,
    target_column: str | int,
    positive_label: int | str | bool = True,
    models: List[str] | None = None,
) -> None:
    """Fit models on synthetic and original datasets and store metrics.

    This function loads synthetic datasets (from CSV files), trains models
    using them and the original dataset, evaluates performance on test data,
    and exports metrics to the specified output directory.

    Args:
        metrics_dir: Directory where the resulting metrics CSV will be saved.
        data_dir: Directory containing the original training and testing CSV files.
            Expected to contain `train.csv` and `test.csv`.
        synth_dir: Directory containing synthetic dataset CSVs.
        models_dir: Directory where trained models will be saved.
        target_column: The target column name or index in the dataset.
        positive_label: The value representing the positive class in classification.
        models: Optional list of synthetic model names (without `.csv` extension).
            If None, defaults to all available synthetic dataset types.

    Returns:
        None
    """
    data_dir = Path(data_dir)
    synth_dir = Path(synth_dir)
    models_dir = Path(models_dir)
    metrics_dir = Path(metrics_dir)

    # Ensure output directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_data = data_dir / "train.csv"
    test_data = data_dir / "test.csv"

    # Default synthetic dataset names
    if models is None:
        models = [
            "CopulaGAN",
            "CTGAN",
            "GaussianCopula",
            "TVAE",
            "GaussianMultivariate",
            "ARF",
            "WGAN",
        ]

    synthetic_data_paths = [synth_dir / f"{m}.csv" for m in models]

    # Fit models on synthetic datasets
    for syn_path in synthetic_data_paths:
        ModelFitter(
            data_path=syn_path,
            label_column=target_column,
            experiment_name=syn_path.stem,
            models_base_path=models_dir,
            test_data_path=test_data,
            pos_label=positive_label,
        )

    # Fit model on the original dataset
    ModelFitter(
        data_path=train_data,
        label_column=target_column,
        experiment_name="Original",
        models_base_path=models_dir,
        test_data_path=test_data,
        pos_label=positive_label,
    )

    # Save metrics
    ModelFitter.display_metrics()
    metrics_df = ModelFitter.pivoted_results
    metrics_df.to_csv(metrics_dir / "metrics.csv")


if __name__ == "__main__":
    fit_models(
        metrics_dir="/storage/Synthius/examples/metrics",
        data_dir="/storage/Synthius/examples/data",
        synth_dir="/storage/Synthius/examples/results",
        models_dir="/storage/Synthius/examples/models",
        target_column="target_binary",
        positive_label=1,
        models=["CopulaGAN", "CTGAN", "GaussianCopula", "TVAE", "GaussianMultivariate", "ARF"], # Omit WGAN due to deep bugs
    )
