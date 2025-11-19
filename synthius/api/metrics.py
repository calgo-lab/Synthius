# This notebook runs the metrics on the data
from pathlib import Path

import pandas as pd

from synthius.metric.utils import utils
from synthius.utilities import MetricsAggregator


def _get_metrics(  # noqa: PLR0913
    data_dir: str | Path,
    synth_dir: str | Path,
    models_dir: str | Path,
    results_dir: str | Path,
    target_column: str,
    key_fields: list[str],
    sensitive_fields: list[str],
    aux_cols: tuple[list[str], list[str]],
    positive_label: str | bool = True,  # noqa: FBT002
    id_column: str | None = None,
    metric_aggregator_mode: str | None = None,
    inference_all_columns: list[str] | None = None,
    inference_use_custom_model: bool = True,  # noqa: FBT001, FBT002
    inference_sample_attacks: bool = False,  # noqa: FBT001, FBT002
    inference_n_attacks: int | None = None,
) -> None:
    """Module for running all Synthius evaluation metrics in batch mode.

    This script loads real, synthetic, and test datasets, initializes a
    MetricsAggregator object, and executes the desired evaluation mode
    (e.g., synthetic-only, original-only, or both). It saves the resulting
    metrics to a CSV file for later inspection.

    Args:
        data_dir (str | Path): Directory containing real train/test datasets.
        synth_dir (str | Path): Directory containing synthetic data files.
        models_dir (str | Path): Directory containing trained model files.
        results_dir (str | Path): Output directory for metric results.
        target_column (str | int): Name or index of the target column in datasets.
        key_fields (List[str]): List of key fields used for linkage or comparison.
        sensitive_fields (List[str]): List of sensitive columns for privacy metrics.
        aux_cols (List[List[str]]): Auxiliary columns used for linkability attacks.
        positive_label (int | str | bool, optional): Positive label for binary tasks.
        id_column (str | int | None, optional): Unique identifier column, if available.
        metric_aggregator_mode (str | None, optional): Mode of metric aggregation.
        inference_all_columns (List[str], optional): Use all columns for inference metrics.
        inference_use_custom_model (bool, optional): Use custom inference model.
        inference_sample_attacks (bool, optional): Whether to sample inference attacks.
        inference_n_attacks (Optional[int], optional): Number of inference attacks.
    """
    # Directories and data
    data_dir = Path(data_dir)
    synth_dir = Path(synth_dir)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_data = data_dir / "train.csv"
    test_data = data_dir / "test.csv"

    # Just glob the dir path -- Decou
    synthetic_data_paths = list(synth_dir.glob("*.csv"))

    # Also do this...
    # We make sure we use the clean columns from the data
    inference_all_columns = utils.clean_columns(pd.read_csv(test_data)).columns if inference_all_columns is None else inference_all_columns

    # --- Build metrics aggregator ---
    metrics_result = MetricsAggregator(
        real_data_path=train_data,
        synthetic_data_paths=synthetic_data_paths,
        control_data=test_data,
        key_fields=key_fields,
        sensitive_fields=sensitive_fields,
        distance_scaler="MinMaxScaler",
        singlingout_mode="multivariate",
        singlingout_n_attacks=6_000,
        singlingout_n_cols=7,
        linkability_n_neighbors=500,
        linkability_n_attacks=None,
        linkability_aux_cols=aux_cols,
        id_column=id_column,
        utility_test_path=test_data,
        utility_models_path=models_dir,
        inference_all_columns=inference_all_columns,
        inference_use_custom_model=inference_use_custom_model,
        inference_sample_attacks=inference_sample_attacks,
        inference_n_attacks=inference_n_attacks,
        label_column=target_column,
        pos_label=positive_label,
        want_parallel=False,
        need_split=False,
    )

    if metric_aggregator_mode is None:
        metrics_result.run_all_with_original()
    else:
        metric_aggregator_mode = metric_aggregator_mode.lower().replace(" ", "")

    if metric_aggregator_mode == "synthetic":
        metrics_result.run_metrics_for_models()

    elif metric_aggregator_mode == "onlyoriginal":
        metrics_result.run_metrics_for_original()

    else:
        metrics_result.run_all_with_original()

    # Takes a long time VVV
    metrics_result.all_results.to_csv(results_dir / "results.csv")
