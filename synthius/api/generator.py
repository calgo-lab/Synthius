# The API for the generation of synthetic datasets.
# 2025.10.29
from pathlib import Path

import pandas as pd
from sdv.sampling import Condition
from sklearn.model_selection import train_test_split

from synthius.model import (
    ARFSynthesizer,
    SDVCopulaGANSynthesizer,
    SDVCTGANSynthesizer,
    SDVGaussianCopulaSynthesizer,
    SDVTVAESynthesizer,
    Synthesizer,
    SynthesizerGaussianMultivariate,
)


def _preprocess_data(
    original_data_filename: str,
    data_dir: str,
    target_column: str | int,
    random_seed: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, split, preprocess, and save the data."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / original_data_filename

    if not file_path.exists():
        msg = f"File not found: {file_path}"
        raise FileNotFoundError(msg)

    data = pd.read_csv(file_path, low_memory=False)

    if target_column not in data.columns:
        msg = f"Target column '{target_column}' not found in data."
        raise ValueError(msg)

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=random_seed,
        stratify=data[target_column],
    )

    # Save the train and test
    train_file = data_path / "train.csv"
    test_file = data_path / "test.csv"

    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    return data, train_data, test_data


def _run_model(
    model: Synthesizer,
    train_data: pd.DataFrame,
    total_samples: int,
    results_dir: str,
    conditions: list | None = None,
) -> None:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    try:
        model.fit(train_data)
        synthetic_data = model.generate(total_samples, conditions=conditions)
        synthetic_data.to_csv(results_path / f"{model.name}.csv", index=False)
    except Exception as e:  # noqa: BLE001
        print(f"[Error] {type(model).__name__}: {e}")  # noqa: T201


def _generate(  # noqa: PLR0913
    original_data_filename: str,
    target_column: str | int,
    data_dir: str = ".",
    synth_dir: str = ".",
    id_column: str | None = None,
    models: list[Synthesizer] | None = None,
    random_seed: int | None = None,
) -> None:
    """Generate synthetic datasets from a source dataset using Synthesizer instances."""
    data, train_data, test_data = _preprocess_data(original_data_filename, data_dir, target_column, random_seed)

    # Build conditional sampling info
    total_samples = train_data.shape[0]
    unique_classes = data[target_column].unique()
    if len(unique_classes) < 2:  # noqa: PLR2004
        msg = "Target column must contain at least two classes."
        raise ValueError(msg)

    if len(unique_classes) == 2:  # noqa: PLR2004
        true_samples = train_data[target_column].sum()
        false_samples = total_samples - true_samples
        conditions = [
            Condition(num_rows=true_samples, column_values={target_column: True}),
            Condition(num_rows=false_samples, column_values={target_column: False}),
        ]
    else:
        counts = train_data[target_column].value_counts()
        conditions = [
            Condition(num_rows=int(counts.get(1, 0)), column_values={target_column: 1}),
            Condition(num_rows=int(counts.get(0, 0)), column_values={target_column: 0}),
        ]

    # Instantiate default models if user did not provide
    if models is None:
        models = [
            SDVCopulaGANSynthesizer(),
            SDVCTGANSynthesizer(),
            SDVGaussianCopulaSynthesizer(),
            SDVTVAESynthesizer(),
            SynthesizerGaussianMultivariate(results_path=synth_dir),
            ARFSynthesizer(id_column=id_column),
        ]

    # Run each model
    for model_instance in models:
        _run_model(model_instance, train_data, total_samples, synth_dir, conditions=conditions)
