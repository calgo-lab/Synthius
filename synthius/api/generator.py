# The API for the generation of synthetic datasets.
# 2025.10.29
from pathlib import Path
from typing import Tuple

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)
from sklearn.model_selection import train_test_split
from synthius.model import Synthesizer, SDVSynthesizer, SynthesizerGaussianMultivariate, ARFSynthesizer, WGANSynthesizer


def _preprocess_data(
    original_data_filename: str,
    data_dir: str,
    target_column: str | int,
    random_seed: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, split, preprocess, and save the data."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    file_path = data_path / original_data_filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    data = pd.read_csv(file_path, low_memory=False)

    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data.")

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
        print(f"[Info] Model {model.name} finished. Saved to {results_path / f'{model.name}.csv'}")
    except Exception as e:
        print(f"[Error] {type(model).__name__}: {e}")


def _generate(
    original_data_filename: str,
    target_column: str | int,
    data_dir: str = ".",
    synth_dir: str = ".",
    id_column: str | int | None = None,
    models: list[Synthesizer] | None = None,
    random_seed: int | None = None,
) -> None:
    """Generate synthetic datasets from a source dataset using Synthesizer instances."""
    data, train_data, test_data = _preprocess_data(
        original_data_filename, data_dir, target_column, random_seed
    )



    # Build conditional sampling info
    total_samples = train_data.shape[0]
    unique_classes = data[target_column].unique()
    if len(unique_classes) < 2:
        raise ValueError("Target column must contain at least two classes.")

    if len(unique_classes) == 2:
        true_samples = train_data[target_column].sum()
        false_samples = total_samples - true_samples
        conditions = [
            Condition(num_rows=true_samples, column_values={target_column: True}),
            Condition(num_rows=false_samples, column_values={target_column: False})
        ]
    else:
        counts = train_data[target_column].value_counts()
        conditions = [
            Condition(num_rows=int(counts.get(1, 0)), column_values={target_column: 1}),
            Condition(num_rows=int(counts.get(0, 0)), column_values={target_column: 0})
        ]

    # Instantiate default models if user did not provide
    if models is None:
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)
        models = [
            SDVSynthesizer(CopulaGANSynthesizer, metadata),
            SDVSynthesizer(CTGANSynthesizer, metadata),
            SDVSynthesizer(GaussianCopulaSynthesizer, metadata),
            SDVSynthesizer(TVAESynthesizer, metadata),
            SynthesizerGaussianMultivariate(results_path=synth_dir),
            ARFSynthesizer(id_column=id_column),
            WGANSynthesizer(),
        ]

    # Run each model
    for model_instance in models:
        _run_model(model_instance, train_data, total_samples, synth_dir, conditions=conditions)


# For testing purposes
if __name__ == "__main__":
    models = [
        SDVSynthesizer(CopulaGANSynthesizer),
        SDVSynthesizer(CTGANSynthesizer),
        SDVSynthesizer(GaussianCopulaSynthesizer),
        SDVSynthesizer(TVAESynthesizer),
        SynthesizerGaussianMultivariate(results_path="/storage/Synthius/examples/synthetic_data/"),
        ARFSynthesizer(id_column=None),
        # WGANSynthesizer()  # It's broken rn though
    ]
    _generate(
        original_data_filename="data.csv",
        data_dir="/storage/Synthius/examples/data/",
        synth_dir="/storage/Synthius/examples/synthetic_data/",
        target_column="target_binary",
        models=models,
        random_seed=42
    )
