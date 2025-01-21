from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from synthius.data import DataImputationPreprocessor
from synthius.model import (
    ARF,
    WGAN,
    GaussianMultivariateSynthesizer,
    data_batcher,
)

if TYPE_CHECKING:
    from pathlib import Path

    from sdv.metadata import SingleTableMetadata
    from sdv.sampling import Condition

SMALL_DATASET_THRESHOLD = 10_000
MEDIUM_DATASET_THRESHOLD = 25_000
LARGE_DATASET_THRESHOLD = 50_000


def run_ctgan(
    train_data: pd.DataFrame,
    metadata: SingleTableMetadata,
    conditions: list[Condition],
    save_path: Path,
) -> None:
    """Train a CTGAN model on the given data, then sample synthetic data using the specified conditions.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        metadata (SingleTableMetadata): Metadata describing the table schema.
        conditions (List[Condition]): List of conditions specifying how many rows to sample
                                      and what column values to enforce.
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(train_data)
    synthetic_data = synthesizer.sample_from_conditions(conditions=conditions)
    synthetic_data.to_csv(save_path / "CTGAN.csv", index=False)


def run_copulagan(
    train_data: pd.DataFrame,
    metadata: SingleTableMetadata,
    conditions: list[Condition],
    save_path: Path,
) -> None:
    """Train a CopulaGAN model on the given data, then sample synthetic data using the specified conditions.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        metadata (SingleTableMetadata): Metadata describing the table schema.
        conditions (list[Condition]): List of conditions specifying how many rows to sample
                                      and what column values to enforce.
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    synthesizer = CopulaGANSynthesizer(metadata)
    synthesizer.fit(train_data)
    synthetic_data = synthesizer.sample_from_conditions(conditions=conditions)
    synthetic_data.to_csv(save_path / "CopulaGAN.csv", index=False)


def run_gaussian_copula(
    train_data: pd.DataFrame,
    metadata: SingleTableMetadata,
    conditions: list[Condition],
    save_path: Path,
) -> None:
    """Train a Gaussian Copula model on the given data, then sample synthetic data using the specified conditions.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        metadata (SingleTableMetadata): Metadata describing the table schema.
        conditions (list[Condition]): List of conditions specifying how many rows to sample
                                      and what column values to enforce.
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(train_data)
    synthetic_data = synthesizer.sample_from_conditions(conditions=conditions)
    synthetic_data.to_csv(save_path / "GaussianCopula.csv", index=False)


def run_tvae(
    train_data: pd.DataFrame,
    metadata: SingleTableMetadata,
    conditions: list[Condition],
    save_path: Path,
) -> None:
    """Train a TVAE model on the given data, then sample synthetic data using the specified conditions.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        metadata (SingleTableMetadata): Metadata describing the table schema.
        conditions (list[Condition]): List of conditions specifying how many rows to sample
                                      and what column values to enforce.
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(train_data)
    synthetic_data = synthesizer.sample_from_conditions(conditions=conditions)
    synthetic_data.to_csv(save_path / "TVAE.csv", index=False)


def run_gaussian_multivariate(
    train_data: pd.DataFrame,
    num_sample: int,
    save_path: Path,
) -> None:
    """Train a GaussianMultivariateSynthesizer and generate `num_sample` synthetic rows.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        num_sample (int): Number of synthetic rows to generate.
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    synthesizer = GaussianMultivariateSynthesizer(train_data, save_path)
    synthesizer.synthesize(num_sample=num_sample)


def run_wgan(
    train_data: pd.DataFrame,
    num_sample: int,
    id_column: str,
    save_path: Path,
) -> None:
    """Train a WGAN model on the given data and generate `num_sample` synthetic rows.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        num_sample (int): Number of synthetic rows to generate.
        id_column (str): Name of the column containing unique IDs (used by the preprocessor).
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    preprocessor = DataImputationPreprocessor(train_data, id_column)
    processed_data = preprocessor.fit_transform()

    synthesizer = WGAN(
        n_features=processed_data.shape[1],
        base_nodes=256,
        batch_size=64,
        critic_iters=10,
        lambda_gp=10.0,
    )

    dataset = data_batcher(processed_data, batch_size=64)

    if len(train_data) <= SMALL_DATASET_THRESHOLD:
        epochs = 10_000
    if SMALL_DATASET_THRESHOLD < len(train_data) <= MEDIUM_DATASET_THRESHOLD:
        epochs = 30_000
    if MEDIUM_DATASET_THRESHOLD < len(train_data) <= LARGE_DATASET_THRESHOLD:
        epochs = 50_000
    if len(train_data) > LARGE_DATASET_THRESHOLD:
        epochs = 100_000

    synthesizer.train(dataset, num_epochs=epochs, log_interval=int(epochs / 5), log_training=True)

    samples = synthesizer.generate_samples(num_sample)
    synthetic_data = pd.DataFrame(samples, columns=processed_data.columns)
    decoded_synthetic_data = preprocessor.inverse_transform(synthetic_data)
    decoded_synthetic_data.to_csv(save_path / "WGAN.csv", index=False)


def run_arf(
    train_data: pd.DataFrame,
    id_column: str,
    num_sample: int,
    save_path: Path,
) -> None:
    """Train an ARF (Augmented Random Forest) model on the given data and generate `num_sample` synthetic rows.

    Args:
        train_data (pd.DataFrame): The real training data used to fit the model.
        id_column (str): Name of the column containing unique IDs (used by the ARF model).
        num_sample (int): Number of synthetic rows to generate.
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None
    """
    model = ARF(x=train_data, id_column=id_column, min_node_size=5, num_trees=50, max_features=0.35)
    _ = model.forde()
    synthetic_data = model.forge(n=num_sample)
    synthetic_data.to_csv(save_path / "ARF.csv", index=False)


MODEL_RUNNERS: dict[str, tuple] = {
    "CTGAN": (
        run_ctgan,
        ["train_data", "metadata", "conditions", "save_path"],
    ),
    "CopulaGAN": (
        run_copulagan,
        ["train_data", "metadata", "conditions", "save_path"],
    ),
    "GaussianCopula": (
        run_gaussian_copula,
        ["train_data", "metadata", "conditions", "save_path"],
    ),
    "TVAE": (
        run_tvae,
        ["train_data", "metadata", "conditions", "save_path"],
    ),
    "GaussianMultivariate": (
        run_gaussian_multivariate,
        ["train_data", "num_sample", "save_path"],
    ),
    "WGAN": (
        run_wgan,
        ["train_data", "num_sample", "id_column", "save_path"],
    ),
    "ARF": (
        run_arf,
        ["train_data", "id_column", "num_sample", "save_path"],
    ),
}


def run_model(  # noqa: PLR0913
    model_name: str,
    train_data: pd.DataFrame,
    metadata: SingleTableMetadata | None,
    conditions: list[Condition] | None,
    id_column: str,
    num_sample: int,
    save_path: Path,
) -> None:
    """Run a single synthetic data model by name.

    Args:
        model_name (str): Name of the model to run. Must be one of the keys in MODEL_RUNNERS.
        train_data (pd.DataFrame): The real training data used to fit the model.
        metadata (SingleTableMetadata | None): Metadata describing the table schema (may be None for
                                               models that do not require it).
        conditions (List[Condition] | None): Conditions specifying how many rows to sample and
                                             what values to enforce (may be None for models that
                                             do not require it).
        id_column (str): Name of the column containing unique IDs (used for certain models).
        num_sample (int): Number of synthetic rows to generate (for models that need it).
        save_path (Path): Directory to which the generated synthetic data CSV is saved.

    Returns:
        None

    Raises:
        ValueError: If the provided `model_name` is not supported.
    """
    if model_name not in MODEL_RUNNERS:
        msg = f"Model {model_name} is not supported."
        raise ValueError(msg)

    model_func, arg_names = MODEL_RUNNERS[model_name]
    args = {
        "train_data": train_data,
        "metadata": metadata,
        "conditions": conditions,
        "id_column": id_column,
        "save_path": save_path,
        "num_sample": num_sample,
    }
    selected_args = {k: args[k] for k in arg_names}
    model_func(**selected_args)
