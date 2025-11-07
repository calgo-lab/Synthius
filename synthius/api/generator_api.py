# The API for the generation of synthetic datasets.
# 2025.10.29
import warnings
from pathlib import Path
from typing import List, Tuple

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
from synthius.model import GaussianMultivariateSynthesizer, ARF, WGAN
from synthius.data import DataImputationPreprocessor

def preprocess_data(
    original_data_filename: str,
    data_dir: str,
    target_column: str | int,
    random_seed: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, split, preprocess, and save the data.

    Args:
        original_data_filename (str): Name of the CSV file containing the dataset.
        data_dir (str): Directory where the data file is located.
        target_column (str | int): The name or index of the target column for stratification.
        random_seed (int | None): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The full dataset,
            the training split, and the testing split.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the target column does not exist in the dataset.
    """
    data_path = Path(data_dir)
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

    train_data.to_csv(data_path / "train.csv", index=False)
    test_data.to_csv(data_path / "test.csv", index=False)

    return data, train_data, test_data


def run_model(
    model: str,
    train_data: pd.DataFrame,
    total_samples: int,
    true_condition: Condition,
    false_condition: Condition,
    metadata: SingleTableMetadata,
    results_dir: str,
    id_column: str | int | None,
) -> None:
    """Train and execute a synthetic data generation model, then save the results.

    This function fits the specified synthetic data model on the provided training data,
    generates synthetic samples according to the given conditions, and writes the output
    to the results directory.

    Args:
        model (str): The name of the synthesizer to run. Supported models include
            ``"CopulaGAN"``, ``"CTGAN"``, ``"GaussianCopula"``, ``"TAVE"``,
            ``"GaussianMultivariate"``, ``"ARF"``, and ``"WGAN"``.
        train_data (pd.DataFrame): The input training dataset used to fit the model.
        total_samples (int): Total number of synthetic samples to generate.
        true_condition (Condition): Sampling condition for the positive or target class.
        false_condition (Condition): Sampling condition for the negative or non-target class.
        metadata (SingleTableMetadata): Table metadata schema describing the input data.
        results_dir (str): Path to the directory where generated datasets will be saved.
        id_column (str | int | None): Optional identifier column for models that require one.

    Raises:
        ValueError: If an unsupported or unknown model name is provided.
        Exception: For unexpected runtime errors during model training or sampling.
    """
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    try:
        if model == "CopulaGAN":
            # handle CopulaGAN
            copula_gan_synthesizer = CopulaGANSynthesizer(metadata)
            copula_gan_synthesizer.fit(train_data)
            copula_gan_synthetic_data = copula_gan_synthesizer.sample_from_conditions(
                conditions=[true_condition, false_condition]
            )
            copula_gan_synthetic_data.to_csv(results_path / "CopulaGAN.csv", index=False)

        elif model == "CTGAN":
            # handle CTGAN
            ctgan_synthesizer = CTGANSynthesizer(metadata)
            ctgan_synthesizer.fit(train_data)
            ctgan_synthetic_data = ctgan_synthesizer.sample_from_conditions(
                conditions=[true_condition, false_condition]
            )
            ctgan_synthetic_data.to_csv(results_path / "CTGAN.csv", index=False)

        elif model == "GaussianCopula":
            # handle GaussianCopula
            gaussian_copula_synthesizer = GaussianCopulaSynthesizer(metadata)
            gaussian_copula_synthesizer.fit(train_data)
            gaussian_copula_synthetic_data = gaussian_copula_synthesizer.sample_from_conditions(
                conditions=[true_condition, false_condition],
            )
            gaussian_copula_synthetic_data.to_csv(results_path / "GaussianCopula.csv", index=False)

        elif model == "TVAE":
            # handle TVAE
            tvae_synthesizer = TVAESynthesizer(metadata)
            tvae_synthesizer.fit(train_data)
            tvae_synthetic_data = tvae_synthesizer.sample_from_conditions(
                conditions=[true_condition, false_condition]
            )
            tvae_synthetic_data.to_csv(results_path / "TVAE.csv", index=False)

        elif model == "GaussianMultivariate":
            # handle GaussianMultivariate
            gaussian_multivariate_synthesizer = GaussianMultivariateSynthesizer(train_data, results_path)
            gaussian_multivariate_synthesizer.synthesize(num_sample=total_samples)

        elif model == "ARF":
            # handle ARF
            model = ARF(
                x=train_data,
                id_column=id_column,
                min_node_size=5,
                num_trees=50,
                max_features=0.3
            )
            _ = model.forde()
            synthetic_data_arf = model.forge(n=total_samples)
            synthetic_data_arf.to_csv(results_path / "ARF.csv", index=False)

        elif model == "WGAN":
            preprocessor = DataImputationPreprocessor(train_data)
            processed_data = preprocessor.fit_transform()

            n_features = processed_data.shape[1]
            wgan = WGAN(
                n_features=n_features,
                base_nodes=128,
                batch_size=512,
                critic_iters=5,
                lambda_gp=10.0,
                num_epochs=100000,  # There is a bug here
            )
            wgan.train(processed_data, log_interval=5000, log_training=True)  # Also here if the num_epochs is omitted...

            synthetic_samples = wgan.generate_samples(total_samples)
            synthetic_df = pd.DataFrame(synthetic_samples, columns=processed_data.columns)
            decoded_df = preprocessor.inverse_transform(synthetic_df)
            decoded_df.to_csv(results_path / "WGAN.csv", index=False)


        else:
            raise ValueError(f"Unknown model: {model}")

    except ValueError as e:
        print(f"[Error] {e}")
    except Exception as e:
        print(f"[Unexpected Error] {type(e).__name__}: {e}")


def generate(
    original_data_filename: str,
    target_column: str | int,
    data_dir: str = ".",
    synth_dir: str = ".",
    id_column: str | int | None = None,
    models: List[str] | None = None,
    random_seed: int | None = None,
) -> None:
    """Generate synthetic datasets from a source dataset using one or more models.

    This function reads a dataset, preprocesses it into training and test splits,
    automatically detects its metadata, constructs sampling conditions for the
    target variable, and then runs one or more specified synthetic data generation
    models. Each model's generated dataset is saved to the results directory.

    Args:
        original_data_filename (str): Name of the source data file to read.
        data_dir (str): Directory containing the input data file.
            Defaults to the current working directory.
        synth_dir (str): Directory where generated synthetic datasets
            will be stored. Created if it does not exist.
        target_column (str | int): Column name or index representing the target variable
            used for conditional sampling and stratified splitting.
        id_column (str | int | None): Optional identifier column used by certain models
            (e.g., ARF) for entity-based synthesis.
        models (List[str] | None): List of model names to execute. If ``None``,
            all built-in models will be run, including:
            ``["CopulaGAN", "CTGAN", "GaussianCopula", "TAVE", "GaussianMultivariate", "ARF", "WGAN"]``.
        random_seed (int | None): Random seed for reproducibility of train-test splits.

    Raises:
        FileNotFoundError: If the specified data file cannot be found.
        ValueError: If the target column has fewer than two unique classes.
        Exception: For unexpected runtime errors during data loading or synthesis.
    """
    data, train_data, test_data = preprocess_data(
        original_data_filename, data_dir, target_column, random_seed
    )

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)

    total_samples = train_data.shape[0]
    unique_classes = data[target_column].unique()

    if len(unique_classes) > 2:
        category_counts = train_data[target_column].value_counts()
        print(category_counts)
        target_a = int(category_counts.get(1, 0))
        target_b = int(category_counts.get(0, 0))
        true_condition = Condition(num_rows=target_a, column_values={target_column: 1})
        false_condition = Condition(num_rows=target_b, column_values={target_column: 0})

    elif len(unique_classes) == 2:
        true_samples = train_data[target_column].sum()
        false_samples = total_samples - true_samples
        true_condition = Condition(
            num_rows=true_samples, column_values={target_column: True}
        )
        false_condition = Condition(
            num_rows=false_samples, column_values={target_column: False}
        )

    else:
        raise ValueError("The target column must contain at least two classes.")

    # Run all models if none are specified by the user
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

    for model_name in models:
        run_model(
            model_name,
            train_data,
            total_samples,
            true_condition,
            false_condition,
            metadata,
            synth_dir,
            id_column,
        )

# For testing purposes
if __name__ == "__main__":
    generate(
        original_data_filename="iris_setosa_vs_all.csv",
        data_dir="/storage/Synthius/examples/data/",
        synth_dir="/storage/Synthius/examples/synthetic_data/",
        target_column="target_binary",
        models=["CopulaGAN", "CTGAN", "GaussianCopula", "TVAE", "GaussianMultivariate", "ARF"], # Omit WGAN due to deep bugs
        random_seed=42
    )
