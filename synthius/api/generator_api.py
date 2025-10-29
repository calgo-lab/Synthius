# The API for the generation of synthetic datasets.
# 2025.10.29
import warnings
from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from sklearn.model_selection import train_test_split
from synthius.model import GaussianMultivariateSynthesizer, ARF, WGAN
from synthius.data import DataImputationPreprocessor

def preprocess_data(
    original_data_filename: str,
    data_dir: str,
    target_column: str | int,
    random_seed: int | None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, split, preprocess, and save the data

    Args:

    Returns:

    Raises:

    """
    # Read in data
    data_path = Path(data_dir)
    data = pd.read_csv(data_path / original_data_filename, low_memory=False)

    # ERROR CHECKING!

    # Train-Test Split
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=random_seed,
        stratify=data[target_column],
    )

    # Save TTS data
    train_data.to_csv(data_path / "train.csv", index=False)
    test_data.to_csv(data_path / "test.csv", index=False)

    return data, train_data, test_data


def run_model(
    model: str,
    total_samples: int,
    true_condition: Condition,
    false_condition: Condition,
    metadata: Metadata,
    results_dir:str,
    id_column: str | int | None
    ):
    """Run the synthetic generation models and save the results

    Args:

    Returns:

    Raises:

    """
    # Create the directory if it doesn't exist
    try:
        if model == "CopulaGAN":
            # handle CopulaGAN
            copula_gan_synthesizer = CopulaGANSynthesizer(metadata)
            copula_gan_synthesizer.fit(train_data)
            copula_gan_synthetic_data = copula_gan_synthesizer.sample_from_conditions(conditions=[true_condition, false_condition])
            copula_gan_synthetic_data.to_csv(results_dir / "CopulaGAN.csv", index=False)
        elif model == "CTGAN":
            # handle CTGAN
            ctgan_synthesizer = CTGANSynthesizer(metadata)
            ctgan_synthesizer.fit(train_data)
            ctgan_synthetic_data = ctgan_synthesizer.sample_from_conditions(conditions=[true_condition, false_condition])
            ctgan_synthetic_data.to_csv(results_dir / "CTGAN.csv", index=False)
        elif model == "GaussianCopula":
            # handle GaussianCopula
            gaussian_copula_synthesizer = GaussianCopulaSynthesizer(metadata)
            gaussian_copula_synthesizer.fit(train_data)
            gaussian_copula_synthetic_data = gaussian_copula_synthesizer.sample_from_conditions(
                conditions=[true_condition, false_condition],
            )
            gaussian_copula_synthetic_data.to_csv(results_dir / "GaussianCopula.csv", index=False)
        elif model == "TAVE":
            # handle TAVE
            tvae_synthesizer = TVAESynthesizer(metadata)
            tvae_synthesizer.fit(train_data)
            tvae_synthetic_data = tvae_synthesizer.sample_from_conditions(conditions=[true_condition, false_condition])
            tvae_synthetic_data.to_csv(results_dir / "TVAE.csv", index=False)
        elif model == "GaussianMultivariate":
            # handle GaussianMultivariate
            gaussian_multivariate_synthesizer = GaussianMultivariateSynthesizer(train_data, results_dir)
            gaussian_multivariate_synthesizer.synthesize(num_sample=total_samples)
        elif model == "ARF":
            # handle ARF
            model = ARF(x=train_data, id_column=id_column, min_node_size=5, num_trees=50, max_features=0.3)
            forde = model.forde()
            synthetic_data_arf = model.forge(n=total_samples)

            synthetic_data_arf.to_csv(results_dir / "ARF.csv", index=False)
        elif model == "WGAN":
            # handle WGAN
            data_preprocessor = DataImputationPreprocessor(train_data)
            processed_train_data = data_preprocessor.fit_transform()

            n_features = processed_train_data.shape[1]
            wgan_imputer = WGAN(n_features=n_features, base_nodes=128, batch_size=512, critic_iters=5, lambda_gp=10.0, num_epochs=100000)  # Num_epochs in main branch is an unexpected kwarg - removal yields a nasty bug...
            wgan_imputer.train(processed_train_data, log_interval=5000, log_training=True) # Dataframe is not an iterator...

            wgan_synthetic_samples = wgan_imputer.generate_samples(total_samples)
            wgan_synthetic_data = pd.DataFrame(wgan_synthetic_samples, columns=processed_train_data.columns)

            # --------------------- Decoding --------------------- #
            decoded_wgan_synthetic_data = data_preprocessor.inverse_transform(wgan_synthetic_data)
            # --------------------- Saving   --------------------- #
            decoded_wgan_synthetic_data.to_csv(results_dir / "WGAN.csv", index=False)
        else:
            raise ValueError(f"Unknown model: {model}")
    except ValueError as e:
        print(f"[Error] {e}")
    except Exception as e:
        print(f"[Unexpected Error] {type(e).__name__}: {e}")


def generate(
    original_data_filename: str,
    data_dir: str,
    results_dir: str,
    target_column: str | int,
    id_column = None: str | int | None,
    models = None: List[str] | None,
    random_seed = None: int | None
    ) -> None:
    """Generate synthetic datasets given a source dataset.

    Args:
        
    Returns:

    Raises:

    """
    # Read in the data
    full_data, train_data, test_data = preprocess_data(original_data_filename, data_dir, target_column, random_seed)

    # Create metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data)
    metadata_dict = metadata.to_dict()
    total_samples = train_data.shape[0]

    # Create OVA target
    if len(data[target_column].unique()) > 2:
        # This is necessary so that we have the same label distribution in the syntehtic dataset.
        category_counts = train_data[target_column].value_counts()
        print(category_counts)
        target_a = int(category_counts.get(1, 0))
        target_b = int(category_counts.get(0, 0))

        true_condition = Condition(num_rows=target_a, column_values={target_column: 1})
        false_condition = Condition(num_rows=target_b, column_values={target_column: 0})
    elif len(data[target_column].unique()) == 2:
        true_samples = train_data[target_column].sum()
        false_samples = total_samples - true_samples
        true_condition = Condition(num_rows=true_samples, column_values={target_column: True})
        false_condition = Condition(num_rows=false_samples, column_values={target_column: False})
    else:
        raise ValueError # There must be more than one class in the target column!

    # Run suite of models
    if models == None:  # Use all of the built-in models
        models = ["CopulaGAN", "CTGAN", "GaussianCopula", "TAVE", "GaussianMultivariate", "ARF", "WGAN"]
    # Else use the models supplied by the user

    for model in models:
        run_model(model, total_samples, true_condition, false_condition, metadata, results_dir, id_column)
