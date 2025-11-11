# The API for the generation of synthetic datasets.
# 2025.10.29
import warnings
from pathlib import Path
from typing import List, Tuple, Protocol, Optional, runtime_checkable

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


@runtime_checkable
class Synthesizer(Protocol):
    """Any synthetic data model should implement this interface."""
    metadata: Optional[SingleTableMetadata]
    name: str  # unique name for saving/output

    def fit(self, train_data: pd.DataFrame) -> None:
        ...
    
    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        ...


class SDVSynthWrapper(Synthesizer):
    def __init__(self, cls, metadata: Optional[SingleTableMetadata] = None):
        self.model_class = cls
        self.model = None
        self.metadata = metadata
        self.name = cls.__name__

    def fit(self, train_data: pd.DataFrame):
        if self.metadata is None:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(train_data)
        self.model = self.model_class(self.metadata)
        self.model.fit(train_data)

    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model must be fit() before generate()")
        return self.model.sample_from_conditions(conditions)


class ARFSynthWrapper(Synthesizer):
    def __init__(self, id_column=None):
        self.id_column = id_column
        self.model = None
        self.train_data = None
        self.name = "ARF"

    def fit(self, train_data: pd.DataFrame):
        self.train_data = train_data
        self.model = ARF(
            x=train_data,
            id_column=self.id_column,
            min_node_size=5,
            num_trees=50,
            max_features=0.3
        )
        self.model.forde()

    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        return self.model.forge(n=total_samples)


class GaussianMultivariateWrapper(Synthesizer):
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.model = None
        self.name = "GaussianMultivariate"

    def fit(self, train_data: pd.DataFrame):
        self.model = GaussianMultivariateSynthesizer(train_data, self.results_path)

    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        self.model.synthesize(num_sample=total_samples)
        return pd.read_csv(self.model.output_file)


class WGANWrapper(Synthesizer):
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.processed_data = None
        self.name = "WGAN"

    def fit(self, train_data: pd.DataFrame):
        self.preprocessor = DataImputationPreprocessor(train_data)
        self.processed_data = self.preprocessor.fit_transform()
        n_features = self.processed_data.shape[1]
        self.model = WGAN(
            n_features=n_features,
            base_nodes=128,
            batch_size=512,
            critic_iters=5,
            lambda_gp=10.0,
            num_epochs=100000,
        )
        self.model.train(self.processed_data, log_interval=5000, log_training=True)

    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        samples = self.model.generate_samples(total_samples)
        df = pd.DataFrame(samples, columns=self.processed_data.columns)
        return self.preprocessor.inverse_transform(df)


def preprocess_data(
    original_data_filename: str,
    data_dir: str,
    target_column: str | int,
    random_seed: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read, split, preprocess, and save the data."""
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

    return data, train_data, test_data


def run_model(
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


def generate(
    original_data_filename: str,
    target_column: str | int,
    data_dir: str = ".",
    synth_dir: str = ".",
    id_column: str | int | None = None,
    models: list[Synthesizer] | None = None,
    random_seed: int | None = None,
) -> None:
    """Generate synthetic datasets from a source dataset using Synthesizer instances."""
    data, train_data, test_data = preprocess_data(
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
            SDVSynthWrapper(CopulaGANSynthesizer, metadata),
            SDVSynthWrapper(CTGANSynthesizer, metadata),
            SDVSynthWrapper(GaussianCopulaSynthesizer, metadata),
            SDVSynthWrapper(TVAESynthesizer, metadata),
            GaussianMultivariateWrapper(results_path=synth_dir),
            ARFSynthWrapper(id_column=id_column),
            WGANWrapper(),
        ]

    # Run each model
    for model_instance in models:
        run_model(model_instance, train_data, total_samples, synth_dir, conditions=conditions)


# For testing purposes
if __name__ == "__main__":
    models = [
        SDVSynthWrapper(CopulaGANSynthesizer),
        SDVSynthWrapper(CTGANSynthesizer),
        SDVSynthWrapper(GaussianCopulaSynthesizer),
        SDVSynthWrapper(TVAESynthesizer),
        GaussianMultivariateWrapper(results_path="/storage/Synthius/examples/synthetic_data/"),
        ARFSynthWrapper(id_column=None),
        # WGANWrapper()  # It's broken rn though
    ]
    generate(
        original_data_filename="iris_setosa_vs_all.csv",
        data_dir="/storage/Synthius/examples/data/",
        synth_dir="/storage/Synthius/examples/synthetic_data/",
        target_column="target_binary",
        models=models,
        random_seed=42
    )
