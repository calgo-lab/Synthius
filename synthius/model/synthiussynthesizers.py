import pandas as pd
from Synthius.synthius.data.data_imputer import DataImputationPreprocessor
from Synthius.synthius.model.arf import ARF
from Synthius.synthius.model.gaussian_multivariate import GaussianMultivariateSynthesizer
from Synthius.synthius.model.wgan import WGAN

from .synthesizer import Synthesizer


class ARFSynthesizer(Synthesizer):
    """Tabular data synthesizer using an ARF (Adaptive Random Forest) model."""

    def __init__(self, id_column: str | None = None) -> None:
        """Initialize the ARFSynthesizer.

        Parameters:
            id_column : str | None, optional
                Column name to treat as ID. Default is None.
        """
        self.id_column = id_column
        self.model: ARF | None = None
        self.train_data: pd.DataFrame | None = None
        self.name = "ARF"

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the ARF model to training data.

        Parameters:
            train_data : pd.DataFrame
                Tabular dataset to train the ARF model.
        """
        self.train_data = train_data
        self.model = ARF(
            x=train_data,
            id_column=self.id_column,
            min_node_size=5,
            num_trees=50,
            max_features=0.3,
        )
        self.model.forde()

    def generate(self, total_samples: int, conditions: list | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Generate synthetic samples from the fitted ARF model.

        Parameters:
            total_samples : int
                Number of synthetic rows to generate.
            conditions : list | None, optional
                Currently ignored; included for compatibility with the Synthesizer protocol.

        Returns:
            pd.DataFrame
                Synthetic samples as a DataFrame.
        """
        return self.model.forge(n=total_samples)


class SynthesizerGaussianMultivariate(Synthesizer):
    """Tabular data synthesizer using a Gaussian multivariate model."""

    def __init__(self, results_path: str) -> None:
        """Initialize the Gaussian multivariate synthesizer.

        Parameters:
            results_path : str
                Path to save model outputs.
        """
        self.results_path = results_path
        self.model: GaussianMultivariateSynthesizer | None = None
        self.name = "GaussianMultivariate"

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the Gaussian multivariate model to training data.

        Parameters:
            train_data : pd.DataFrame
                Tabular dataset to train the model.
        """
        self.model = GaussianMultivariateSynthesizer(train_data, self.results_path)

    def generate(self, total_samples: int, conditions: list | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Generate synthetic samples from the fitted model.

        Parameters:
            total_samples : int
                Number of synthetic rows to generate.
            conditions : list | None, optional
                Currently ignored; included for compatibility with the Synthesizer protocol.

        Returns:
            pd.DataFrame
                Synthetic samples as a DataFrame.
        """
        self.model.synthesize(num_sample=total_samples)
        return pd.read_csv(self.model.output_file)


class WGANSynthesizer(Synthesizer):
    """Tabular data synthesizer using a WGAN (Wasserstein GAN) model."""

    def __init__(self) -> None:
        """Initialize the WGANSynthesizer."""
        self.model: WGAN | None = None
        self.preprocessor: DataImputationPreprocessor | None = None
        self.processed_data: pd.DataFrame | None = None
        self.name = "WGAN"

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the WGAN model to training data.

        Parameters:
            train_data : pd.DataFrame
                Tabular dataset to train the WGAN model.
        """
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

    def generate(self, total_samples: int, conditions: list | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Generate synthetic samples from the fitted WGAN model.

        Parameters:
            total_samples : int
                Number of synthetic rows to generate.
            conditions : list | None, optional
                Currently ignored; included for compatibility with the Synthesizer protocol.

        Returns:
            pd.DataFrame
                Synthetic samples as a DataFrame with preprocessing reversed.
        """
        samples = self.model.generate_samples(total_samples)
        data = pd.DataFrame(samples, columns=self.processed_data.columns)
        return self.preprocessor.inverse_transform(data)
