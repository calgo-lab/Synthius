from typing import List, Optional

import pandas as pd
from .synthesizer import Synthesizer

class ARFSynthesizer(Synthesizer):
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

class SynthesizerGaussianMultivariate(Synthesizer):
    def __init__(self, results_path: str):
        self.results_path = results_path
        self.model = None
        self.name = "GaussianMultivariate"

    def fit(self, train_data: pd.DataFrame):
        self.model = GaussianMultivariateSynthesizer(train_data, self.results_path)

    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        self.model.synthesize(num_sample=total_samples)
        return pd.read_csv(self.model.output_file)


class WGANSynthesizer(Synthesizer):
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
