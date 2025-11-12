from typing import List, Optional

import pandas as pd
from sdv.metadata import SingleTableMetadata
from .synthesizer import Synthesizer


class SDVSynthesizer(Synthesizer):
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
