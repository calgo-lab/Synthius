from typing import List, Protocol, Optional, runtime_checkable

import pandas as pd
from sdv.metadata import SingleTableMetadata

@runtime_checkable
class Synthesizer(Protocol):
    """Any synthetic data model should implement this interface."""
    metadata: Optional[SingleTableMetadata]
    name: str  # unique name for saving/output

    def fit(self, train_data: pd.DataFrame) -> None:
        ...
    
    def generate(self, total_samples: int, conditions: list = None) -> pd.DataFrame:
        ...
