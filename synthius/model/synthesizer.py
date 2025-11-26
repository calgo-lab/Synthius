from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Synthesizer(Protocol):
    """Interface for all synthetic data generators used in Synthius.

    This protocol defines the minimal contract that any tabular synthetic
    data model must implement to be compatible with the Synthius pipeline.

    Attributes:
    ----------
    name : str
        Human-readable name identifying the synthesizer. Used for saving
        outputs and experiment tracking.

    Methods:
    -------
    fit(train_data)
        Train or initialize the synthesizer on the provided dataset.
    generate(total_samples, conditions)
        Generate synthetic samples from the fitted model.
    """

    name: str

    def fit(self, train_data: pd.DataFrame) -> None:
        """Train the synthesizer on a provided dataset.

        Parameters:
        ----------
        train_data : pd.DataFrame
            The tabular dataset to use for fitting the model. Columns may be
            numeric, categorical, or mixed. Implementations are free to
            preprocess the data internally.

        Returns:
        -------
        None
            The method modifies the synthesizer's internal state but does not
            return a value.
        """
        ...

    def generate(self, total_samples: int, conditions: list | None = None) -> pd.DataFrame:
        """Generate synthetic data samples based on the fitted model.

        Parameters:
        ----------
        total_samples : int
            The number of synthetic rows to generate.
        conditions : list, optional
            Optional conditional constraints to guide generation. For example,
            fixing certain column values. If not supported by the synthesizer,
            this can be ignored.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the synthetic samples. The schema should
            generally match the training data unless the model intentionally
            modifies it.
        """
        ...
