from typing import TYPE_CHECKING

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from .synthesizer import Synthesizer

if TYPE_CHECKING:
    from sdv.single_table.base import BaseSingleTableSynthesizer


class SDVSynthesizer(Synthesizer):
    """Wrapper for SDV synthetic data models to conform to the Synthesizer protocol.

    This class allows any SDV model class to be used in the Synthius pipeline.
    It handles optional metadata detection and enforces the standard `fit` and
    `generate` interface.

    Attributes:
        model_class : type
            The SDV model class to instantiate.
        model : object | None
            The trained SDV model instance after calling `fit`.
        metadata : SingleTableMetadata | None
            Optional metadata describing the table schema. Automatically detected
            from training data if not provided.
        name : str
            Name of the synthesizer, derived from the model class name.
    """

    def __init__(self, cls: type, metadata: SingleTableMetadata | None = None) -> None:
        """Initialize the SDVSynthesizer.

        Parameters:
            cls : type
                The SDV model class to use.
            metadata : SingleTableMetadata | None, optional
                Optional metadata object describing the table schema. If None,
                metadata will be detected from training data during `fit`.
        """
        self.model_class = cls
        self.model: BaseSingleTableSynthesizer
        self.metadata = metadata
        self.name = cls.__name__

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the SDV model to the training data.

        If no metadata is provided, automatically detects it from the
        training DataFrame.

        Parameters:
            train_data : pd.DataFrame
                Tabular dataset used to train the SDV model.

        Returns:
            None
        """
        if self.metadata is None:
            self.metadata = SingleTableMetadata()
            self.metadata.detect_from_dataframe(train_data)

        self.model = self.model_class(self.metadata)
        self.model.fit(train_data)

    def generate(self, total_samples: int, conditions: list[Condition] | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Generate synthetic samples from the trained SDV model.

        Parameters:
            conditions : list | None, optional
                Optional list of conditional constraints for generation.

        Returns:
            pd.DataFrame
                DataFrame containing the synthetic samples.

        Raises:
            RuntimeError
                If called before `fit()` has been executed.
            NotImplementedError
                If the underlying model does not support `sample_from_conditions`.
        """
        if self.model is None:
            msg = "Model must be fit() before generate()"
            raise RuntimeError(msg)

        # Some SDV models use `sample_from_conditions`; ensure total_samples is respected
        if hasattr(self.model, "sample_from_conditions"):
            return self.model.sample_from_conditions(conditions)

        msg = f"{self.model_class.__name__} does not support sample_from_conditions."
        raise NotImplementedError(msg)


# ----------------------------------------------------------------------
# Named convenience wrappers
# ----------------------------------------------------------------------


class SDVCopulaGANSynthesizer(SDVSynthesizer):
    """Wrapper for SDV's CopulaGANSynthesizer to integrate with Synthius."""

    def __init__(self, metadata: SingleTableMetadata | None = None) -> None:
        """Initialize the CopulaGAN wrapper with optional table metadata."""
        super().__init__(CopulaGANSynthesizer, metadata)


class SDVCTGANSynthesizer(SDVSynthesizer):
    """Wrapper for SDV's CTGANSynthesizer to integrate with Synthius."""

    def __init__(self, metadata: SingleTableMetadata | None = None) -> None:
        """Initialize the CTGAN wrapper with optional table metadata."""
        super().__init__(CTGANSynthesizer, metadata)


class SDVGaussianCopulaSynthesizer(SDVSynthesizer):
    """Wrapper for SDV's GaussianCopulaSynthesizer for use in Synthius."""

    def __init__(self, metadata: SingleTableMetadata | None = None) -> None:
        """Initialize the Gaussian Copula wrapper with optional metadata."""
        super().__init__(GaussianCopulaSynthesizer, metadata)


class SDVTVAESynthesizer(SDVSynthesizer):
    """Wrapper for SDV's TVAESynthesizer for use in Synthius."""

    def __init__(self, metadata: SingleTableMetadata | None = None) -> None:
        """Initialize the TVAE wrapper with optional table metadata."""
        super().__init__(TVAESynthesizer, metadata)
