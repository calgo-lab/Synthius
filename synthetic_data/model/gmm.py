from __future__ import annotations

from abc import ABC
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


class BaseProcessor(ABC, BaseEstimator, TransformerMixin):
    """Implements a basic data processor compatible with scikit-learn's transformer interface.

    This class serves as an abstract base for data processors, facilitating the
    integration of custom preprocessing pipelines within scikit-learn's framework.
    It requires subclasses to define a specific pipeline for data processing.
    """

    def __init__(self: BaseProcessor) -> None:
        """Initializes the BaseProcessor."""
        self._pipeline: Pipeline | None = None
        self._col_transform_info: SimpleNamespace | None = None
        self._types: pd.Series | None = None

    @property
    def pipeline(self: BaseProcessor) -> Pipeline | None:
        """Retrieves the pipeline for processing columns.

        Returns:
            The pipeline for processing, if defined.
        """
        return self._pipeline

    @property
    def types(self: BaseProcessor) -> pd.Series | None:
        """Provides the data types of each column in the fitted DataFrame.

        Returns:
            A pandas Series containing the data types of the columns.
        """
        return self._types

    @property
    def col_transform_info(self: BaseProcessor) -> SimpleNamespace:
        """Retrieves metadata about the transformations applied by this processor.

        This includes information on input and output features for the processing pipeline.

        Returns:
            An instance of SimpleNamespace with detailed transformation metadata.
        """
        self._check_is_fitted()
        if self._col_transform_info is None:
            self._col_transform_info = self.__create_metadata_synth()
        return self._col_transform_info

    def __create_metadata_synth(self: BaseProcessor) -> SimpleNamespace:
        """Generates metadata for tracking input/output feature mappings.

        Returns:
            A SimpleNamespace object containing detailed mappings.
        """

        def new_pipeline_info(feat_in: list[str], feat_out: list[str]) -> SimpleNamespace:
            return SimpleNamespace(feat_names_in=feat_in, feat_names_out=feat_out)

        if self._pipeline is not None:
            info = new_pipeline_info(
                self._pipeline.feature_names_in_,
                self._pipeline.get_feature_names_out(),
            )
        else:
            info = new_pipeline_info([], [])

        return SimpleNamespace(features=info)

    def _check_is_fitted(self: BaseProcessor) -> None:
        """Validates if the processor has been fitted.

        Raises:
            NotFittedError: If the processor has not been fitted yet.
        """
        if self._pipeline is None:
            message = "This data processor has not yet been fitted."
            raise NotFittedError(message)


class RegularDataProcessor(BaseProcessor):
    """Enhances BaseProcessor for regular or tabular data preprocessing.

    This class builds upon the BaseProcessor by implementing specific fit, transform,
    and inverse_transform methods tailored for regular or tabular data sets, making it
    compatible with scikit-learn's transformer workflow.
    """

    def __init__(self: RegularDataProcessor) -> None:
        """Initializes RegularDataProcessor."""
        super().__init__()
        self._col_order_: list[str] | None = None

    def fit(self: RegularDataProcessor, X: pd.DataFrame) -> RegularDataProcessor:  # noqa: N803
        """Fits the processor to a DataFrame, preparing the pipeline for transformation.

        Args:
            X (pd.DataFrame): The DataFrame used to fit the processor.

        Returns:
            RegularDataProcessor: The fitted data processor instance.
        """
        self._types = X.dtypes
        self._col_order_ = list(X.columns)

        self._pipeline = Pipeline([("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])
        self._pipeline = self._pipeline.fit(X)

        return self

    def transform(self: RegularDataProcessor, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Transforms the given DataFrame using the fitted pipeline.

        Args:
            X (pd.DataFrame): DataFrame to be transformed.

        Returns:
            np.ndarray: The transformed data as a NumPy array.
        """
        self._check_is_fitted()
        return self._pipeline.transform(X) if self._pipeline is not None else np.zeros((len(X), 0))

    def inverse_transform(self: RegularDataProcessor, X: np.ndarray) -> pd.DataFrame:  # noqa: N803
        """Reverses the transformations applied to the data.

        Args:
            X (np.ndarray): The transformed data to revert.

        Returns:
            pd.DataFrame: The original data after reversing the transformations.
        """
        self._check_is_fitted()

        data = self._pipeline.inverse_transform(X) if self._pipeline else pd.DataFrame()

        result = pd.DataFrame(data, columns=self._col_order_)
        return result.loc[:, self._col_order_].astype(self._types)


class GMM(BaseEstimator):
    """Implements Gaussian Mixture Modeling for data synthesis.

    This class encapsulates the process of fitting a Gaussian Mixture Model to data
    for the purpose of generating synthetic datasets that mimic the statistical
    properties of the input data.

    Attributes:
        covariance_type (str): The type of covariance parameters to use.
        random_state (int): The seed used by the random number generator.
    """

    def __init__(self: GMM, covariance_type: str = "tied", random_state: int = 0) -> None:
        """Initializes the GMM synthesizer with specified configuration.

        Args:
            covariance_type (str): Type of covariance to use, e.g., 'full', 'tied'.
            random_state (int): Seed for the random number generator.
        """
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianMixture(covariance_type=covariance_type, random_state=random_state)
        self.processor: RegularDataProcessor = RegularDataProcessor()

    def _optimize(self: GMM, prep_data: np.ndarray) -> int:
        """Determines the optimal number of components for the GMM.

        Args:
            prep_data (np.ndarray): Preprocessed data for optimization.

        Returns:
            int: The optimal number of components determined.
        """
        best_n_components = 2
        max_silhouette = float("-inf")
        for n in tqdm(range(2, 40, 5), desc="Optimizing GMM components"):
            model = GaussianMixture(n, covariance_type=self.covariance_type, random_state=self.random_state)
            labels = model.fit_predict(prep_data)
            silhouette = silhouette_score(prep_data, labels, metric="euclidean")
            if model.converged_ and silhouette > max_silhouette:
                best_n_components = n
                max_silhouette = silhouette
        return best_n_components

    def fit(self: GMM, data: pd.DataFrame | np.ndarray) -> GMM:
        """Fits the GMM synthesizer to the provided dataset.

        Args:
            data (pd.DataFrame | np.ndarray): The dataset to fit the model to.

        Returns:
            GMM: The fitted GMM synthesizer instance.
        """
        if isinstance(data, pd.DataFrame):
            self.processor = RegularDataProcessor().fit(data)
            data = self.processor.transform(data)
        n_components = self._optimize(data)
        self.model.n_components = n_components
        self.model.fit(data)
        return self

    def sample(self: GMM, n_samples: int) -> pd.DataFrame:
        """Generates synthetic samples using the fitted GMM.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            pd.DataFrame: The generated synthetic samples.
        """
        samples = self.model.sample(n_samples=n_samples)[0]
        return self.processor.inverse_transform(samples)
