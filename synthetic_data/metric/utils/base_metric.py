from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseMetric(ABC):
    """Abstract base class for metric classes."""

    @abstractmethod
    def display_results(self: BaseMetric) -> pd.DataFrame:
        """This method should return a DataFrame containing the results of the metric evaluations.

        Must be implemented by all subclasses to ensure they produce compatible output for aggregation.
        """
