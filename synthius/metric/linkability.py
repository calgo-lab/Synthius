from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from anonymeter.evaluators import LinkabilityEvaluator

from synthius.metric.utils import apply_preprocessing, load_data, preprocess_data
from synthius.metric.utils.anonymeter_metric import AnonymeterMetric

logger = logging.getLogger("anonymeter")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


class LinkabilityMetric(AnonymeterMetric):
    """A class to compute `Linkability Risk` for synthetic data compared to real data.

    Adapted from Anonymeter:
    https://github.com/statice/anonymeter

    This involves assessing privacy risks by comparing synthetic datasets against a real dataset and a control dataset.

    This evaluator checks if it's possible to link records from two vertically split parts of the original dataset
    using the synthetic data. It measures the risk of reconnecting these split records based on their closest neighbors
    in the synthetic data.

    Outputs:

    - `Main Attack`:    This uses the synthetic dataset to guess information about records in the original dataset.
                        It's the primary evaluation of how much private information the synthetic data reveals.
    - `Control Attack`: This uses the synthetic dataset to guess information about records in the control dataset
                        (a subset of the original data not used for generating synthetic data). It helps to
                        differentiate between what an attacker learns from the utility of the synthetic data and what
                        indicates actual privacy leakage.
    - `Baseline Attack`: This is a naive attack where guesses are made randomly without using the synthetic data.
                        It serves as a sanity check to ensure that the main attack's success rate is meaningful.

    The `.risk()` method provides an estimate of the privacy risk. For example:

    ```
    PrivacyRisk(value=0.0, ci=(0.0, 0.023886115062824436))
    ```
    This means:
    - `value=0.0`: The estimated privacy risk is 0 (no risk detected).
    - `ci=(0.0, 0.023886115062824436)`: The 95% confidence interval ranges from 0.0 to approximately 0.024,
    indicating the uncertainty in the risk estimate.

    Example of result interpretation:

    ```
    Success rate of main attack: SuccessRate(value=0.04152244529323714, error=0.017062327237406746)
    Success rate of baseline attack: SuccessRate(value=0.009766424188006807, error=0.00772382791604657)
    Success rate of control attack: SuccessRate(value=0.04303265336542551, error=0.017370801326913685)
    ```
    This means:
    - `Main Attack Success Rate`: 4.15% with an error margin of ±1.71%
    - `Baseline Attack Success Rate`: 0.98% with an error margin of ±0.77%
    - `Control Attack Success Rate`: 4.30% with an error margin of ±1.74%


    Attributes:
        real_data_path (Path): The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
        control_data_path (Path): The path to the control dataset.
        aux_cols (List[List[str]]): Auxiliary columns for evaluation.
                                    It specify what the attacker knows about its target, i.e. which columns are known
                                    to the attacker.
        n_attacks (int | None): Number of records to attack.
                                If None each record in the original dataset will be attacked.
        n_neighbors (int): The number of closest neighbors to include in the analysis.
        results (List[dict]): A list to store the computed metrics results.
        selected_metrics (list[str]): A list of metrics to evaluate. If None, all metrics are evaluated.
        want_parallel (bool): A boolean indicating whether to use parallel processing.
        display_result (bool): A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: LinkabilityMetric,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list[Path],
        aux_cols: tuple[list[str], list[str]],
        n_neighbors: int,
        n_attacks: int | None = None,
        control_data_path: Path | None = None,
        selected_metrics: list[str] | None = None,
        *,
        want_parallel: bool = False,
        display_result: bool = True,
    ) -> None:
        """Initializes the LinkabilityMetric class by setting paths, auxiliary columns, and other configurations.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
            aux_cols (List[List[str]]): Auxiliary columns for evaluation.
                                        It specify what the attacker knows about its target, i.e. which columns are
                                        known to the attacker.
            n_neighbors (int): The number of closest neighbors to include in the analysis.
            n_attacks (int | None): Number of records to attack.
                                    If None each record in the original dataset will be attacked.
                                    If control data is provided, sampling will also be done on the control dataset.
            control_data_path (Path | None): The path to the control dataset.
            selected_metrics (list[str] | None): Optional list of metrics to evaluate. If None,
                                    all metrics are evaluated.
            want_parallel (bool): Whether to use parallel processing. The default is False.
            display_result (bool): Whether to display the results. The default is True.
        """
        super().__init__()
        if isinstance(real_data_path, Path):
            self.real_data_path: Path = real_data_path
            self.real_data = load_data(real_data_path)
        elif isinstance(real_data_path, pd.DataFrame):
            self.real_data = real_data_path
        else:
            msg = "real_data_path must be either a pathlib.Path object pointing to a file or a pandas DataFrame."
            raise TypeError(
                msg,
            )

        self.synthetic_data_paths: list[Path] = synthetic_data_paths

        self.results: list[dict[str, Any]] = []

        self.real_data, self.fill_values = preprocess_data(self.real_data, need_clean_columns=True)

        self.control_data_path: Path | None = control_data_path
        if control_data_path:
            self.control_data = apply_preprocessing(control_data_path, self.fill_values, need_clean_columns=True)
            control_size = len(self.control_data) - 1
            self.n_attacks = min(n_attacks, control_size) if n_attacks is not None else control_size
        else:
            original_size = len(self.real_data) - 1
            self.n_attacks = min(n_attacks, original_size) if n_attacks is not None else original_size

        self.aux_cols = self.clean_list(aux_cols)
        self.n_neighbors = n_neighbors

        self.want_parallel = want_parallel
        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics: list[str] | None = selected_metrics

        LinkabilityMetric.__name__ = "Linkability"

        self.evaluate_all()

    @staticmethod
    def clean_list(aux_cols: tuple[list[str], list[str]]) -> tuple[list[str], ...]:
        """Cleans a list of auxiliary column lists by removing unwanted characters.

        Args:
            aux_cols (Tuple[List[str], List[str]]): A list of lists containing auxiliary column names.

        Returns:
            Tuple[List[str], List[str]]: Cleaned list of lists with auxiliary column names.
        """
        cleaned_cols = []
        for sublist in aux_cols:
            cleaned_sublist = []
            for item in sublist:
                cleaned_item = re.sub(r"[-./]", "", item)
                cleaned_sublist.append(cleaned_item)
            cleaned_cols.append(cleaned_sublist)
        return tuple(cleaned_cols)

    def evaluate(
        self: LinkabilityMetric,
        synthetic_data_path: Path,
    ) -> dict[str, Any]:
        """Evaluates a synthetic dataset against the real dataset using linkability metrics.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, Any]: A dictionary of computed metric scores or None if evaluation fails.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values, need_clean_columns=True).copy()
        model_name = synthetic_data_path.stem

        if self.control_data_path:
            evaluator = LinkabilityEvaluator(
                ori=self.real_data,
                syn=synthetic_data,
                control=self.control_data,
                aux_cols=self.aux_cols,
                n_attacks=self.n_attacks,
                n_neighbors=self.n_neighbors,
            )
        else:
            evaluator = LinkabilityEvaluator(
                ori=self.real_data,
                syn=synthetic_data,
                aux_cols=self.aux_cols,
                n_attacks=self.n_attacks,
                n_neighbors=self.n_neighbors,
            )

        evaluator.evaluate(n_jobs=-2)  # n_jobs follow joblib convention. -1 = all cores, -2 = all except one

        risk = evaluator.risk(confidence_level=0.95)
        res = evaluator.results()

        results = {
            "Model Name": model_name,
            "Privacy Risk": round(risk.value, 6),
            "CI(95%)": f"({round(risk.ci[0], 6)}, {round(risk.ci[1], 6)})",
            "Main Attack Success Rate": round(res.attack_rate[0], 6),
            "Main Attack Marginal Error ±": round(res.attack_rate[1], 6),
            "Baseline Attack Success Rate": round(res.baseline_rate[0], 6),
            "Baseline Attack Error ±": round(res.baseline_rate[1], 6),
        }

        if self.control_data_path:
            results.update(
                {
                    "Control Attack Success Rate": round(res.control_rate[0], 6),
                    "Control Attack Error ±": round(res.control_rate[1], 6),
                },
            )

        # Filter only explicitly selected metrics
        if self.selected_metrics:
            filtered_results = {
                "Model Name": model_name,
                **{metric: results[metric] for metric in self.selected_metrics if metric in results},
            }
        else:
            filtered_results = results

        self.results.append(filtered_results)
        return filtered_results
