from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from anonymeter.evaluators import InferenceEvaluator

from synthius.metric.utils import apply_preprocessing, load_data, preprocess_data
from synthius.metric.utils.anonymeter_metric import AnonymeterMetric

logger = logging.getLogger("anonymeter")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


class InferenceMetric(AnonymeterMetric):
    """A class to compute `InferenceMetric Risk` for synthetic data compared to real data.

    Adapted from Anonymeter:
    https://github.com/statice/anonymeter

    This involves assessing privacy risks by comparing synthetic datasets against a real dataset and a control dataset.

    This evaluator measures the risk that the attacker can learn the attributes of a target record in the original data
    based on the knowledge from the synthetic data and the partial knowledge of some attributes of the
    target record (the auxiliary information).


    Attributes:
        real_data_path (Path): The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
        control_data_path (Path): The path to the control dataset.
        aux_cols (List[str]): Auxiliary columns for evaluation.
                                    It specifies what the attacker knows about its target, i.e. which columns are known
                                    to the attacker.
        secret: The "secret" column that the attacker should guess.
        regression: Whether the `secret` is continuous (True) or categorical (False).
        n_attacks int: Number of attack attempts.
        results (List[dict]): A list to store the computed metrics results.
        selected_metrics (list[str]): A list of metrics to evaluate. If None, all metrics are evaluated.
        want_parallel (bool): A boolean indicating whether to use parallel processing.
        display_result (bool): A boolean indicating whether to display the results.
    """

    def __init__(self: InferenceMetric,
                 real_data_path: Path | pd.DataFrame,
                 synthetic_data_paths: list[Path],
                 aux_cols: list[str],
                 secret: str,
                 n_attacks: int,
                 regression: Optional[bool] = None,
                 control_data_path: Path | None = None,
                 selected_metrics: list[str] | None = None,
                 *,
                 want_parallel: bool = False,
                 display_result: bool = True) -> None:
        """Initializes the InferenceMetric class by setting paths, auxiliary columns, and other configurations.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
            aux_cols (List[str]): Auxiliary columns for evaluation.
                                        It specifies what the attacker knows about its target, i.e. which columns are
                                        known to the attacker.
            secret: The "secret" column that the attacker should guess.
            regression: Whether the `secret` is continuous (True) or categorical (False).
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

        self.results: list[dict[str, str | float]] = []

        self.real_data, self.fill_values = preprocess_data(self.real_data, need_clean_columns=True)
        self.real_data.columns = self.clean_list(self.real_data.columns)

        self.control_data = None
        if control_data_path:
            self.control_data = apply_preprocessing(control_data_path, self.fill_values, need_clean_columns=True)
            self.control_data.columns = self.clean_list(self.control_data.columns)
            control_size = len(self.control_data) - 1
            self.n_attacks = min(n_attacks, control_size) if n_attacks is not None else control_size
        else:
            original_size = len(self.real_data) - 1
            self.n_attacks = min(n_attacks, original_size) if n_attacks is not None else original_size

        self.aux_cols = self.clean_list(aux_cols)
        self.secret = secret
        self.regression = regression

        self.want_parallel = want_parallel
        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

        InferenceMetric.__name__ = "Inference"

        self.evaluate_all()

    @staticmethod
    def clean_list(aux_cols: list[str]) -> list[str]:
        """Cleans a list of auxiliary column lists by removing unwanted characters.

        Args:
            aux_cols (List[str]): A list of auxiliary column names.

        Returns:
            List[str]: Cleaned list with the auxiliary column names.
        """
        clean_cols = []
        for item in aux_cols:
            cleaned_item = re.sub(r"[-./]", "", item)
            clean_cols.append(cleaned_item)
        return clean_cols

    def evaluate(
            self: InferenceMetric,
            synthetic_data_path: Path,
    ) -> dict[str, str | float]:
        """Evaluates a synthetic dataset against the real dataset using Inference metrics.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, str | float]: A dictionary of computed metric scores or None if evaluation fails.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values, need_clean_columns=True).copy()
        model_name = synthetic_data_path.stem

        evaluator = InferenceEvaluator(
            ori=self.real_data,
            syn=synthetic_data,
            control=self.control_data,
            aux_cols=self.aux_cols,
            n_attacks=self.n_attacks,
            secret=self.secret,
            regression=self.regression
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

        if self.control_data is not None:
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
