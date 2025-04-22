from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator
from IPython.display import display

from synthius.metric.utils import BaseMetric, apply_preprocessing, load_data, preprocess_data
from synthius.metric.utils.anonymeter_metric import AnonymeterMetric

logger = logging.getLogger("anonymeter")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


class SinglingOutMetric(AnonymeterMetric):
    """A class to compute `Singling Out Risk` for synthetic data compared to real data.

    Adapted from Anonymeter:
    https://github.com/statice/anonymeter

    This involves assessing privacy risks by comparing synthetic datasets against a real dataset and a control dataset.

    This evaluator checks if it's possible to identify individual records from the original dataset using the
    synthetic dataset. It tries to find unique combinations of attributes that point to a single individual
    in the original dataset.

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
        mode (str): The evaluation mode ('univariate' or 'multivariate').
        n_attacks (int): The number of attacks to simulate during evaluation.
        n_cols (Optional[int]): The number of columns to consider for multivariate mode.
        control_data_path (Path): The path to the control dataset.
        results (List[dict]): A list to store the computed metrics results.
        selected_metrics (list[str]): A list of metrics to evaluate. If None, all metrics are evaluated.
        want_parallel (bool): A boolean indicating whether to use parallel processing.
        display_result (bool): A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: SinglingOutMetric,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list[Path],
        mode: str,
        n_attacks: int,
        n_cols: int | None = None,
        control_data_path: Path | None = None,
        selected_metrics: list[str] | None = None,
        *,
        want_parallel: bool = False,
        display_result: bool = True,
    ) -> None:
        """Initializes the SinglingOutMetric class by setting paths, mode, and other configurations.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
            mode (str): The evaluation mode ('univariate' or 'multivariate').
            n_attacks (int): The number of attacks to simulate.
            n_cols (int | None ): The number of columns to consider for multivariate mode.
            control_data_path (Path | None): The path to the control dataset. The default is None.
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
        self.control_data_path: Path | None = control_data_path

        self.mode = mode
        self.n_attacks = n_attacks
        self.n_cols = n_cols

        self.results: list[dict[str, str | float]] = []

        self.real_data, self.fill_values = preprocess_data(self.real_data, need_clean_columns=True)

        if control_data_path:
            self.control_data = apply_preprocessing(control_data_path, self.fill_values, need_clean_columns=True)

        self.want_parallel = want_parallel
        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

        SinglingOutMetric.__name__ = "Singling Out"

        self.evaluate_all()

    def evaluate(
        self: SinglingOutMetric,
        synthetic_data_path: Path,
    ) -> dict[str, str | float]:
        """Evaluates a synthetic dataset against the real dataset using singling out metrics.

        Args:
            synthetic_data_path: The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, str | float]: A dictionary of computed metric scores or None if evaluation fails.
        """
        synthetic_data = apply_preprocessing(synthetic_data_path, self.fill_values, need_clean_columns=True).copy()
        model_name = synthetic_data_path.stem

        if self.control_data_path:
            evaluator = SinglingOutEvaluator(
                ori=self.real_data,
                syn=synthetic_data,
                control=self.control_data,
                n_attacks=self.n_attacks,
                n_cols=self.n_cols if self.mode == "multivariate" else None,
            )
        else:
            evaluator = SinglingOutEvaluator(
                ori=self.real_data,
                syn=synthetic_data,
                n_attacks=self.n_attacks,
                n_cols=self.n_cols if self.mode == "multivariate" else None,
            )

        try:
            evaluator.evaluate(mode=self.mode)
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
            return filtered_results  # noqa: TRY300

        except RuntimeError as ex:
            logger.error(  # noqa: TRY400
                "Singling out evaluation failed for %s with %s. Please re-run this evaluation. "
                "For more stable results, increase `n_attacks`. Note that this will make the evaluation slower.",
                model_name,
                ex,
            )
            return {
                "Model Name": synthetic_data_path.stem,
                "Privacy Risk": "Failed",
                "CI(95%)": "Failed",
                "Main Attack Success Rate": "Failed",
                "Main Attack Marginal Error ±": "Failed",
                "Baseline Attack Success Rate": "Failed",
                "Baseline Attack Error ±": "Failed",
                "Control Attack Success Rate": "Failed",
                "Control Attack Error ±": "Failed",
            }
