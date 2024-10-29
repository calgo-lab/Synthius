from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator
from IPython.display import display

if TYPE_CHECKING:
    from pathlib import Path
from synthetic_data.metric.utils import BaseMetric, apply_preprocessing, load_data, preprocess_data

logger = logging.getLogger("anonymeter")
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003


class SinglingOutMetric(BaseMetric):
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
        real_data_path (Path): The path to the real dataset.
        synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
        control_data_path (Path): The path to the control dataset.
        mode (str): The evaluation mode ('univariate' or 'multivariate').
        n_attacks (int): The number of attacks to simulate during evaluation.
        n_cols (Optional[int]): The number of columns to consider for multivariate mode.
        results (List[dict]): A list to store the computed metrics results.
    """

    def __init__(  # noqa: PLR0913
        self: SinglingOutMetric,
        real_data_path: Path,
        synthetic_data_paths: list[Path],
        control_data_path: Path | None,
        mode: str,
        n_attacks: int,
        n_cols: int | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the SinglingOutMetric class by setting paths, mode, and other configurations.

        Args:
            real_data_path (Path): The path to the real dataset.
            synthetic_data_paths (List[Path]): A list of paths to the synthetic datasets.
            control_data_path (Path | None): The path to the control dataset.
            mode (str): The evaluation mode ('univariate' or 'multivariate').
            n_attacks (int): The number of attacks to simulate.
            n_cols (int | None ): The number of columns to consider for multivariate mode.
            display_result (bool): Whether to display the results. The default is True.
        """
        self.real_data_path: Path = real_data_path
        self.synthetic_data_paths: list[Path] = synthetic_data_paths
        self.control_data_path: Path | None = control_data_path

        self.mode = mode
        self.n_attacks = n_attacks
        self.n_cols = n_cols

        self.results: list[dict[str, str | float]] = []
        self.real_data: pd.DataFrame = load_data(real_data_path)

        self.real_data, self.fill_values = preprocess_data(self.real_data, need_clean_columns=True)

        if control_data_path:
            self.control_data = apply_preprocessing(control_data_path, self.fill_values, need_clean_columns=True)

        self.display_result = display_result
        self.pivoted_results = None

        SinglingOutMetric.__name__ = "Singling Out"

        self.evaluate_all()

    def evaluate(
        self: SinglingOutMetric,
        synthetic_data_path: Path,
    ) -> dict[str, str | float] | None:
        """Evaluates a synthetic dataset against the real dataset using singling out metrics.

        Args:
            synthetic_data_path: The path to the synthetic dataset to evaluate.

        Returns:
            dict[str, str | float] | None: A dictionary of computed metric scores or None if evaluation fails.
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

            return {
                "Model Name": model_name,
                "Privacy Risk": round(risk.value, 6),
                "CI(95%)": f"({round(risk.ci[0], 6)}, {round(risk.ci[1], 6)})",
                "Main Attack Success Rate": round(res.attack_rate[0], 6),
                "Main Attack Marginal Error ±": round(res.attack_rate[1], 6),
                "Baseline Attack Success Rate": round(res.baseline_rate[0], 6),
                "Baseline Attack Error ±": round(res.baseline_rate[1], 6),
                "Control Attack Success Rate": round(res.control_rate[0], 6),
                "Control Attack Error ±": round(res.control_rate[1], 6),
            }

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

    def pivot_results(self: SinglingOutMetric) -> pd.DataFrame:
        """Pivots the accumulated results to organize models as columns and metrics as rows.

        Returns:
            pd.DataFrame: A pivoted DataFrame of the evaluation results.
        """
        try:
            df_results = pd.DataFrame(self.results)

            numeric_metrics = [
                "Privacy Risk",
                "Main Attack Success Rate",
                "Main Attack Marginal Error ±",
                "Baseline Attack Success Rate",
                "Baseline Attack Error ±",
                "Control Attack Success Rate",
                "Control Attack Error ±",
            ]
            non_numeric_metrics = ["CI(95%)"]

            # Replace 'Failed' with NaN in numeric columns
            df_results[numeric_metrics] = df_results[numeric_metrics].apply(pd.to_numeric, errors="coerce")

            df_melted_numeric = df_results.melt(
                id_vars=["Model Name"],
                value_vars=numeric_metrics,
                var_name="Metric",
                value_name="Value",
            )

            df_melted_non_numeric = df_results.melt(
                id_vars=["Model Name"],
                value_vars=non_numeric_metrics,
                var_name="Metric",
                value_name="Value",
            )

            pivoted_df_numeric = df_melted_numeric.pivot_table(
                index="Metric",
                columns="Model Name",
                values="Value",
                aggfunc="mean",  # Use mean to handle NaN
            )

            pivoted_df_non_numeric = df_melted_non_numeric.pivot_table(
                index="Metric",
                columns="Model Name",
                values="Value",
                aggfunc="first",  # First is okay for non-numeric
            )

            pivoted_df = pd.concat([pivoted_df_numeric, pivoted_df_non_numeric])

            desired_order = [
                "Privacy Risk",
                "CI(95%)",
                "Main Attack Success Rate",
                "Main Attack Marginal Error ±",
                "Baseline Attack Success Rate",
                "Baseline Attack Error ±",
                "Control Attack Success Rate",
                "Control Attack Error ±",
            ]
            return pivoted_df.reindex(desired_order)

        except Exception as e:
            logger.exception("Error while pivoting the DataFrame: %s", e)  # noqa: TRY401
            return pd.DataFrame()

    def evaluate_all(self: SinglingOutMetric) -> None:
        """Evaluates all synthetic datasets in parallel and stores the results."""
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self.evaluate, path): path for path in self.synthetic_data_paths}
            for future in as_completed(futures):
                path = futures[future]
                model_name = path.stem

                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                        logger.info("Singling Out for %s Done.", model_name)

                except RuntimeError as ex:
                    logger.exception("Evaluation failed for %s: %s", path, ex)  # noqa: TRY401
                except Exception as ex:
                    logger.exception("An unexpected error occurred for %s: %s", path, ex)  # noqa: TRY401

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: SinglingOutMetric) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")
