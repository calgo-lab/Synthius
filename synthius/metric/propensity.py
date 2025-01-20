from __future__ import annotations

import shutil
from logging import getLogger
from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.tabular import TabularPredictor
from IPython.display import display
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from synthius.metric.utils import BaseMetric, load_data

logger = getLogger()

# flake8: noqa:N806


class PropensityScore(BaseMetric):
    """A class to evaluate and compare the propensity scores of synthetic data against real data.

    This method combines synthetic data with real data, assigns a new label, and uses three ML methods
    (`AutoGluon`, `XGBClassifier`, and `HistGradientBoostingClassifier`) to predict the target variable.
    The real and synthetic datasets are labeled accordingly and combined into a single dataset. The combined
    dataset is randomly shuffled to ensure a mix of real and synthetic data. The models are evaluated based
    on their accuracy in distinguishing between real and synthetic data points. The accuracy scores serve as
    the Propensity Scores, indicating the degree to which the synthetic data can be differentiated from the
    real data.

    Attributes:
        real_data_path (Path): The path to the real dataset Or real data as pd.DataFrame.
        synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
        id_column (str | None): The name of the ID column to be dropped from the datasets.
        results (list[dict[str, Any]]): A list to store the evaluation results.
        real_data (pd.DataFrame): The loaded real dataset.
        model_dir (Path): Directory for storing temporary model files.
        selected_metrics (list[str]): A list of metrics to evaluate. If None, all metrics are evaluated.
        display_result (bool): A boolean indicating whether to display the results.
    """

    def __init__(  # noqa: PLR0913
        self: PropensityScore,
        real_data_path: Path | pd.DataFrame,
        synthetic_data_paths: list[Path],
        id_column: str | None = None,
        selected_metrics: list[str] | None = None,
        *,
        display_result: bool = True,
    ) -> None:
        """Initializes the PropensityScore object with real and synthetic dataset paths, and the ID column name.

        Args:
            real_data_path (Path | pd.DataFrame): The file path to the real dataset or real data as pd.DataFrame.
            synthetic_data_paths (list[Path]): A list of file paths to the synthetic datasets.
            id_column (str | None): The name of the ID column to be dropped from the datasets.
            selected_metrics (list[str] | None): Optional list of metrics to evaluate. If None,
                                                 all metrics are evaluated.
            display_result (bool): Whether to display the results after evaluation.
        """
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

        self.id_column = id_column
        if self.id_column is None:
            logger.warning("No ID column selected; all columns will be used for analysis.")

        self.model_dir = Path("./TempAGModel")

        self.display_result = display_result
        self.pivoted_results = None

        self.selected_metrics = selected_metrics

        PropensityScore.__name__ = "Propensity Score"

        self.evaluate_all()
        self.cleanup()

    def load_data(self: PropensityScore, data_path: Path) -> pd.DataFrame:
        """Loads the dataset from the specified path, checking for the ID column.

        Args:
            data_path (Path): The path to the dataset file.

        Returns:
            pd.DataFrame: The loaded dataset, with the ID column dropped if it exists.
        """
        data = pd.read_csv(data_path, low_memory=False)

        if self.id_column:
            if self.id_column in data.columns:
                data = data.drop(columns=[self.id_column])
            else:
                logger.warning("The ID column %s does not exist in the dataset.", self.id_column)
        return data

    def clean_columns(self: PropensityScore, data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the column names in the DataFrame by removing unwanted characters.

        Args:
            data (pd.DataFrame): DataFrame whose columns need cleaning.

        Returns:
            pd.DataFrame: DataFrame with cleaned column names.
        """
        data.columns = data.columns.str.replace("[-./]", "", regex=True)
        return data

    def preprocess_data(
        self: PropensityScore,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepares the training and testing datasets by combining, shuffling, and splitting real and synthetic data.

        This function also cleans the column names by removing unwanted characters before the combination.

        Args:
            original_data (pd.DataFrame): The original real dataset.
            synthetic_data (pd.DataFrame): The synthetic dataset.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The training and testing datasets.
        """
        orginal = original_data.copy()
        synthetic = synthetic_data.copy()

        orginal = self.clean_columns(orginal)
        synthetic = self.clean_columns(synthetic)

        self.lable = "combined_label"

        orginal[self.lable] = 0
        synthetic[self.lable] = 1

        combined_data = pd.concat([orginal, synthetic], ignore_index=True).sample(frac=1).reset_index(drop=True)

        train_data, test_data = train_test_split(
            combined_data,
            test_size=0.2,
            random_state=42,
            stratify=combined_data[self.lable],
        )
        return train_data, test_data

    def compute_autogluon_score(
        self: PropensityScore,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_subdir: str,
    ) -> float:
        """Trains and evaluates an AutoGluon model on the provided data.

        Args:
            train_data (pd.DataFrame): The training dataset.
            test_data (pd.DataFrame): The testing dataset.
            model_subdir (str): Subdirectory for storing AutoGluon model files.

        Returns:
            float: The accuracy score of the AutoGluon model.
        """
        predictor = TabularPredictor(
            label=self.lable,
            problem_type="binary",
            verbosity=1,
            path=self.model_dir / model_subdir,
        ).fit(train_data=train_data)
        performance = predictor.evaluate(test_data)

        return performance["accuracy"]

    def compute_xgboost_score(
        self: PropensityScore,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> float:
        """Trains and evaluates an XGBoost model on the provided data.

        Args:
            train_data (pd.DataFrame): The training dataset.
            test_data (pd.DataFrame): The testing dataset.

        Returns:
            float: The accuracy score of the XGBoost model.
        """
        # Convert object types to category
        for col in train_data.select_dtypes(include=["object"]).columns:
            train_data[col] = train_data[col].astype("category")
            test_data[col] = test_data[col].astype("category")

        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", enable_categorical=True)
        X_train = train_data.drop(columns=[self.lable])
        y_train = train_data[self.lable]
        X_test = test_data.drop(columns=[self.lable])
        y_true = test_data[self.lable]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_true, predictions)

    def compute_hist_gradient_boosting_score(
        self: PropensityScore,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> float:
        """Trains and evaluates a Histogram-based Gradient Boosting model on the provided data.

        Args:
            train_data (pd.DataFrame): The training dataset.
            test_data (pd.DataFrame): The testing dataset.

        Returns:
            float: The accuracy score of the Histogram-based Gradient Boosting model.
        """
        model = HistGradientBoostingClassifier(random_state=42)
        X_train = train_data.drop(columns=[self.lable])
        y_train = train_data[self.lable]
        X_test = test_data.drop(columns=[self.lable])
        y_true = test_data[self.lable]

        # Convert categorical columns to numerical
        for col in X_train.select_dtypes(include=["object", "category"]).columns:
            X_train[col] = X_train[col].astype("category").cat.codes
            X_test[col] = X_test[col].astype("category").cat.codes

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        return accuracy_score(y_true, predictions)

    def evaluate(self: PropensityScore, synthetic_data_path: Path) -> pd.DataFrame:
        """Evaluates a synthetic dataset against the real dataset using various models.

        Args:
            synthetic_data_path (Path): The path to the synthetic dataset.

        Returns:
            pd.DataFrame: A pd.DataFrame containing the evaluation scores of the models.
        """
        synthetic_data = self.load_data(synthetic_data_path).copy()
        train_data, test_data = self.preprocess_data(self.real_data, synthetic_data)

        model_name = synthetic_data_path.stem

        metric_dispatch = {
            "Autogluon": lambda: self.compute_autogluon_score(train_data, test_data, model_subdir=f"{model_name}"),
            "XGBoost": lambda: self.compute_xgboost_score(train_data, test_data),
            "HistGradientBoosting": lambda: self.compute_hist_gradient_boosting_score(train_data, test_data),
        }

        selected_metrics = self.selected_metrics if self.selected_metrics is not None else list(metric_dispatch.keys())

        results: dict[str, str | float] = {"Model Name": model_name}

        for metric in selected_metrics:
            if metric in metric_dispatch:
                try:
                    results[metric] = metric_dispatch[metric]()
                    logger.info("%s for %s Done.", metric, model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error computing %s for %s: %s", metric, model_name, exc)  # noqa: TRY400
                    results[metric] = float("nan")
            else:
                logger.warning("Metric %s is not supported and will be skipped.", metric)

        logger.info("Propensity Score evaluation for %s completed.", model_name)
        self.results.append(results)
        return results

    def pivot_results(self: PropensityScore) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(self.results)

        available_metrics = [
            "Autogluon",
            "XGBoost",
            "HistGradientBoosting",
        ]
        present_metrics = [metric for metric in available_metrics if metric in df_results.columns]

        if not present_metrics:
            msg = "No valid metrics found in the results. Check the selected metrics."
            raise ValueError(msg)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=present_metrics,
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def evaluate_all(self: PropensityScore) -> None:
        """Evaluates all synthetic datasets against the real dataset and stores the results."""
        for synthetic_data_path in self.synthetic_data_paths:
            self.evaluate(synthetic_data_path)

        self.pivoted_results = self.pivot_results()
        if self.display_result:
            self.display_results()

    def display_results(self: PropensityScore) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")

    def cleanup(self: PropensityScore) -> None:
        """Cleans up by removing the temporary model directory."""
        if self.model_dir.exists():
            shutil.rmtree(self.model_dir)
