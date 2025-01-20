from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd
from autogluon.tabular import TabularPredictor
from IPython.display import display
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from synthius.metric.utils import format_value

logger = getLogger()


class ModelFitter:
    """A class to fit and evaluate machine learning models using AutoGluon's TabularPredictor.

    This class is specifically designed for `binary` classification problems.

    Attributes:
        data_path (Path |pd.DataFrame): The path to the dataset or data as pd.DataFrame.
        label_column (str): The name of the target variable in the dataset.
        experiment_name (str): The name of the experiment, used for saving models.
        models_base_path (Path): The base path where models will be saved.
        test_data_path (Path | pd.DataFrame| optional): The path to the test dataset or test data as DataFrame,
                                                        if provided.
        pos_label (bool | str): The label of the positive class. Default is True.
        results_list (list): A list to store the results.
        pivoted_results (pd.DataFrame): A DataFrame to store pivoted results.

    Methods:
        fit(): Fits a model to the provided dataset and evaluates its performance.
        plot_metrics(): Plots the precision-recall curve for all experiments.
        display_metrics(): Displays the performance metrics for all experiments.

    Usage Example:
    ----------------------
    ```python
    for syn_path in synthetic_data_paths:
        ModelFitter(
            data_path=syn_path,
            label_column=TARGET,
            experiment_name=syn_path.stem,
            models_base_path=models_path,
            test_data_path=test_data,
            pos_label=POS_LABEL,
        )
    ModelFitter(
        data_path=train_data,
        label_column=TARGET,
        experiment_name="Original",
        models_base_path=models_path,
        test_data_path=test_data,
        pos_label=POS_LABEL,
    )
    # Plot the metrics
    plot = ModelFitter.plot_metrics(POS_LABEL)
    # Display the metrics
    ModelFitter.display_metrics()
    ````
    """

    results_list: ClassVar[list[dict[str, float]]] = []
    pivoted_results: pd.DataFrame

    def __init__(  # noqa: PLR0913
        self: ModelFitter,
        data_path: Path | pd.DataFrame,
        label_column: str,
        experiment_name: str,
        models_base_path: Path,
        test_data_path: Path | pd.DataFrame | None = None,
        *,
        pos_label: bool | str = True,
    ) -> None:
        """Initializes the ModelFitter with dataset information and model configuration.

        Parameters:
            data_path (Path | pd.DataFrame): The path to the dataset. It can be a file path or a pandas DataFrame.
            label_column (str): The name of the target variable in the dataset.
            experiment_name (str): The name of the experiment.
            models_base_path (Path): The base path for saving models.
            test_data_path (Path| pd.DataFrame | optional): The path to the test dataset or test data as DataFrame,
                                                            if provided.
            pos_label (bool | str): The label of the positive class. Default is True.
        """
        if isinstance(data_path, Path):
            self.data = pd.read_csv(data_path, low_memory=False)
        elif isinstance(data_path, pd.DataFrame):
            self.data = data_path
        else:
            msg = "real_data_path must be either a pathlib.Path object pointing to a file or a pandas DataFrame."
            raise TypeError(
                msg,
            )

        self.label_column = label_column
        self.experiment_name = experiment_name
        self.models_base_path = models_base_path
        self.test_data_path = test_data_path
        self.pos_label = pos_label
        self.fit()

    def fit(self: ModelFitter) -> None:
        """Fits a model to the provided dataset and evaluates its performance.

        This method uses the provided test dataset if available; otherwise, it splits the dataset into training and
        testing sets.
        It trains an AutoGluon TabularPredictor and stores evaluation metrics in a global dictionary.
        """
        if isinstance(self.test_data_path, Path):
            train_data = self.data
            test_data = pd.read_csv(self.test_data_path, low_memory=False)
        elif isinstance(self.test_data_path, pd.DataFrame):
            train_data = self.data
            test_data = self.test_data_path

        else:
            train_data, test_data = train_test_split(
                self.data,
                test_size=0.2,
                random_state=123,
                stratify=self.data[self.label_column],
            )

        predictor = TabularPredictor(
            label=self.label_column,
            problem_type="binary",
            eval_metric="f1_macro",
            path=self.models_base_path / self.experiment_name,
            verbosity=1,
        ).fit(train_data=train_data, fit_weighted_ensemble=False)

        predictions = predictor.predict(test_data)
        predictions_proba = predictor.predict_proba(test_data, as_multiclass=False)

        y_true = test_data[self.label_column]

        true_support = sum(y_true == self.pos_label)
        total_true_support = sum(self.data[self.label_column] == self.pos_label)

        results = {
            "Model Name": self.experiment_name,
            "f1": f1_score(y_true, predictions, pos_label=self.pos_label),
            "f1_weighted": f1_score(y_true, predictions, average="weighted"),
            "f1_macro": f1_score(y_true, predictions, average="macro"),
            "precision_macro": precision_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(y_true, predictions, average="macro"),
            "accuracy": accuracy_score(y_true, predictions),
            "true_support": true_support,
            "total_true_support": total_true_support,
            "predict_proba": predictions_proba,
            "y_true": y_true,
            "classification_report": classification_report(y_true, predictions, zero_division=0),
        }

        ModelFitter.results_list.append(results)
        ModelFitter.pivoted_results = self.pivot_results()

    @staticmethod
    def plot_metrics(*, pos_label: str | bool = True) -> None:
        """Plots the precision-recall curve for all experiments."""
        fig, ax = plt.subplots()

        for result in ModelFitter.results_list:
            predictions = result["predict_proba"]
            y_true = result["y_true"]
            PrecisionRecallDisplay.from_predictions(
                y_true=y_true,
                y_pred=predictions,
                pos_label=pos_label,
                name=result["Model Name"],
                ax=ax,
            )

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
        plt.close(fig)

    def pivot_results(self: ModelFitter) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(ModelFitter.results_list)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=[
                "f1",
                "f1_weighted",
                "f1_macro",
                "precision_macro",
                "recall_macro",
                "accuracy",
                "true_support",
                "total_true_support",
            ],
            var_name="Metric",
            value_name="Value",
        )

        pivot_table = df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

        return pivot_table.apply(lambda x: x.apply(format_value))

    @staticmethod
    def display_metrics() -> None:
        """Displays the performance metrics for all experiments."""
        if ModelFitter.pivoted_results is not None:
            display(ModelFitter.pivoted_results)
        else:
            logger.info("No results to display.")


class ModelLoader:
    """A class for loading and evaluating machine learning models on a given dataset.

    This class is specifically designed for `binary` classification problems.

    Attributes:
        data_path (Path): The path to the dataset file.
        label_column (str): The name of the column that contains the target variable.
        experiment_name (str): A unique identifier for the current experiment.
        models_base_path (Path): The path to the directory containing the trained models.
        pos_label (bool | str): The label of the positive class. Default is True.
        need_split (bool): A flag indicating whether the dataset should be split into training and testing sets.
        results_list (list): A list to store the results.
        pivoted_results (pd.DataFrame): A DataFrame to store pivoted results.

    Methods:
        load(): Loads the model, predicts and evaluates on test data, and stores results in results_list.
        plot_metrics(): Plots the precision-recall curve for all experiments.
        display_metrics(): Displays the performance metrics for all experiments.

    Usage Example:
    ----------------------
    ```python
    experiment_names = [path.stem for path in synthetic_data_paths] + ["Original"]
    for exp in experiment_names:
        ModelLoader(
            data_path=test_data,
            label_column=TARGET,
            experiment_name=exp,
            models_base_path=models_path / exp,
            pos_label=POS_LABEL,
            need_split=False,
        )
    # Plot the metrics
    ModelLoader.plot_metrics(POS_LABEL)
    # Display the metrics
    ModelLoader.display_metrics()
    ````
    """

    results_list: ClassVar[list[dict[str, float]]] = []
    pivoted_results: pd.DataFrame

    def __init__(  # noqa: PLR0913
        self: ModelLoader,
        data_path: Path,
        label_column: str,
        experiment_name: str,
        models_base_path: Path,
        *,
        pos_label: bool | str = True,
        need_split: bool = True,
    ) -> None:
        """Initializes ModelLoader with dataset, label column, experiment name, results dictionary, and model path.

        Parameters:
            data_path (Path): Path to CSV dataset.
            label_column (str): Target variable column name.
            experiment_name (str): Experiment name for tracking.
            models_base_path (Path): Path to trained model.
            pos_label (bool | str): The label of the positive class. Default is True.
            need_split (bool): A flag indicating whether the dataset should be split into training and testing sets.
                               If True, the dataset is split and a portion is used as test data. If False, the entire
                               dataset is used as test data.

        """
        self.data_path = data_path
        self.label_column = label_column
        self.experiment_name = experiment_name
        self.models_base_path = models_base_path
        self.pos_label = pos_label

        self.data = pd.read_csv(self.data_path, low_memory=False)

        if need_split:
            _, self.test_data = train_test_split(
                self.data,
                test_size=0.2,
                random_state=123,
                stratify=self.data[self.label_column],
            )
        else:
            self.test_data = self.data

        self.total_true_support = sum(self.data[self.label_column] == self.pos_label)

        self.load()

    def load(self: ModelLoader) -> None:
        """Loads the model, predicts and evaluates on test data, and stores results in results_dict."""
        predictor = TabularPredictor.load(self.models_base_path)
        predictions = predictor.predict(self.test_data)

        true_count = predictions.value_counts().get(self.pos_label, 0)

        predictions_proba = predictor.predict_proba(self.test_data, as_multiclass=False)

        y_true = self.test_data[self.label_column]

        results = {
            "Model Name": self.experiment_name,
            "f1": f1_score(y_true, predictions, pos_label=self.pos_label),
            "f1_weighted": f1_score(y_true, predictions, average="weighted"),
            "f1_macro": f1_score(y_true, predictions, average="macro"),
            "precision_macro": precision_score(y_true, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, predictions, average="macro"),
            "accuracy": accuracy_score(y_true, predictions),
            "predict_proba": predictions_proba,
            "y_true": y_true,
            "classification_report": classification_report(y_true, predictions, zero_division=0),
            "Predict_True": true_count,
        }

        ModelLoader.results_list.append(results)
        ModelLoader.pivoted_results = self.pivot_results()

    @staticmethod
    def plot_metrics(*, pos_label: str | bool = True) -> plt.Figure:
        """Plots the precision-recall curve for all experiments."""
        fig, ax = plt.subplots()

        for result in ModelLoader.results_list:
            predictions = result["predict_proba"]
            y_true = result["y_true"]
            PrecisionRecallDisplay.from_predictions(
                y_true=y_true,
                y_pred=predictions,
                pos_label=pos_label,
                name=result["Model Name"],
                ax=ax,
            )

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.show()
        return fig

    def pivot_results(self: ModelLoader) -> pd.DataFrame:
        """Transforms the accumulated results list into a pivoted DataFrame.

        Returns:
        pandas.DataFrame: A pivoted DataFrame where the columns are the model names and the rows are the different
                          metrics calculated for each model. Each cell in the DataFrame represents the metric value
                          for a specific model.
        """
        df_results = pd.DataFrame(ModelLoader.results_list)

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=["f1", "f1_weighted", "f1_macro", "precision_macro", "recall_macro", "accuracy", "Predict_True"],
            var_name="Metric",
            value_name="Value",
        )

        pivot_table = df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

        return pivot_table.apply(lambda x: x.apply(format_value))

    @staticmethod
    def display_metrics() -> None:
        """Displays the performance metrics for all experiments."""
        if ModelLoader.pivoted_results is not None:
            display(ModelLoader.pivoted_results)
        else:
            logger.info("No results to display.")
