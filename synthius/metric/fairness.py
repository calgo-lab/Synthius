from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from IPython.display import display
from plotly.subplots import make_subplots
from scipy.linalg import LinAlgError
from sdmetrics.visualization import get_column_plot

if TYPE_CHECKING:
    from pathlib import Path

    from plotly.graph_objs import Figure
from synthius.metric.utils import BaseMetric, load_data

logger = getLogger()


class LogDisparityMetrics(BaseMetric):
    """Evaluates fairness of synthetic data via log disparity across selected features.

    This class is designed to help assess the fairness of synthetic datasets by comparing them to a real dataset
    using log disparity metrics on selected features. The goal is to ensure that synthetic data generation techniques
    do not introduce or exacerbate biases that could undermine fairness goals.

    Attributes:
        real_data (pd.DataFrame): The real dataset loaded from a CSV file.
        synthetic_data_paths (list[Path]): Paths to the synthetic dataset CSV files.
        features (list[str]): List of features to evaluate for log disparity.
        results (dict): Stores the log disparity results for each feature across all synthetic datasets.

    Methods:
        calculate_log_disparity: Calculates the log disparity between real and synthetic data for a given feature.
        evaluate_all: Evaluates all synthetic datasets against the real dataset for all features.
        highlight_zeros_background: Applies conditional formatting based on the value being close to zero.
        display_results: Displays the evaluation results with conditional formatting.
    """

    def __init__(
        self: LogDisparityMetrics,
        real_data_path: Path,
        synthetic_data_paths: list[Path],
        features: list[str],
    ) -> None:
        """Initializes the LogDisparityMetrics class with real and synthetic data paths and features.

        Args:
            real_data_path (Path): The path to the real dataset CSV file.
            synthetic_data_paths (list[Path]): List of paths to synthetic dataset CSV files.
            features (list[str]): List of features to be evaluated for log disparity.
        """
        self.real_data = pd.read_csv(real_data_path, low_memory=False)
        self.synthetic_data_paths = synthetic_data_paths
        self.features = features
        self.results = {feature: pd.DataFrame() for feature in features}

        self.evaluate_all()
        self.display_results()

    def calculate_log_disparity(
        self: LogDisparityMetrics,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        feature: str,
    ) -> pd.Series:
        """Calculates log disparity between real and synthetic data for a specific feature.

        The log disparity is computed by first calculating the odds ratios of feature values in both real and synthetic
        datasets, and then taking the logarithm of the ratio of these odds.

        Args:
            real_data (pd.DataFrame): The real dataset.
            synthetic_data (pd.DataFrame): A synthetic dataset.
            feature (str): The feature for which to calculate the log disparity.

        Returns:
            pd.Series: The log disparity values for the feature.
        """
        real_counts = real_data[feature].value_counts()
        synthetic_counts = synthetic_data[feature].value_counts()

        epsilon = 1e-8
        real_odds = (real_counts / (real_counts.sum() - real_counts + epsilon)).replace(np.inf, 0)
        synthetic_odds = (synthetic_counts / (synthetic_counts.sum() - synthetic_counts + epsilon)).replace(np.inf, 0)

        real_odds, synthetic_odds = real_odds.align(synthetic_odds, fill_value=epsilon)

        return np.log(synthetic_odds) - np.log(real_odds)

    def evaluate_all(self: LogDisparityMetrics) -> None:
        """Evaluates all synthetic datasets against the real dataset for all features and stores the results."""
        for synthetic_data_path in self.synthetic_data_paths:
            synthetic_data = pd.read_csv(synthetic_data_path, low_memory=False)
            model_name = synthetic_data_path.stem

            for feature in self.features:
                log_disparity = self.calculate_log_disparity(self.real_data, synthetic_data, feature)
                self.results[feature][model_name] = log_disparity

    @staticmethod
    def highlight_zeros_background(val: float) -> str:
        """Applies conditional formatting to values close to zero.

        Args:
            val (float): The value to be formatted.

        Returns:
            str: The CSS style to be applied based on the value.
        """
        epsilon = 0.05

        if val == 0:
            color = "blue"
        elif -epsilon <= val <= epsilon:
            color = "green"
        else:
            color = None
        return f"background-color: {color}" if color else ""

    def display_results(self: LogDisparityMetrics) -> None:
        """Displays the evaluation results with conditional formatting using Styler.map."""
        for feature, df in self.results.items():
            styled_df = df.style.map(self.highlight_zeros_background)
            logger.info("Results for %s:", feature)
            display(styled_df)


class DistributionVisualizer:
    """Creates and visualizes distribution plots to compare data fairness between real and synthetic datasets.

    This class facilitates the visual comparison between the distributions of features in real and synthetic datasets
    to assess data fairness. It supports visualization for numeric, categorical, and other specific feature types.
    Features can be automatically selected or specified manually. The class includes methods to prepare data by
    selecting features and excluding identifiers, and to create distribution plots for various feature categories.

    Attributes:
        real_data (pd.DataFrame): Dataframe containing the real dataset loaded from a specified path.
        synthetic_data (pd.DataFrame): Dataframe containing the synthetic dataset loaded from a specified path.
        features (list[str]): List of feature names to be visualized. This can be specified by the user or determined
                              from the dataset.
        categorical_features (list[str]): List of categorical feature names identified from `features`.
        numeric_features (list[str]): List of numeric feature names identified from `features`.
        other_features (list[str]): List of other feature names, not strictly numeric or categorical, identified
                                    from `features`.

    Methods:
        prepare_data: Prepares the data for visualization by optionally dropping an ID feature and selecting specific
                      features for comparison.
        create_category_plots: Creates and returns a figure containing distribution plots for the specified features,
                                organized in a grid layout with a given number of columns per row.
        visualize_features: A method to visualize distribution plots for all feature categories (categorical, numeric,
                            and others) after preparing the data accordingly.
    """

    def __init__(self: DistributionVisualizer, real_data_path: Path, synthetic_data_path: Path) -> None:
        """Initializes the DistributionVisualizer object with real and synthetic datasets.

        Parameters:
            real_data_path (Path): Path to the CSV file containing the real dataset.
            synthetic_data_path (Path): Path to the CSV file containing the synthetic dataset.
        """
        self.real_data: pd.DataFrame = load_data(real_data_path)
        self.synthetic_data: pd.DataFrame = load_data(synthetic_data_path)
        self.features: list[str] = []
        self.categorical_features: list[str] = []
        self.numeric_features: list[str] = []
        self.other_features: list[str] = []

    def prepare_data(
        self: DistributionVisualizer,
        id_feature: str | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        """Prepares the datasets by optionally removing an ID feature and selecting specific features for visualization.

        Parameters:
            id_feature (str | None): The name of the ID feature to be removed from the datasets, if any.
            selected_features (list[str] | None): A list of feature names to be visualized. If None, all features
                                                  are considered.
        """
        if id_feature:
            self.real_data = self.real_data.drop(id_feature, axis=1)
            self.synthetic_data = self.synthetic_data.drop(id_feature, axis=1)

        if selected_features is not None:
            self.features = selected_features
        else:
            self.features = list(self.real_data.columns)

        self.categorical_features = [f for f in self.features if self.real_data[f].nunique() == 2 and self.real_data[f].dtype not in ["int64", "float64"]]
        self.numeric_features = [f for f in self.features if self.real_data[f].dtype in ["int64", "float64"]]
        self.other_features = [f for f in self.features if f not in self.categorical_features + self.numeric_features and self.real_data[f].nunique() <= 45]

    def create_category_plots(
        self: DistributionVisualizer,
        features: list[str],
        title: str,
        cols_per_row: int,
    ) -> Figure:
        """Creates a figure with distribution plots for specified features.

        This method organizes the plots in a grid layout, allowing for a visual comparison of real and synthetic data
        distributions.

        Parameters:
            features (list[str]): List of feature names for which distribution plots will be created.
            title (str): The title of the figure.
            cols_per_row (int): Number of plots per row in the figure layout.

        Returns:
            Figure: A plotly.graph_objs._figure.Figure object containing the distribution plots.
        """
        if not features:
            return None

        fig, plot_width, plot_height = self._initialize_figure(features, cols_per_row)

        for i, feature in enumerate(features, start=1):
            if i > len(features):
                break
            row, col = self._get_plot_position(i, cols_per_row)

            try:
                temp_fig = self._get_temp_figure(feature)
                if not temp_fig.data:
                    logger.info("Skipping feature %s due to no data.", feature)
                    continue

                self._add_traces_to_figure(fig, temp_fig, i, row, col)

            except LinAlgError as e:
                logger.error(  # noqa: TRY400
                    "LinAlgError encountered when plotting feature %s: %s. Skipping this feature.",
                    feature,
                    e,
                )
            except IndexError as e:
                logger.error(  # noqa: TRY400
                    "Error plotting feature %s because synthetic data has no values for this feature: %s",
                    feature,
                    e,
                )
            if feature not in self.numeric_features:
                # Hides x-axis labels for categorical feature subplots
                fig.update_xaxes(title_text="", visible=False, row=row, col=col)

        self._finalize_figure(fig, plot_width, plot_height, title)
        return fig

    def _initialize_figure(
        self: DistributionVisualizer,
        features: list[str],
        cols_per_row: int,
    ) -> tuple[Figure, int, int]:
        """Initialize the figure with proper layout and dimensions.

        Parameters:
            features (list[str]): List of feature names for which distribution plots will be created.
            cols_per_row (int): Number of plots per row in the figure layout.

        Returns:
            Tuple[Figure, int, int]: A tuple containing the initialized figure, plot width, and plot height.
        """
        dpi = 96  # DPI setting
        mm_to_inches = 1 / 25.4  # mm to inches conversion
        a4_width_mm = 210  # A4 width in mm
        a4_height_mm = 297  # A4 height in mm
        a4_width_inches = a4_width_mm * mm_to_inches  # Convert A4 width to inches
        a4_width_pixels = a4_width_inches * dpi  # Convert A4 width to pixels
        a4_height_inches = a4_height_mm * mm_to_inches  # Convert A4 height to inches
        a4_height_pixels = a4_height_inches * dpi  # Convert A4 height to pixels

        # Calculate the total number of rows and the actual number of columns needed
        rows = max((len(features) + cols_per_row - 1) // cols_per_row, 1)
        actual_cols = min(len(features), cols_per_row)

        # Adjust the figure's width based on the actual number of columns to reduce whitespace
        margin_per_subplot = 20  # Margin between subplots
        total_usable_width = a4_width_pixels - (actual_cols)
        plot_width = int(total_usable_width)  # Adjust plot width based on actual columns and convert to int

        # Set fixed subplot height if rows <= 6, otherwise calculate dynamically
        max_rows_for_fixed_size = 6
        fixed_subplot_height = 200  # Fixed height for each subplot

        if rows <= max_rows_for_fixed_size:
            plot_height = max(rows * fixed_subplot_height, 400)  # Ensure a minimum height
        else:
            # Use dynamic calculation for more rows
            available_height_per_row = (a4_height_pixels * 0.8) / rows
            extra_space = a4_height_pixels * 0.2  # Reserve for margins and legends
            plot_height = int(min(rows * available_height_per_row + extra_space, a4_height_pixels))

        fig = make_subplots(
            rows=rows,
            cols=actual_cols,
            subplot_titles=features[: rows * actual_cols],
            horizontal_spacing=margin_per_subplot / plot_width,
            vertical_spacing=50 / plot_height,
        )
        return fig, plot_width, plot_height

    def _get_plot_position(self: DistributionVisualizer, index: int, cols_per_row: int) -> tuple[int, int]:
        """Get the position (row, col) for a subplot based on its index and columns per row.

        Parameters:
            index (int): The index of the subplot.
            cols_per_row (int): Number of plots per row in the figure layout.

        Returns:
            Tuple[int, int]: A tuple containing the row and column positions for the subplot.
        """
        row = (index - 1) // cols_per_row + 1
        col = (index - 1) % cols_per_row + 1
        return row, col

    def _get_temp_figure(self: DistributionVisualizer, feature: str) -> Figure:
        """Generate a temporary figure for a specific feature.

        Parameters:
            feature (str): The feature name for which the temporary figure will be generated.

        Returns:
            Figure: A plotly.graph_objs._figure.Figure object containing the temporary figure.
        """
        return get_column_plot(
            real_data=self.real_data,
            synthetic_data=self.synthetic_data,
            column_name=feature,
        )

    def _add_traces_to_figure(
        self: DistributionVisualizer,
        fig: Figure,
        temp_fig: Figure,
        index: int,
        row: int,
        col: int,
    ) -> None:
        """Add traces from a temporary figure to the main figure.

        Parameters:
            fig (Figure): The main figure to which traces will be added.
            temp_fig (Figure): The temporary figure containing traces to be added.
            index (int): The index of the subplot.
            row (int): The row position of the subplot.
            col (int): The column position of the subplot.
        """
        for trace in temp_fig.data:
            trace.showlegend = index == 1
            trace.legendgroup = trace.name
            fig.add_trace(trace, row=row, col=col)

    def _finalize_figure(
        self: DistributionVisualizer,
        fig: Figure,
        plot_width: int,
        plot_height: int,
        title: str,
    ) -> None:
        """Finalize the figure layout and appearance.

        Parameters:
            fig (Figure): The figure to be finalized.
            plot_width (int): The width of the plot.
            plot_height (int): The height of the plot.
            title (str): The title of the figure.
        """
        for annotation in fig.layout.annotations:
            annotation.font.size = 8

        # Adjust the layout
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title_text=title,
            margin={"b": 10, "l": 20, "r": 20, "t": 60},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "xanchor": "center", "x": 0.5},
            font_size=8,
        )

    def visualize_features(
        self: DistributionVisualizer,
        id_feature: str | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        """High-level method to visualize distribution plots for all feature categories.

        This method prepares the data based on the provided parameters and then creates and displays distribution
        plots for categorical, numeric, and other features.

        Parameters:
            id_feature (str | None): The name of the ID feature to be removed from the datasets, if any.
            selected_features (list[str] | None): A list of feature names to be visualized. If None, all features
                                                  are considered.
        """
        self.prepare_data(id_feature, selected_features)

        fig_configs: list[tuple[str, list[str], int]] = [
            ("Categorical Features", self.categorical_features, 5),
            ("Numeric Features", self.numeric_features, 4),
            ("Other Features", self.other_features, 2),
        ]

        for title, features, cols_per_row in fig_configs:
            fig = self.create_category_plots(features, title, cols_per_row)
            if fig:
                fig.show("notebook")
