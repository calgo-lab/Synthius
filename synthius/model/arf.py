from __future__ import annotations

from logging import getLogger

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.impute import SimpleImputer

logger = getLogger()

# flake8: noqa
# The ARF class was moved directly from the ARF library, so all linter errors are ignored, and nothing was changed.


def bnd_fun(tree: int, p: int, forest: RandomForestRegressor, feature_names: list[str]) -> pd.DataFrame:
    """Computes the lower and upper bounds for each node in a decision tree within a random forest.

    Parameters
    ----------
    tree : int
        The index of the tree within the forest.
    p : int
        The number of features.
    forest : RandomForestRegressor
        The random forest model containing the tree.
    feature_names : list[str]
        List of feature names corresponding to the features used by the forest.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing the bounds for each leaf node, with columns 'tree', 'leaf', 'variable', 'min', and 'max'.
    """
    my_tree = forest.estimators_[tree].tree_
    num_nodes = my_tree.node_count

    lb = np.full(shape=(num_nodes, p), fill_value=float("-inf"))
    ub = np.full(shape=(num_nodes, p), fill_value=float("inf"))

    for i in range(num_nodes):
        left_child = my_tree.children_left[i]
        right_child = my_tree.children_right[i]

        if left_child > -1:  # leaf nodes are indicated by -1
            ub[left_child, :] = ub[right_child, :] = ub[i, :]
            lb[left_child, :] = lb[right_child, :] = lb[i, :]

            if left_child != right_child:
                # If no pruned node, split changes bounds
                ub[left_child, my_tree.feature[i]] = lb[right_child, my_tree.feature[i]] = my_tree.threshold[i]

    leaves = np.where(my_tree.children_left < 0)[0]

    # Lower and upper bounds, to long format, to single DataFrame for return
    lower = pd.concat(
        [
            pd.Series(np.full(shape=(leaves.shape[0]), fill_value=tree), name="tree"),
            pd.Series(leaves, name="leaf"),
            pd.DataFrame(lb[leaves,], columns=feature_names),
        ],
        axis=1,
    )

    upper = pd.concat(
        [
            pd.Series(np.full(shape=(leaves.shape[0]), fill_value=tree), name="tree"),
            pd.Series(leaves, name="leaf"),
            pd.DataFrame(ub[leaves,], columns=feature_names),
        ],
        axis=1,
    )

    # Using .merge method instead of pd.merge function
    ret = pd.melt(lower, id_vars=["tree", "leaf"], value_name="min").merge(
        pd.melt(upper, id_vars=["tree", "leaf"], value_name="max"),
        on=["tree", "leaf", "variable"],
    )

    del lower, upper
    return ret


class DataProcessor:
    """A class for processing and transforming data, including handling missing values.

    Attributes:
    ----------
    data : pd.DataFrame
        The input data to be processed.
    id_column : str | None
        The name of the ID column, if any.
    imputers : dict
        A dictionary to store imputers for continuous columns.
    fill_values : dict
        A dictionary to store fill values for continuous columns.
    missing_value_proportions : dict
        A dictionary to store missing value proportions for continuous columns.
    decimal_places : dict
        A dictionary to store decimal places for continuous columns.
    transformed_data : pd.DataFrame | None
        The transformed version of the input data.
    categorical_placeholder : str
        Placeholder for missing categorical values.
    continuous_columns : list
        List of continuous column names.
    categorical_columns : list
        List of categorical column names.
    original_id_values : pd.Series | None
        The original ID column values to be reintroduced later.
    id_column_index : int | None
        The original index position of the ID column.
    """

    def __init__(self: DataProcessor, data: pd.DataFrame, id_column: str | None = None) -> None:
        """Initializes the DataProcessor with the provided data and ID column.

        Parameters
        ----------
        data : pd.DataFrame
            The input data to be processed.
        id_column : str | None
            The name of the ID column, if any.
        """
        self.data = data
        self.id_column = id_column
        self.imputers: dict[str, SimpleImputer] = {}
        self.fill_values: dict[str, float] = {}
        self.missing_value_proportions: dict[str, float] = {}
        self.decimal_places: dict[str, int] = {}
        self.transformed_data: pd.DataFrame | None = None
        self.categorical_placeholder = "missed"
        self.continuous_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.original_id_values: pd.Series | None = None
        self.id_column_index: int | None = None

        if self.id_column:
            if self.id_column in self.data.columns:
                self.original_id_values = self.data[self.id_column].copy()
                self.id_column_index = self.data.columns.get_loc(self.id_column)
                self.data = self.data.drop(self.id_column, axis=1)
            else:
                logger.warning("The ID column %s does not exist in the dataset.", self.id_column)

    def is_continuous(self: DataProcessor, series: pd.Series) -> bool:
        """Determines if a series is continuous by checking if any values have decimals.

        Parameters
        ----------
        series : pd.Series
            The series to check.

        Returns:
        -------
        bool
            True if the series is continuous, otherwise False.
        """
        series = pd.to_numeric(series, errors="coerce").dropna()
        return series.apply(lambda x: x % 1 != 0).any()

    def get_decimal_places(self: DataProcessor, series: pd.Series) -> int:
        """Gets the maximum number of decimal places in the series.

        Parameters
        ----------
        series : pd.Series
            The series from which to calculate decimal places.

        Returns:
        -------
        int
            Maximum number of decimal places in the series.
        """
        series = series.dropna().astype(str)
        decimals = series.apply(lambda x: len(x.split(".")[1]) if "." in x else 0)
        return int(decimals.max())

    def fit_transform(self: DataProcessor) -> pd.DataFrame:
        """Transforms the data by imputing missing values and converting categorical data.

        Returns:
        -------
        pd.DataFrame
            The transformed version of the input data.
        """
        self.transformed_data = self.data.copy()

        for col in self.data.columns:
            if self.is_continuous(self.data[col]):
                self.continuous_columns.append(col)

                missing_value_proportion = self.data[col].isna().mean()
                self.missing_value_proportions[col] = missing_value_proportion

                self.decimal_places[col] = self.get_decimal_places(self.data[col])

                imputer = SimpleImputer(strategy="median")
                self.transformed_data[col] = imputer.fit_transform(self.data[[col]]).ravel()
                self.fill_values[col] = imputer.statistics_[0]
                self.imputers[col] = imputer

                # Round to maintain the original decimal precision
                self.transformed_data[col] = self.transformed_data[col].round(self.decimal_places[col])
            else:
                self.categorical_columns.append(col)

                # Convert to categories and replace NaN with the placeholder
                self.transformed_data[col] = (
                    self.data[col].astype("category").cat.add_categories([self.categorical_placeholder]).fillna(self.categorical_placeholder)
                )
        return self.transformed_data

    def inverse_transform(self: DataProcessor, transformed_data: pd.DataFrame) -> pd.DataFrame:
        """Inverses the transformation process, reintroducing missing values and ID column.

        Parameters
        ----------
        transformed_data : pd.DataFrame
            The transformed data to revert.

        Returns:
        -------
        pd.DataFrame
            The original version of the input data with ID column if specified.
        """
        original_transformed = transformed_data.copy()
        rng = np.random.default_rng()

        for col in self.continuous_columns:
            filled_value = self.fill_values[col]
            original_missing_count = int(self.missing_value_proportions[col] * len(self.data))

            filled_indices = original_transformed.index[original_transformed[col] == filled_value].tolist()
            available_fill_count = len(filled_indices)

            if available_fill_count >= original_missing_count:
                reintroduce_indices = rng.choice(filled_indices, original_missing_count, replace=False)
            else:
                proportion_to_reintroduce = available_fill_count / original_missing_count
                reintroduce_count = int(proportion_to_reintroduce * available_fill_count)
                reintroduce_indices = rng.choice(filled_indices, reintroduce_count, replace=False)

            original_transformed.loc[reintroduce_indices, col] = np.nan

            # Round to maintain the original decimal precision
            original_transformed[col] = original_transformed[col].round(self.decimal_places[col])

        for col in self.categorical_columns:
            original_transformed[col] = original_transformed[col].replace(self.categorical_placeholder, np.nan)

        # Reintroduce the original ID column values at the correct index
        if self.id_column_index is not None:
            rng = np.random.default_rng()
            random_ids = [f"ID-{rng.integers(10000000, 99999999)}" for _ in range(len(original_transformed))]
            original_transformed.insert(self.id_column_index, self.id_column, random_ids)

        return original_transformed


class ARF:
    """Implements Adversarial Random Forests (ARF) for data generation and density estimation.

    This class allows you to fit an ARF model, estimate data density, and generate synthetic data.

    Attributes:
    ----------
    x : pd.DataFrame
        Input data to fit the ARF model.
    num_trees : int
        Number of trees to grow in each forest.
    delta : float
        Tolerance parameter for convergence.
    max_iters : int
        Maximum iterations for the adversarial loop.
    early_stop : bool
        Whether to terminate the loop if performance fails to improve.
    verbose : bool
        Whether to print discriminator accuracy after each round.
    min_node_size : int
        Minimum number of samples in terminal nodes.

    Methods:
    -------
    forde(dist='truncnorm', oob=False, alpha=0)
        Performs density estimation using the ARF model.
    forge(n)
        Generates synthetic data based on the fitted ARF model.
    """

    def __init__(  # type: ignore
        self,
        x: pd.DataFrame,
        id_column: str | None = None,
        num_trees: int = 30,
        delta: float = 0.0,
        max_iters: int = 10,
        min_node_size: int = 5,
        *,
        early_stop: bool = True,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """Initializes the ARF class with input data and model parameters.

        Parameters
        ----------
        x : pd.DataFrame
            The input data to be processed.
        id_column : str | None
            The name of the ID column, if any.
        num_trees : int, optional
            Number of trees to grow in each forest (default is 30).
        delta : float, optional
            Tolerance parameter for convergence (default is 0).
        max_iters : int, optional
            Maximum iterations for the adversarial loop (default is 10).
        early_stop : bool, optional
            Whether to terminate loop if performance fails to improve (default is True).
        verbose : bool, optional
            Whether to print discriminator accuracy after each round (default is True).
        min_node_size : int, optional
            Minimum number of samples in terminal nodes (default is 5).
        """
        # Assertions
        assert isinstance(x, pd.core.frame.DataFrame), f"Expected pandas DataFrame as input, got: {type(x)}"
        assert len(set(list(x))) == x.shape[1], "Every column must have a unique column name"
        assert max_iters >= 0, "Negative number of iterations is not allowed: parameter max_iters must be >= 0"
        assert min_node_size > 0, "Minimum number of samples in terminal nodes (parameter min_node_size) must be greater than zero"
        assert num_trees > 0, "Number of trees in the random forest (parameter num_trees) must be greater than zero"
        assert 0 <= delta <= 0.5, "Parameter delta must be in range 0 <= delta <= 0.5"

        if id_column is not None:
            self.processor = DataProcessor(x, id_column)
        else:
            self.processor = DataProcessor(x)

        # Preprocess the data
        x_real = self.processor.fit_transform()

        self.p = x_real.shape[1]
        self.orig_colnames = list(x_real)
        self.num_trees = num_trees

        # Find object columns and convert to category
        self.object_cols = x_real.dtypes == "object"
        for col in list(x_real):
            if self.object_cols[col]:
                x_real[col] = x_real[col].astype("category")

        # Find factor columns
        self.factor_cols = x_real.dtypes == "category"

        # Save factor levels
        self.levels = {}
        for col in list(x_real):
            if self.factor_cols[col]:
                self.levels[col] = x_real[col].cat.categories

        # Recode factors to integers
        for col in list(x_real):
            if self.factor_cols[col]:
                x_real[col] = x_real[col].cat.codes

        # If no synthetic data provided, sample from marginals
        x_synth = x_real.apply(lambda x: x.sample(frac=1).to_numpy())

        # Merge real and synthetic data
        x = pd.concat([x_real, x_synth])
        y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])
        # Real observations = 0, synthetic observations = 1

        # Pass on x_real
        self.x_real = x_real

        # Fit initial RF model
        clf_0 = RandomForestClassifier(
            oob_score=True,
            n_estimators=self.num_trees,
            min_samples_leaf=min_node_size,
            **kwargs,
        )
        clf_0.fit(x, y)

        iters = 0

        acc_0 = clf_0.oob_score_  # is accuracy directly
        acc = [acc_0]

        if verbose:
            print(f"Initial accuracy is {acc_0}")

        if acc_0 > 0.5 + delta and iters < max_iters:
            converged = False
            while not converged:  # Start adversarial loop
                # Get nodeIDs
                nodeIDs = clf_0.apply(self.x_real)  # dimension [terminalnode, tree]

                # Add observation ID to x_real
                x_real_obs = x_real.copy()
                x_real_obs["obs"] = range(x_real.shape[0])

                # Add observation ID to nodeIDs
                nodeIDs_pd = pd.DataFrame(nodeIDs)
                tmp = nodeIDs_pd.copy()
                tmp["obs"] = range(x_real.shape[0])
                tmp = tmp.melt(id_vars=["obs"], value_name="leaf", var_name="tree")

                # Match real data to trees and leaves (node id for tree)
                x_real_obs = x_real_obs.merge(tmp, on=["obs"], sort=False)
                x_real_obs = x_real_obs.drop("obs", axis=1)

                # Sample leaves
                tmp = tmp.drop("obs", axis=1)
                tmp = tmp.sample(x_real.shape[0], axis=0, replace=True)
                tmp = pd.Series(tmp.value_counts(sort=False), name="cnt").reset_index()
                draw_from = tmp.merge(x_real_obs, on=["tree", "leaf"], sort=False)

                # Sample synthetic data from leaf
                grpd = draw_from.groupby(["tree", "leaf"])
                x_synth = [
                    grpd.get_group(ind).apply(
                        lambda x: x.sample(n=grpd.get_group(ind)["cnt"].iloc[0], replace=True).to_numpy(),
                    )
                    for ind in grpd.indices
                ]
                x_synth = pd.concat(x_synth).drop(["cnt", "tree", "leaf"], axis=1)

                # Delete unnecessary objects
                del nodeIDs, nodeIDs_pd, tmp, x_real_obs, draw_from

                # Merge real and synthetic data
                x = pd.concat([x_real, x_synth])
                y = np.concatenate([np.zeros(x_real.shape[0]), np.ones(x_real.shape[0])])

                # Discriminator
                clf_1 = RandomForestClassifier(
                    oob_score=True,
                    n_estimators=self.num_trees,
                    min_samples_leaf=min_node_size,
                    **kwargs,
                )
                clf_1.fit(x, y)

                # Update iters and check for convergence
                acc_1 = clf_1.oob_score_

                acc.append(acc_1)

                iters += 1
                plateau = bool(early_stop and acc[iters] > acc[iters - 1])
                if verbose:
                    print(f"Iteration number {iters} reached accuracy of {acc_1}.")
                if acc_1 <= 0.5 + delta or iters >= max_iters or plateau:
                    converged = True
                else:
                    clf_0 = clf_1
        self.clf = clf_0
        self.acc = acc

        # Pruning
        pred = self.clf.apply(self.x_real)
        for tree_num in range(self.num_trees):
            tree = self.clf.estimators_[tree_num]
            left = tree.tree_.children_left
            right = tree.tree_.children_right
            leaves = np.where(left < 0)[0]

            # Get leaves that are too small
            unique, counts = np.unique(pred[:, tree_num], return_counts=True)
            to_prune = unique[counts < min_node_size]

            # Also add leaves with 0 obs.
            to_prune = np.concatenate([to_prune, np.setdiff1d(leaves, unique)])

            while len(to_prune) > 0:
                for tp in to_prune:
                    # Find parent
                    parent = np.where(left == tp)[0]
                    if len(parent) > 0:
                        # Left child
                        left[parent] = right[parent]
                    else:
                        # Right child
                        parent = np.where(right == tp)[0]
                        right[parent] = left[parent]
                # Prune again if child was pruned
                to_prune = np.where(np.isin(left, to_prune))[0]

    def forde(self, dist: str = "truncnorm", *, oob: bool = False, alpha: float = 0) -> dict:
        """Performs density estimation using the ARF model.

        Parameters
        ----------
        dist : str, optional
            Distribution to use for density estimation of continuous features (default is "truncnorm").
        oob : bool, optional
            Only use out-of-bag samples for parameter estimation (default is False).
        alpha : float, optional
            Optional pseudocount for Laplace smoothing of categorical features (default is 0).

        Returns:
        -------
        dict
            A dictionary containing the estimated density parameters.
        """
        self.dist = dist
        self.oob = oob
        self.alpha = alpha

        # Get terminal nodes for all observations
        pred = self.clf.apply(self.x_real)

        # If OOB, use only OOB trees
        if self.oob:
            for tree in range(self.num_trees):
                idx_oob = np.isin(
                    range(self.x_real.shape[0]),
                    _generate_unsampled_indices(
                        self.clf.estimators_[tree].random_state,
                        self.x.shape[0],  # type: ignore
                        self.x.shape[0],  # type: ignore
                    ),
                )
                pred[np.invert(idx_oob), tree] = -1

        # Compute leaf bounds and coverage
        bnds = pd.concat(
            [bnd_fun(tree=j, p=self.p, forest=self.clf, feature_names=self.orig_colnames) for j in range(self.num_trees)],
        )
        bnds["f_idx"] = bnds.groupby(["tree", "leaf"]).ngroup()

        bnds_2 = pd.DataFrame()
        for t in range(self.num_trees):
            unique, freq = np.unique(pred[:, t], return_counts=True)
            vv = pd.concat([pd.Series(unique, name="leaf"), pd.Series(freq / pred.shape[0], name="cvg")], axis=1)
            zz = bnds[bnds["tree"] == t]
            bnds_2 = pd.concat([bnds_2, vv.merge(zz, on=["leaf"])])
        bnds = bnds_2
        del bnds_2

        # Set coverage for nodes with single observations to zero
        if np.invert(self.factor_cols).any():
            bnds.loc[bnds["cvg"] == 1 / pred.shape[0], "cvg"] = 0

        # No parameters to learn for zero coverage leaves - drop zero coverage nodes
        bnds = bnds[bnds["cvg"] > 0]

        # Rename leaves to nodeids
        bnds = bnds.rename(columns={"leaf": "nodeid"})

        # Save bounds to later use coverage for drawing new samples
        self.bnds = bnds
        # Fit continuous distribution in all terminal nodes
        self.params = pd.DataFrame()
        if np.invert(self.factor_cols).any():
            for tree in range(self.num_trees):
                dt = self.x_real.loc[:, np.invert(self.factor_cols)].copy()
                dt["tree"] = tree
                dt["nodeid"] = pred[:, tree]
                # Merge bounds and make it long format
                long = (
                    dt[dt["nodeid"] >= 0]
                    .melt(id_vars=["tree", "nodeid"])
                    .merge(
                        bnds[["tree", "nodeid", "variable", "min", "max", "f_idx"]],
                        on=["tree", "nodeid", "variable"],
                        how="left",
                    )
                )
                # Get distribution parameters
                if self.dist == "truncnorm":
                    res = long.groupby(["tree", "nodeid", "variable"], as_index=False).agg(
                        mean=("value", "mean"),
                        sd=("value", "std"),
                        min=("min", "min"),
                        max=("max", "max"),
                    )
                else:
                    raise ValueError("Unknown distribution, make sure to enter a valid value for dist")
                self.params = pd.concat([self.params, res])

        # Get class probabilities in all terminal nodes
        self.class_probs = pd.DataFrame()
        if self.factor_cols.any():
            for tree in range(self.num_trees):
                dt = self.x_real.loc[:, self.factor_cols].copy()
                dt["tree"] = tree
                dt["nodeid"] = pred[:, tree]
                dt = pd.melt(dt[dt["nodeid"] >= 0], id_vars=["tree", "nodeid"])
                long = dt.merge(bnds, on=["tree", "nodeid", "variable"])
                long["count_var"] = long.groupby(["tree", "nodeid", "variable"])["variable"].transform("count")
                long["count_var_val"] = long.groupby(["tree", "nodeid", "variable", "value"])["variable"].transform(
                    "count",
                )
                long = long.drop_duplicates()
                if self.alpha == 0:
                    long["prob"] = long["count_var_val"] / long["count_var"]
                else:
                    # Define the range of each variable in each leaf
                    long["k"] = long.groupby(["variable"])["value"].transform("nunique")
                    long.loc[long["min"] == float("-inf"), "min"] = 0.5 - 1
                    long.loc[long["max"] == float("inf"), "max"] = long["k"] + 0.5 - 1
                    long.loc[round(long["min"] % 1, 2) != 0.5, "min"] = long["min"] - 0.5
                    long.loc[round(long["max"] % 1, 2) != 0.5, "min"] = long["max"] + 0.5
                    long["k"] = long["max"] - long["min"]
                    # Enumerate each possible leaf-variable-value combo
                    tmp = long[["f_idx", "tree", "nodeid", "variable", "min", "max"]].copy()
                    tmp["rep_min"] = tmp["min"] + 0.5
                    tmp["rep_max"] = tmp["max"] - 0.5
                    tmp["levels"] = tmp.apply(
                        lambda row: list(range(int(row["rep_min"]), int(row["rep_max"] + 1))),
                        axis=1,
                    )
                    tmp = tmp.explode("levels")
                    cat_val = pd.DataFrame(self.levels).melt()
                    cat_val["levels"] = cat_val["value"]
                    tmp = tmp.merge(cat_val, on=["variable", "levels"])[["variable", "f_idx", "tree", "nodeid", "value"]]
                    # Populate count, k
                    tmp = tmp.merge(
                        long[["f_idx", "variable", "tree", "nodeid", "count_var", "k"]],
                        on=["f_idx", "nodeid", "variable", "tree"],
                    )
                    # Merge with long, set val_count = 0 for possible but unobserved levels
                    long = tmp.merge(
                        long,
                        on=["f_idx", "tree", "nodeid", "variable", "value", "count_var", "k"],
                        how="left",
                    )
                    long.loc[long["count_var_val"].isna(), "count_var_val"] = 0
                    long = long[["f_idx", "tree", "nodeid", "variable", "value", "count_var_val", "count_var", "k"]].drop_duplicates()
                    # Compute posterior probabilities
                    long["prob"] = (long["count_var_val"] + self.alpha) / (long["count_var"] + self.alpha * long["k"])
                    long["value"] = long["value"].astype("int8")

                long = long[["f_idx", "tree", "nodeid", "variable", "value", "prob"]]
                self.class_probs = pd.concat([self.class_probs, long])
        return {
            "cnt": self.params,
            "cat": self.class_probs,
            "forest": self.clf,
            "meta": pd.DataFrame(data={"variable": self.orig_colnames, "family": self.dist}),
        }

    def forge(self, n: int) -> pd.DataFrame:
        """Generates synthetic data based on the fitted ARF model.

        Parameters
        ----------
        n : int
            Number of synthetic samples to generate.

        Returns:
        -------
        pd.DataFrame
            The generated synthetic data.
        """
        if not hasattr(self, "bnds"):
            raise AttributeError("Need density estimates to generate data -- run .forde() first!")

        # Sample new observations and get their terminal nodes
        # Draw random leaves with probability proportional to coverage

        rng = np.random.default_rng()

        unique_bnds = self.bnds[["tree", "nodeid", "cvg"]].drop_duplicates()

        # Generate the draws using the Generator instance
        draws = rng.choice(
            a=range(unique_bnds.shape[0]),
            p=unique_bnds["cvg"] / self.num_trees,
            size=n,
        )

        sampled_trees_nodes = unique_bnds[["tree", "nodeid"]].iloc[draws,].reset_index(drop=True).reset_index().rename(columns={"index": "obs"})

        # Get distributions parameters for each new obs.
        if np.invert(self.factor_cols).any():
            obs_params = sampled_trees_nodes.merge(self.params, on=["tree", "nodeid"]).sort_values(
                by=["obs"],
                ignore_index=True,
            )

        # Get probabilities for each new obs.
        if self.factor_cols.any():
            obs_probs = sampled_trees_nodes.merge(self.class_probs, on=["tree", "nodeid"]).sort_values(
                by=["obs"],
                ignore_index=True,
            )

        # Sample new data from mixture distribution over trees
        data_new = pd.DataFrame(index=range(n), columns=range(self.p))
        for j in range(self.p):
            colname = self.orig_colnames[j]

            if self.factor_cols.iloc[j]:
                # Factor columns: Multinomial distribution
                data_new.isetitem(
                    j,
                    obs_probs[obs_probs["variable"] == colname].groupby("obs").sample(weights="prob")["value"].reset_index(drop=True),
                )

            elif self.dist == "truncnorm":
                # Set minimum bound to zero for non-negative features
                myclip_a = obs_params.loc[obs_params["variable"] == colname, "min"].clip(lower=0)
                myclip_b = obs_params.loc[obs_params["variable"] == colname, "max"]
                myloc = obs_params.loc[obs_params["variable"] == colname, "mean"]
                myscale = obs_params.loc[obs_params["variable"] == colname, "sd"]

                # Avoid division by zero in case of zero standard deviation
                myscale = myscale.replace(0, 1)

                data_new.isetitem(
                    j,
                    truncnorm(
                        a=(myclip_a - myloc) / myscale,
                        b=(myclip_b - myloc) / myscale,
                        loc=myloc,
                        scale=myscale,
                    ).rvs(size=n),
                )
                del myclip_a, myclip_b, myloc, myscale
            else:
                raise ValueError("Other distributions not yet implemented")

        # Use original column names
        data_new = data_new.set_axis(self.orig_colnames, axis=1, copy=False)

        # Convert categories back to category
        for col in self.orig_colnames:
            if self.factor_cols[col]:
                data_new[col] = pd.Categorical.from_codes(data_new[col], categories=self.levels[col])

        # Convert object columns back to object
        for col in self.orig_colnames:
            if self.object_cols[col]:
                data_new[col] = data_new[col].astype("object")

        # Reverse transformation using the processor and Return newly sampled data
        return self.processor.inverse_transform(data_new)
