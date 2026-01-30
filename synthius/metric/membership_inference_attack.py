import os
from logging import getLogger
from pathlib import Path

import numpy as np
import pandas as pd
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, ShadowModels
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier
from art.utils import to_categorical
from category_encoders import OneHotEncoder
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

from synthius.metric.utils import BaseMetric

logger = getLogger()

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_true: int, n_false: int) -> dict:
    """Calculates the classification metrics and returns them as a dictionary.

    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param n_true: The number of true labels.
    :param n_false: The number of false labels.
    :return:
    """
    logger.info(classification_report(y_pred=y_pred, y_true=y_true))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn)  # recall for members
    fpr = fp / (fp + tn)  # non-members predicted as members
    tnr = tn / (tn + fp)  # recall for non members
    ppv = tp / (tp + fp)  # precision for members
    acc = (tp + tn) / (tp + tn + fp + fn)  # acc

    advantage = tpr - fpr  # https://arxiv.org/pdf/2010.12112

    return {
        "attack_accuracy": acc,
        "member_tpr": tpr,
        "member_ppv": ppv,
        "nonmember_tnr": tnr,
        "nonmember_fpr": fpr,
        "advantage": advantage,
        "n_eval_members": n_true,
        "n_eval_nonmembers": n_false,
    }


class MIABbox:
    """A Black Box Membership Inference Attack (MIA).

    Implements a Black Box MIA where a target (downstream) model is trained on the `train_data`.
    Then a MembershipInferenceBlackBox model is trained on a subset of the `train_data` and a subset of the `test_data`.
    A black box attack is then performed on the disjoint subset of `train_data` and a disjoint subset of `test_data`.

    https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html

    """

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, label: str, attack_train_ratio: float = 0.5) -> None:
        """A Black Box Membership Inference Attack (MIA).

        :param train_data: The training data for both target and attack model.
        :param test_data: The test data for both target and attack model.
        :param label: The target label for the downstream task.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.label = label
        self._preprocess_data()

        self.attack_train_ratio = attack_train_ratio
        self.art_classifier = None

    def _preprocess_data(self) -> None:
        """Preprocesses the data with a sklearn preprocessing pipeline."""
        input_cols = [col for col in self.train_data.columns if col != self.label]
        categorical_cols = self.train_data[input_cols].select_dtypes(include=["object", "category"]).columns
        numerical_cols = self.train_data[input_cols].select_dtypes(exclude=["object", "category"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", StandardScaler(), numerical_cols),
            ]
        )

        preprocessor.fit(pd.concat([self.train_data[input_cols], self.test_data[input_cols]]))

        label_encoder = LabelEncoder()
        self.x_train = preprocessor.transform(self.train_data[input_cols])
        self.y_train = label_encoder.fit_transform(self.train_data[self.label])

        self.x_test = preprocessor.transform(self.test_data[input_cols])
        self.y_test = label_encoder.transform(self.test_data[self.label])

    def train_target_model(self) -> None:
        """Trains the target, downstream RandomForestClassifier, and wraps it in ScikitlearnRandomForestClassifier."""
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

        model.fit(self.x_train, self.y_train)
        self.art_classifier = ScikitlearnRandomForestClassifier(model)

    def attack(self) -> None:
        """Performs the Black Box attack."""
        if self.art_classifier is None:
            self.train_target_model()

        # Extract the train data and test data splits for the attack (and leave the rest for holdout datasets).
        self.attack_train_size = int(self.x_train.shape[0] * self.attack_train_ratio)
        self.attack_test_size = int(self.x_test.shape[0] * self.attack_train_ratio)

        self.bb_attack = MembershipInferenceBlackBox(self.art_classifier)
        self.bb_attack.fit(
            self.x_train[: self.attack_train_size],
            self.y_train[: self.attack_train_size],
            self.x_test[: self.attack_test_size],
            self.y_test[: self.attack_test_size],
        )

    def infer(self) -> dict[str, float]:
        """Infers the membership of the holdout train and holdout test sets."""
        if self.bb_attack is None:
            raise ValueError("An attack-attack() needs to be performed before inference.")  # noqa: TRY003, EM101
        inferred_members = self.bb_attack.infer(self.x_train[self.attack_train_size :], self.y_train[self.attack_train_size :])
        inferred_nonmembers = self.bb_attack.infer(self.x_test[self.attack_test_size :], self.y_test[self.attack_test_size :])

        y_pred = np.concatenate((inferred_members, inferred_nonmembers))
        y_true = np.concatenate((np.ones_like(inferred_members), np.zeros_like(inferred_nonmembers)))

        return calculate_metrics(y_true=y_true, y_pred=y_pred, n_true=len(inferred_members), n_false=len(inferred_nonmembers))


# ruff: noqa: PLR0913
class MIAShadow:
    """A Shadow-based Membership Inference Attack (MIA).

    Given the `shadow_data` implements `n_shadow_models` models.
    Based on these we derive labeled train and test data for the attacks (member vs nonmember).
    Then a MembershipInferenceBlackBox model is fit on the shadow data.
    A black box attack is then performed on original `train_data` and `test_data`.

    https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/attribute_inference.html

    """

    def __init__(
        self, train_data: pd.DataFrame, shadow_data: pd.DataFrame, test_data: pd.DataFrame, label: str, n_shadow_models: int = 3, random_state: int = 42
    ) -> None:
        """A Shadow-based Membership Inference Attack (MIA).

        :param train_data: The original training data used for the attack inference only.
        :param shadow_data: The shadow (synthetic) data used for generating labeled member/nonmember data.
        :param test_data: The original test data split used for the attack inference only.
        :param label: The target label for the downstream task.
        :param n_shadow_models: The number of shadow models to be trained, default 3.
        :param random_state: The random state for the models, default 42.
        """
        self.train_data = train_data
        self.shadow_data = shadow_data
        self.test_data = test_data
        self.label = label
        self._preprocess_data()

        self.art_classifier = None
        self.shadow_models: ShadowModels | None = None

        self.n_shadows = n_shadow_models
        self.random_state = random_state

    def _preprocess_data(self) -> None:
        """Preprocesses the data with a sklearn preprocessing pipeline."""
        input_cols = [col for col in self.train_data.columns if col != self.label]
        categorical_cols = self.train_data[input_cols].select_dtypes(include=["object", "category"]).columns
        numerical_cols = self.train_data[input_cols].select_dtypes(exclude=["object", "category"]).columns

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", StandardScaler(), numerical_cols),
            ]
        )
        preprocessor.fit(pd.concat([self.train_data[input_cols], self.shadow_data[input_cols], self.test_data[input_cols]]))

        label_encoder = LabelEncoder()

        self.x_train = preprocessor.transform(self.train_data[input_cols])
        self.y_train = label_encoder.fit_transform(self.train_data[self.label])

        self.x_shadow = preprocessor.transform(self.shadow_data[input_cols])
        self.y_shadow = label_encoder.transform(self.shadow_data[self.label])

        self.x_test = preprocessor.transform(self.test_data[input_cols])
        self.y_test = label_encoder.transform(self.test_data[self.label])

    def train_target_model(self) -> None:
        """Trains the target, downstream RandomForestClassifier, and wraps it in ScikitlearnRandomForestClassifier."""
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=1)

        model.fit(self.x_train, self.y_train)
        self.art_classifier = ScikitlearnRandomForestClassifier(model)

    def train_shadow(self) -> None:
        """Trains the shadow model and extracts the shadow data for the attack."""
        if self.art_classifier is None:
            self.train_target_model()

        self.shadow_models = ShadowModels(self.art_classifier, num_shadow_models=self.n_shadows)

        shadow_dataset = self.shadow_models.generate_shadow_dataset(self.x_shadow, to_categorical(self.y_shadow, nb_classes=np.unique(self.y_shadow).shape[0]))
        ((self.member_x, self.member_y, self.member_predictions), (self.nonmember_x, self.nonmember_y, self.nonmember_predictions)) = shadow_dataset

    def attack(self) -> None:
        """Performs the Black Box attack with the shadow model derived data."""
        if self.shadow_models is None:
            self.train_shadow()

        self.bb_attack = MembershipInferenceBlackBox(self.art_classifier, attack_model_type="rf")
        self.bb_attack.fit(self.member_x, self.member_y, self.nonmember_x, self.nonmember_y, self.member_predictions, self.nonmember_predictions)

    def infer(self) -> dict[str, float]:
        """Runs the inference on the attack model and returns the classification metrics.

        :return:
            dict[str, float]: The classification metrics for the inference.
        """
        if self.bb_attack is None:
            raise ValueError("An attack-attack() needs to be performed before inference.")  # noqa: TRY003, EM101
        inferred_members = self.bb_attack.infer(self.x_train, self.y_train)
        inferred_nonmembers = self.bb_attack.infer(self.x_test, self.y_test)

        y_pred = np.concatenate((inferred_members, inferred_nonmembers))
        y_true = np.concatenate((np.ones_like(inferred_members), np.zeros_like(inferred_nonmembers)))

        return calculate_metrics(y_true=y_true, y_pred=y_pred, n_true=len(inferred_members), n_false=len(inferred_nonmembers))


class MIAMetric(BaseMetric):
    """A Metric for running Membership Inference Attacks (MIA), both Black Box and Shadow-based."""

    def __init__(self, train_data_path: Path, test_data_path: Path | None, synthetic_data_paths: list[Path], label: str, id_column: str | None) -> None:
        """A Metric for running Membership Inference Attacks (MIA), both Black Box and Shadow-based.

        :param train_data_path: The path to the train data.
        :param test_data_path: The path to the test data.
        :param synthetic_data_paths: The paths to the synthetic data.
        :param label: The label for the downstream task.
        :param id_column: The id column of the dataset.
        """
        if test_data_path is None:
            raise ValueError("Test data path must be provided.")  # noqa: TRY003, EM101
        self.train_data = pd.read_csv(train_data_path)
        self.test_data = pd.read_csv(test_data_path)
        self.synthetic_data_paths = synthetic_data_paths
        self.label = label
        self.id_column = id_column

        if self.id_column is not None:
            self.train_data = self.train_data.drop([self.id_column], axis=1)
            self.test_data = self.test_data.drop([self.id_column], axis=1)
        self.results = self._compute_results()
        self.pivoted_results = self.pivot_results()
        MIAMetric.__name__ = "MIA"

    def display_results(self) -> None:
        """Displays the evaluation results."""
        if self.pivoted_results is not None:
            display(self.pivoted_results)
        else:
            logger.info("No results to display.")

    def pivot_results(self) -> pd.DataFrame:
        """Returns `self.results` as a dataframe."""
        df_results = pd.DataFrame(self.results)

        available_metrics = [
            "BBox - TPR",
            "BBox - PPV",
            "BBox - FPR",
            "BBox - TNR",
            "BBox - Acc",
            "BBox - Adv",
            "Shadow - TPR",
            "Shadow - PPV",
            "Shadow - FPR",
            "Shadow - TNR",
            "Shadow - Acc",
            "Shadow - Adv",
        ]
        present_metrics = [m for m in available_metrics if m in df_results.columns]

        if not present_metrics:
            raise ValueError("No valid MIA metrics found in results. Check self.results.")  # noqa: TRY003, EM101

        df_melted = df_results.melt(
            id_vars=["Model Name"],
            value_vars=present_metrics,
            var_name="Metric",
            value_name="Value",
        )

        return df_melted.pivot_table(index="Metric", columns="Model Name", values="Value")

    def _compute_results(self) -> list[dict]:
        """Runs the MIA based attacks and computes the metrics for the attacks.

        For each synthetic dataset, runs the MIABbox - Black box attack and MIAShadow a shadow-based attack.
        For the original data run the MIABBox only.

        :return:
            list[dict]: A list of dictionaries containing B.

        """
        rows = []

        # Run a Black box attack on the original data.
        mia_basic = MIABbox(train_data=self.train_data, test_data=self.test_data, label=self.label)
        mia_basic.attack()
        bbox = mia_basic.infer()

        rows.append(
            {
                "Model Name": "Original",
                "BBox - Acc": bbox["attack_accuracy"],
                "BBox - TPR": bbox["member_tpr"],
                "BBox - PPV": bbox["member_ppv"],
                "BBox - FPR": bbox["nonmember_fpr"],
                "BBox - FNR": bbox["nonmember_tnr"],
                "BBox - Adv": bbox["advantage"],
            }
        )

        # For every synthetic dataset ...
        for synth in self.synthetic_data_paths:
            synthetic_data = pd.read_csv(synth)
            if self.id_column is not None:
                synthetic_data = synthetic_data.drop([self.id_column], axis=1)

            # ... run a Black box attack (trains on synthetic, tests on synthetic/real test) ...
            mia_basic = MIABbox(train_data=synthetic_data, test_data=self.test_data, label=self.label)
            mia_basic.attack()
            bbox = mia_basic.infer()

            # ... then run a shadow (trained on real train, shadows on synthetic, evaluated on real train/test) ...
            mia_shadow = MIAShadow(train_data=self.train_data, shadow_data=synthetic_data, test_data=self.test_data, label=self.label)
            mia_shadow.attack()
            shadow = mia_shadow.infer()

            rows.append(
                {
                    "Model Name": synth.stem,
                    "BBox - Acc": bbox["attack_accuracy"],
                    "BBox - TPR": bbox["member_tpr"],
                    "BBox - PPV": bbox["member_ppv"],
                    "BBox - FPR": bbox["nonmember_fpr"],
                    "BBox - FNR": bbox["nonmember_tnr"],
                    "BBox - Adv": bbox["advantage"],
                    "Shadow - Acc": shadow["attack_accuracy"],
                    "Shadow - TPR": shadow["member_tpr"],
                    "Shadow - PPV": shadow["member_ppv"],
                    "Shadow - FPR": shadow["nonmember_fpr"],
                    "Shadow - FNR": shadow["nonmember_tnr"],
                    "Shadow - Adv": shadow["advantage"],
                }
            )

        return rows
