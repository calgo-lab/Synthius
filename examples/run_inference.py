import json
from pathlib import Path

import pandas as pd

from synthius.metric import InferenceMetric
from synthius.metric.utils import utils

datasets = {"adult": "adult",
            "AIDS": "aids",
            "charite": "charite",
            "diabetes": "diabetes",
            "EGZB": "egzb",
            "MIMIC": "mimic"}

base_path = Path.home() / "projects/KIP-SDM/synthetic"
ori_base_path = base_path / "data"
synt_base_path = base_path / "data/synthetic"
results_base_path = base_path / "results/attribute_inference"
n_attacks = 6_000


def get_synt_paths(synt_path: Path) -> list[Path]:
    return [
        synt_path / "ARF.csv",
        synt_path / "CopulaGAN.csv",
        synt_path / "CTGAN.csv",
        synt_path / "GaussianCopula.csv",
        synt_path / "GaussianMultivariate.csv",
        synt_path / "TVAE.csv",
        synt_path / "WGAN.csv",
    ]


def run_for_secret(ori_path: Path,
                   synt_paths: list[Path],
                   secret: str,
                   aux_columns: list[str]) -> list[dict[str, str | float]]:
    """Runs the Inference attacks for a single `secret` feature.

    :param ori_path: The path where the original data is.
    :param synt_paths: The path where the synthetic data is.
    :param secret: The "sensitive" parameter which should be used for the attack.
    :param aux_columns: The auxiliary columns used for the attack.
    :return:
        The results from the attack in a form of a list of dictionaries (InferenceMetric::results for more details).
    """
    return InferenceMetric(real_data_path=ori_path / "train.csv",
                           synthetic_data_paths=synt_paths,
                           control_data_path=ori_path / "test.csv",
                           secret=secret,
                           aux_cols=aux_columns,
                           n_attacks=n_attacks).results


def run_single_inference(ori_name: str, synt_name: str) -> None:
    ori_path = ori_base_path / ori_name
    control_path = ori_path / "test.csv"
    synt_paths = get_synt_paths(synt_base_path / synt_name)
    results_path = results_base_path / synt_name
    results_path.mkdir(parents=True, exist_ok=True)

    all_columns = utils.clean_columns(pd.read_csv(control_path)).columns
    metrics_per_secret = {}
    for secret in all_columns:
        print(secret)
        metrics_per_secret[secret] = []
        aux_columns = [c for c in all_columns if c != secret]
        output = run_for_secret(ori_path, synt_paths, secret, aux_columns)
        metrics_per_secret[secret].append(output)
        with open(results_path / f"inference_{secret}.json", "w") as f:
            print(f"Storing results for {ori_name}::{secret} under {results_path / f'inference_{secret}.json'}")
            json.dump(output, f, indent=2)

    with open(results_path / "inference.json", "w") as f:
        print(f"Storing all results for {ori_name} under {results_path / 'inference.json'}")
        json.dump(metrics_per_secret, f, indent=2)


if __name__ == '__main__':
    for k, v in datasets.items():
        run_single_inference(k, v)
