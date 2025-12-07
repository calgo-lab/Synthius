import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tomli
import torch
from torch.utils.data import DataLoader

from synthius.data import DataImputationPreprocessor, TorchDataset
from synthius.model import Synthesizer
from synthius.TabDiff.src import get_categories
from synthius.TabDiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
from synthius.TabDiff.tabdiff.modules.main_modules import Model, UniModMLP
from synthius.TabDiff.tabdiff.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TABDIFF_BASE_PATH = Path(__file__).parent / "../TabDiff"
CONFIG_PATH = TABDIFF_BASE_PATH / "tabdiff/configs/tabdiff_configs.toml"


@dataclass
class DataConfig:  # noqa: D101
    dequant_dist: str
    int_dequant_factor: int


@dataclass
class UniModMLPParams:  # noqa: D101
    num_layers: int
    d_token: int
    n_head: int
    factor: int
    bias: bool
    dim_t: int
    use_mlp: bool
    d_numerical: int
    categories: list[int]


@dataclass
class SamplerParams:  # noqa: D101
    stochastic_sampler: bool
    second_order_correction: bool


@dataclass
class EDMParams:  # noqa: D101
    precond: bool
    sigma_data: float
    net_conditioning: str


@dataclass
class NoiseDistParams:  # noqa: D101
    P_mean: float
    P_std: float


@dataclass
class NoiseScheduleParams:  # noqa: D101
    sigma_min: float
    sigma_max: float
    rho: float
    eps_max: float
    eps_min: float
    rho_init: float
    rho_offset: float
    k_init: float
    k_offset: float


@dataclass
class DiffusionParams:  # noqa: D101
    num_timesteps: int
    scheduler: str
    cat_scheduler: str
    noise_dist: str
    sampler_params: SamplerParams
    edm_params: EDMParams
    noise_dist_params: NoiseDistParams
    noise_schedule_params: NoiseScheduleParams


@dataclass
class TrainMain:  # noqa: D101
    steps: int
    lr: float
    weight_decay: float
    ema_decay: float
    batch_size: int
    check_val_every: int
    lr_scheduler: str
    factor: float
    reduce_lr_patience: int
    closs_weight_schedule: str
    c_lambda: float
    d_lambda: float
    num_workers: int


@dataclass
class TrainConfig:  # noqa: D101
    main: TrainMain


@dataclass
class SampleConfig:  # noqa: D101
    batch_size: int


@dataclass
class TabDiffConfig:  # noqa: D101
    data: DataConfig
    unimodmlp_params: UniModMLPParams
    diffusion_params: DiffusionParams
    train: TrainConfig
    sample: SampleConfig
    model_save_path: str | Path
    result_save_path: str | Path


def load_config(path: Path) -> TabDiffConfig:
    """Loads the config from `path`.

    Parameters:
        path: Path
            The path to the config.

    Returns:
            TabDiffConfig
                The config loaded as TabDiffConfig.
    """
    with path.open("rb") as f:
        logger.warning("!!Loading the model config %s with the provided parameters in it!!", str(path))
        toml_data = tomli.load(f)
        # Build the dataclasses from the TOML
        data_cfg = DataConfig(**toml_data["data"])

        unimodmlp_params = UniModMLPParams(**toml_data["unimodmlp_params"], d_numerical=0, categories=[])

        diffusion_params = DiffusionParams(
            num_timesteps=toml_data["diffusion_params"]["num_timesteps"],
            scheduler=toml_data["diffusion_params"]["scheduler"],
            cat_scheduler=toml_data["diffusion_params"]["cat_scheduler"],
            noise_dist=toml_data["diffusion_params"]["noise_dist"],
            sampler_params=SamplerParams(**toml_data["diffusion_params"]["sampler_params"]),
            edm_params=EDMParams(**toml_data["diffusion_params"]["edm_params"]),
            noise_dist_params=NoiseDistParams(**toml_data["diffusion_params"]["noise_dist_params"]),
            noise_schedule_params=NoiseScheduleParams(**toml_data["diffusion_params"]["noise_schedule_params"]),
        )

        train_config = TrainConfig(main=TrainMain(**toml_data["train"]["main"]))
        sample_config = SampleConfig(**toml_data["sample"])

        return TabDiffConfig(
            data=data_cfg,
            unimodmlp_params=unimodmlp_params,
            diffusion_params=diffusion_params,
            train=train_config,
            sample=sample_config,
            model_save_path="",
            result_save_path="",
        )


class TabDiffSynthesizer(Synthesizer):
    """Tabular data synthesizer using a TabDiff::UnifiedCtimeDiffusion model."""

    def __init__(self, id_column: str | None = None, gpu: int = -1) -> None:
        """Initialize the TabDiffSynthesizer."""
        self.exp_name = "learnable_schedule"
        self.id_column = id_column
        if gpu != -1 and torch.cuda.is_available():
            self.device = f"cuda:{gpu}"
        else:
            self.device = "cpu"

        self.model: Trainer
        self.preprocessor: DataImputationPreprocessor
        self.name = "TabDiffSynthesizer"
        self.raw_config: TabDiffConfig = load_config(CONFIG_PATH)
        self.data_cols: list
        self.categories: np.ndarray

    def _update_config(self, data: pd.DataFrame) -> None:
        """Updates self.raw_config with metadata from `data`.

        Following the template config from TabDiff, we need to update it with metadata derived
        from the categorical features from `data`, a well as the number of numerical columns in `data`.

        Parameters:
            data : pd.DataFrame
                Tabular dataset used for metadata extraction (originally used for the model training).
        """
        self.categories = np.array(get_categories(data[self.preprocessor.cat_cols].to_numpy()))
        d_numerical = len(self.preprocessor.float_cols) + len(self.preprocessor.int_cols)
        self.raw_config.unimodmlp_params.d_numerical = d_numerical
        self.raw_config.unimodmlp_params.categories = (self.categories + 1).tolist()

        self.raw_config.model_save_path = TABDIFF_BASE_PATH / "tabdiff" / f"ckpt/{self.exp_name}"
        self.raw_config.result_save_path = TABDIFF_BASE_PATH / "tabdiff" / f"result/{self.exp_name}"

        logger.info("TabDiff: Model save path %s", self.raw_config.model_save_path)
        logger.info("TabDiff: Result path %s", self.raw_config.result_save_path)
        self.raw_config.model_save_path.mkdir(parents=True, exist_ok=True)
        self.raw_config.result_save_path.mkdir(parents=True, exist_ok=True)

    def _generate_processor(self, data: pd.DataFrame) -> None:
        """Generates DataImputationPreprocessor from `data` and self.id_column.

        Parameters:
            data : pd.DataFrame
                Tabular dataset used for creating self.preprocessor (originally used for the model training).
        """
        self.preprocessor = DataImputationPreprocessor(data, self.id_column)
        self.data_cols = self.preprocessor.num_cols + self.preprocessor.cat_cols
        self._update_config(data)

    def _generate_torch_dataset(self) -> tuple[TorchDataset, DataLoader]:
        """Generates TorchDataset and DataLoader for TabDiff from self.preprocessor.

        Returns:
            tuple[TorchDataset, DataLoader]: The dataset and dataloader for training TabDiff::UnifiedCtimeDiffusion.
        """
        preprocessed_data = self.preprocessor.fit_transform()
        dataset = TorchDataset(
            preprocessed_data[self.data_cols], d_numerical=self.raw_config.unimodmlp_params.d_numerical, categories=self.raw_config.unimodmlp_params.categories
        )
        train_loader = DataLoader(
            dataset,
            batch_size=self.raw_config.train.main.batch_size,
            shuffle=True,
            num_workers=self.raw_config.train.main.num_workers,
        )
        return dataset, train_loader

    def _create_model(self, train_loader: DataLoader, train_dataset: TorchDataset) -> Trainer:
        """Creates the model for training (TabDiff::Trainer).

        First creates a UniModMLP then the TabDiff diffusion model - UnifiedCtimeDiffusion,
        finally, wraps everything in the Trainer wrapper class from TabDiff.

        Parameters:
            train_loader: DataLoader
                The torch DataLoader for creating TabDiff::UnifiedCtimeDiffusion.
            train_dataset: TorchDataset
                The dataset TorchDataset for creating TabDiff::UnifiedCtimeDiffusion.

        Returns:
            Trainer
                The generated Trainer from TabDiff.
        """
        backbone = UniModMLP(**asdict(self.raw_config.unimodmlp_params))
        model = Model(backbone, **asdict(self.raw_config.diffusion_params.edm_params))
        model.to(self.device)

        diffusion = UnifiedCtimeDiffusion(
            num_classes=self.categories,
            num_numerical_features=self.raw_config.unimodmlp_params.d_numerical,
            denoise_fn=model,
            y_only_model=None,
            **asdict(self.raw_config.diffusion_params),
            device=self.device,
        )
        diffusion.to(self.device)
        diffusion.train()

        sample_batch_size = self.raw_config.sample.batch_size
        return Trainer(
            diffusion=diffusion,
            train_iter=train_loader,
            dataset=train_dataset,
            test_dataset=None,
            metrics=None,
            logger=None,
            **asdict(self.raw_config.train.main),
            sample_batch_size=sample_batch_size,
            num_samples_to_generate=0,
            model_save_path=self.raw_config.model_save_path,
            result_save_path=self.raw_config.result_save_path,
            device=self.device,
            ckpt_path=None,
            y_only=False,
            id_col=self.id_column,
        )

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the TabDiff::UnifiedCtimeDiffusion model to training data.

        Parameters:
            train_data : pd.DataFrame
                Tabular dataset to train the TabDiff::UnifiedCtimeDiffusion model.
        """
        self._generate_processor(train_data)
        dataset, data_loader = self._generate_torch_dataset()
        self.model = self._create_model(train_dataset=dataset, train_loader=data_loader)
        self.model.run_loop()

    def generate(self, total_samples: int, conditions: list | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Generate synthetic samples from the fitted TabDiff::UnifiedCtimeDiffusion model.

        Parameters:
            total_samples : int
                Number of synthetic rows to generate.
            conditions : list | None, optional
                Currently ignored; included for compatibility with the Synthesizer protocol.

        Returns:
            pd.DataFrame
                Synthetic samples as a DataFrame with preprocessing reversed.
        """
        samples = self.model.sample(total_samples)
        data = pd.DataFrame(samples, columns=self.data_cols)
        return self.preprocessor.inverse_transform(data, multiply_categories=False)  # tabdiff does this on its own
