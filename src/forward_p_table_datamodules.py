import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from src.periodic_table import PeriodicTableSurrogateModel


class PeriodicTableDataset(Dataset):
    def __init__(
        self,
        phase: str,
        processed_data_dir: str,
        dataset_name: str,
        test_mode: bool = False,
        test_batch_size: int = 32,
        rng=np.random.default_rng(1234),
    ):
        super().__init__()
        if phase == "SuperConTest":
            dataset = np.load(os.path.join(processed_data_dir, dataset_name))
            self.x = dataset["x_test"].astype(np.float32)
            self.y = dataset["y_test"].astype(np.float32)
            self.years = dataset["years_test"].astype(np.int32)
            self.x = self.x[..., np.newaxis, np.newaxis, np.newaxis]
            self.ids = dataset["ids_test"].astype(np.int32)
            # ID によるスクリーニング
            self.x = self.x[self.ids > 0]
            self.y = self.y[self.ids > 0]
            self.years = self.years[self.ids > 0]
            self.ids = self.ids[self.ids > 0]

        else:
            dataset = np.load(os.path.join(processed_data_dir, dataset_name))
            self.x = dataset[f"x_{phase}"].astype(np.float32)
            self.y = dataset[f"y_{phase}"].astype(np.float32)
            self.years = dataset[f"years_{phase}"].astype(np.int32)
            self.x = self.x[..., np.newaxis, np.newaxis, np.newaxis]
            self.ids = dataset[f"ids_{phase}"].astype(np.int32)

        if phase == "train":
            # shuffle
            print("Shuffling the training data")
            shfl = rng.choice(len(self.x), len(self.x), replace=False)
            self.x = self.x[shfl]
            self.y = self.y[shfl]
            self.year = self.years[shfl]
            self.ids = self.ids[shfl]

        if test_mode:
            self.x = self.x[: test_batch_size * 2]
            self.y = self.y[: test_batch_size * 2]
            self.year = self.years[: test_batch_size * 2]
            self.ids = self.ids[: test_batch_size * 2]

    def __getitem__(self, idx):
        x = self.x[idx]
        x = torch.from_numpy(x).float()

        y = np.array(self.y[idx])
        y = torch.from_numpy(y).float()

        # year is not used in the model, but we need it for the evaluation
        year = np.array(self.years[idx])
        year = torch.from_numpy(year).int()

        # ID is not used in the model, but we need it for the evaluation
        ids = np.array(self.ids[idx])
        ids = torch.from_numpy(ids).int()

        # return (x, year), y
        return (x, ids), y

    def __len__(self):
        return len(self.x)


class PeriodicTableDataModule(pl.LightningDataModule):
    """
    Data Module for PeriodicTableDataset, train, validation, test splits are supported.

    Args:
        processed_data_dir (str): Path to the processed data directory.
        dataset_name (str): Name of the dataset file.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 128.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.

    Attributes:
        train_dataset (torch.utils.data.Dataset): Training set.
        val_dataset (torch.utils.data.Dataset): Validation set.
        test_dataset (torch.utils.data.Dataset): Test set.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_workers: int = 4,
        test_mode: bool = False,
    ):
        super().__init__()

        self.processed_data_dir = cfg.general.processed_data_dir
        self.dataset_name = cfg.dataset.dataset_name
        self.batch_size = cfg.sg_model.batch_size
        self.atom_map_type = cfg.sg_model.atom_map_type
        self.num_workers = num_workers

        # self.train_dataset = Optional[PeriodicTableDataset] = None
        # self.val_dataset = Optional[PeriodicTableDataset] = None
        # self.test_dataset = Optional[PeriodicTableDataset] = None

        self.test_mode = test_mode

    def setup(self, stage: Optional[str] = None):
        """Loads datasets and prepare them for use in dataloaders.

        Args:
            stage (Optional[str], optional): The stage ('fit' or 'test').
            Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PeriodicTableDataset(
                "train",
                self.processed_data_dir,
                self.dataset_name,
                self.test_mode,
                self.batch_size,
            )
            self.val_dataset = PeriodicTableDataset(
                "valid",
                self.processed_data_dir,
                self.dataset_name,
                self.test_mode,
                self.batch_size,
            )

            self.test_dataset = PeriodicTableDataset(
                "test",
                self.processed_data_dir,
                self.dataset_name,
                self.test_mode,
                self.batch_size,
            )

    def train_dataloader(self):
        """Creates a DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Creates a DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Creates a DataLoader for the test set."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Creates a DataLoader for the prediction set."""
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return [val_loader, test_loader]


def define_pretrain_dataset_name(pretrain_file_name):
    """Return the name of the processed data file given the divide method, divide infos and train balancing"""
    return pretrain_file_name + ".npz"


class PeriodicTablePretrainDataset(Dataset):
    def __init__(
        self,
        phase: str,
        processed_data_dir: str,
        dataset_name: str,
        task_name: str,
        test_mode: bool = False,
        test_batch_size: int = 32,
        rng=np.random.default_rng(1234),
    ):
        super().__init__()
        dataset = np.load(
            os.path.join(processed_data_dir, define_pretrain_dataset_name(dataset_name))
        )
        assert phase in ["train", "valid", "test"]
        self.x = dataset[f"x_{phase}"].astype(np.float32)
        self.x = self.x[..., np.newaxis, np.newaxis, np.newaxis]

        self.define_target(dataset, dataset_name, task_name, phase)
        self.define_mask_and_clean(dataset_name, phase, dataset)

        if phase == "train":
            # shuffle
            print("Shuffling the training data")
            shfl = rng.choice(len(self.x), len(self.x), replace=False)
            self.x = self.x[shfl]
            self.y = self.y[shfl]
            self.mask = self.mask[shfl]

        if test_mode:
            self.x = self.x[: test_batch_size * 2]
            self.y = self.y[: test_batch_size * 2]
            self.mask = self.mask[: test_batch_size * 2]

    def define_mask_and_clean(self, dataset_name, phase, dataset):
        if (
            dataset_name == "JARVIS_multitask_all"
            or dataset_name.startswith("JARVIS_OQMD-cate")
            or dataset_name.startswith("JARVIS_MegNet-cate")
        ):
            self.mask = dataset[f"loss_mask_{phase}"].astype(np.float32)
        else:
            self.mask = (~np.isnan(self.y)).astype(np.float32)
            data_bool = (np.sum(self.mask, axis=1) > 0).reshape(-1)

            self.x = self.x[data_bool]
            self.y = self.y[data_bool]
            self.y = np.nan_to_num(self.y)
            self.mask = self.mask[data_bool]

    def define_target(self, dataset, dataset_name: str, task_name: str, phase: str):
        if dataset_name == "JARVIS_multitask_all":
            assert task_name == "multitask"
            ys = [
                dataset[f"{task_name}_{phase}"].astype(np.float32)
                for task_name in ["target_values", "sgr_cate"]
            ]

            self.y = np.concatenate(ys, axis=1)
        elif dataset_name == "JARVIS_multitask":
            assert task_name == "multitask"
            ys = [
                dataset[f"{task_name}_{phase}"].astype(np.float32)
                for task_name in [
                    "bandgap_m",
                    "e_form",
                    "bandgap_oq",
                    "delta_e",
                    "stability",
                ]
            ]
            self.y = np.stack(ys, axis=1)

        elif (
            dataset_name == "JARVIS_OQMD_BG-Only"
            or dataset_name == "JARVIS_OQMD_BG-Only-balanced"
        ):
            self.y = dataset[f"y_{phase}"].astype(np.float32)[..., np.newaxis]

        elif dataset_name == "JARVIS_OQMD_bandgap":
            if task_name in ["bandgap_oq", "delta_e", "stability"]:
                self.y = dataset[f"{task_name}_{phase}"].astype(np.float32)[
                    ..., np.newaxis
                ]
            elif task_name == "multitask":
                ys = [
                    dataset[f"{task_name}_{phase}"].astype(np.float32)
                    for task_name in ["bandgap_oq", "delta_e", "stability"]
                ]
                self.y = np.stack(ys, axis=1)
            else:
                raise ValueError(f"Invalid task name: {task_name}")
        elif dataset_name == "JARVIS_megnet_bandgap":
            if task_name in ["bandgap_m", "e_form"]:
                self.y = dataset[f"{task_name}_{phase}"].astype(np.float32)[
                    ..., np.newaxis
                ]
            elif task_name == "multitask":
                ys = [
                    dataset[f"{task_name}_{phase}"].astype(np.float32)
                    for task_name in ["bandgap_m", "e_form"]
                ]
                self.y = np.stack(ys, axis=1)
            else:
                raise ValueError(f"Invalid task name: {task_name}")

        elif dataset_name.startswith("JARVIS_OQMD-cate") or dataset_name.startswith(
            "JARVIS_MegNet-cate"
        ):
            assert task_name == "multitask"
            self.y = dataset[f"y_{phase}"].astype(np.float32)

        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

    def __getitem__(self, idx):
        x = self.x[idx]
        x = torch.from_numpy(x).float()

        y = np.array(self.y[idx])
        y = torch.from_numpy(y).float()

        mask = np.array(self.mask[idx])
        mask = torch.from_numpy(mask).float()

        return x, y, mask

    def __len__(self):
        return len(self.x)


class PeriodicTablePretrainDataModule(pl.LightningDataModule):
    """
    Data Module for PeriodicTableDataset, train, validation, test splits are supported.

    Args:
        processed_data_dir (str): Path to the processed data directory.
        dataset_name (str): Name of the dataset file.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 128.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 4.

    Attributes:
        train_dataset (torch.utils.data.Dataset): Training set.
        val_dataset (torch.utils.data.Dataset): Validation set.
        test_dataset (torch.utils.data.Dataset): Test set.
    """

    def __init__(
        self,
        cfg: DictConfig,
        num_workers: int = 4,
        test_mode: bool = False,
    ):
        super().__init__()

        self.processed_data_dir = cfg.general.processed_data_dir
        self.dataset_name = cfg.sg_model.pretrain.dataset_name
        self.batch_size = cfg.sg_model.pretrain.batch_size
        self.task_name = cfg.sg_model.pretrain.task_name
        self.atom_map_type = cfg.sg_model.atom_map_type
        self.num_workers = num_workers

        self.test_mode = test_mode

    def setup(self, stage: Optional[str] = None):
        """Loads datasets and prepare them for use in dataloaders.

        Args:
            stage (Optional[str], optional): The stage ('fit' or 'test').
            Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PeriodicTablePretrainDataset(
                "train",
                processed_data_dir=self.processed_data_dir,
                dataset_name=self.dataset_name,
                test_mode=self.test_mode,
                test_batch_size=self.batch_size,
                task_name=self.task_name,
            )
            self.val_dataset = PeriodicTablePretrainDataset(
                "valid",
                processed_data_dir=self.processed_data_dir,
                dataset_name=self.dataset_name,
                test_mode=self.test_mode,
                test_batch_size=self.batch_size,
                task_name=self.task_name,
            )

            self.test_dataset = PeriodicTablePretrainDataset(
                "test",
                processed_data_dir=self.processed_data_dir,
                dataset_name=self.dataset_name,
                test_mode=self.test_mode,
                test_batch_size=self.batch_size,
                task_name=self.task_name,
            )

    def train_dataloader(self):
        """Creates a DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        """Creates a DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        """Creates a DataLoader for the test set."""
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        """Creates a DataLoader for the prediction set."""
        valid_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return [valid_loader, test_loader]


# from surrogate_models.periodic_table import PeriodicTableSurrogateModel


class LitModel(pl.LightningModule):
    """
    Lightning Module for regression task using either pre-trained Vision Transformer or ResNet18.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.opti_method = cfg.sg_model.optimizer
        self.lr = cfg.sg_model.lr
        if cfg.sg_model.pretrain.model_path is None:
            self.sg_model = PeriodicTableSurrogateModel(cfg)
        else:
            print("load the pre-trained model: ", cfg.sg_model.pretrain.model_path)
            self.sg_model = self.load_pretrained_model(cfg)

    def forward(self, x):
        return self.sg_model(x)

    def load_pretrained_model(self, cfg: DictConfig):
        # load the mode
        sg_model = PeriodicTableSurrogateModel(cfg)

        if cfg.sg_model.pretrain.task_name == "multitask":
            if cfg.sg_model.pretrain.dataset_name == "JARVIS_multitask":
                num_output = 5
            elif cfg.sg_model.pretrain.dataset_name == "JARVIS_OQMD_bandgap":
                num_output = 3
            elif cfg.sg_model.pretrain.dataset_name == "JARVIS_megnet_bandgap":
                num_output = 2
            elif cfg.sg_model.pretrain.dataset_name == "JARVIS_multitask_all":
                num_output = 239
            else:
                raise ValueError(
                    f"Invalid dataset name: {cfg.sg_model.pretrain.dataset_name}"
                )
        else:
            num_output = 1

        if cfg.sg_model.model_type.startswith("ViT4PT-"):
            sg_model.base_model.fin_fc = torch.nn.Linear(
                sg_model.base_model.fin_fc.in_features, num_output
            )
            sg_model.base_model.fin_act = torch.nn.Identity()
        elif cfg.sg_model.model_type.startswith("T4PT-"):
            sg_model.base_model.fin_fc = torch.nn.Linear(
                sg_model.base_model.fin_fc.in_features, num_output
            )
            sg_model.base_model.fin_act = torch.nn.Identity()

        else:
            sg_model.base_model.base_model.fc = torch.nn.Linear(
                sg_model.base_model.base_model.fc.in_features, num_output
            )
            sg_model.base_model.fin_act = torch.nn.Identity()

        # load the parameters
        sg_model.load_state_dict(torch.load(cfg.sg_model.pretrain.model_path))

        # change the output layer
        if cfg.sg_model.model_type.startswith("ViT4PT-"):
            sg_model.base_model.fin_fc = torch.nn.Linear(
                sg_model.base_model.fin_fc.in_features, 1
            )
            sg_model.base_model.fin_act = torch.nn.ReLU()
        elif cfg.sg_model.model_type.startswith("T4PT-"):
            sg_model.base_model.fin_fc = torch.nn.Linear(
                sg_model.base_model.fin_fc.in_features, 1
            )
            sg_model.base_model.fin_act = torch.nn.ReLU()
        else:
            sg_model.base_model.base_model.fc = torch.nn.Linear(
                sg_model.base_model.base_model.fc.in_features, 1
            )
            sg_model.base_model.fin_act = torch.nn.ReLU()

        return sg_model

    def training_step(self, batch, batch_idx):
        (x, _), y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.view(-1), y)  # MAE loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        (x, _), y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.view(-1), y)  # MAE loss
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        (x, _), y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat.view(-1), y)  # MAE loss
        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx):
        (x, year), y = batch
        y_hat = self(x)
        return x, y, y_hat, year

    def configure_optimizers(self):
        if self.opti_method == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            self.use_sheduler = False
            return optimizer

        elif self.opti_method == "Adam-CA":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs * 2)
            self.use_sheduler = True
            return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        if self.use_sheduler:
            sch = self.lr_schedulers()
            sch.step()
