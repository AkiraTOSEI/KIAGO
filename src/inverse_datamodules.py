from typing import Optional

import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


class InverseDataset4PariodicTable(Dataset):
    def __init__(
        self,
        phase: str,
        cfg: DictConfig,
        test_mode: bool = False,
    ):
        super().__init__()
        self.iv_batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
        self.opt_steps = cfg.inverse_problem.method_parameters.optimization_steps

        if test_mode:
            self.opt_steps = 2

        Y = np.array([cfg.inverse_problem.target_tc] * self.iv_batch_size)
        self.Y = Y.astype(np.float32)

    def __len__(self):
        return len(self.Y) * self.opt_steps

    def __getitem__(self, _index):
        index = _index % self.iv_batch_size
        return self.Y[index]


class InverseDataModule4PariodicTable(pl.LightningDataModule):
    """
    Data Module for InverseProblem, train, validation, test splits are supported.

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
        self.batch_size = cfg.inverse_problem.method_parameters.iv_batch_size
        self.atom_map_type = cfg.sg_model.atom_map_type
        self.num_workers = num_workers
        self.cfg = cfg

        # self.train_dataset = Optional[PeriodicTableDataset] = None
        # self.val_dataset = Optional[PeriodicTableDataset] = None
        # self.test_dataset = Optional[PeriodicTableDataset] = None

        self.test_mode = test_mode
        if self.test_mode:
            self.batch_size = 32

    def setup(self, stage: Optional[str] = None):
        """Loads datasets and prepare them for use in dataloaders.

        Args:
            stage (Optional[str], optional): The stage ('fit' or 'test').
            Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = InverseDataset4PariodicTable(
                "train",
                self.cfg,
                self.test_mode,
            )

    def train_dataloader(self):
        """Creates a DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        """Creates a DataLoader for the prediction set."""
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
        return train_loader
