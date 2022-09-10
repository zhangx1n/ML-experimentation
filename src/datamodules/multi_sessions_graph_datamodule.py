#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : Zhang Xin
@Contact: xinzhang_hp@163.com
@Time : 2022/8/14
"""
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader, Dataset


class SessionsGraphDataModule(LightningDataModule):
    def __init__(
            self,
            # data_dir: str = "data/",
            # train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_dataset: Optional[Dataset] = None,
            val_dataset: Optional[Dataset] = None,
            test_dataset: Optional[Dataset] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = train_dataset
        self.data_val = val_dataset
        self.data_test = test_dataset

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

                This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
                careful not to execute things like random split twice!
                """
        # load and split datasets only if not loaded already
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    # import hydra
    # import omegaconf
    # import pyrootutils
    #
    # root = pyrootutils.setup_root(__file__, pythonpath=True)
    # cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mnist.yaml")
    # cfg.data_dir = str(root / "data")
    # _ = hydra.utils.instantiate(cfg)
    pass
