#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : Zhang Xin
@Contact: xinzhang_hp@163.com
@Time : 2022/12/25
"""
import os
import pathlib
from typing import Optional, Dict, Any

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.datamodules.components.mind import MINDDataset


class MINDdatamodule(LightningDataModule):
    def __init__(self,
                 data_dir: str = "data/MIND_small",
                 batch_size: int = 64,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 data_train: Optional[Dataset] = None,
                 data_val: Optional[Dataset] = None,
                 data_test: Optional[Dataset] = None,
                 ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train = self.hparams.data_train
        self.data_val = self.hparams.data_val
        self.data_test = self.hparams.data_test

    def prepare_data(self):
        """Process data
        """
        path = pathlib.Path(os.path.join(self.hparams.data_dir, 'MINDsmall_train', 'behaviors_parsrd.tsv'))
        if path.exists():
            print('processed data!')
        else:
            print('please process data!')

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """
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
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    print(root)
    import os
    print(os.getcwd())
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mind.yaml")
    cfg.data_dir = str(root / "data")
    print(omegaconf.OmegaConf.to_yaml(cfg))
    _ = hydra.utils.instantiate(cfg)
