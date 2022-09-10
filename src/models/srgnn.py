#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author : Zhang Xin
@Contact: xinzhang_hp@163.com
@Time : 2022/8/11
"""
from typing import Any, List

import torch
import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.core import datamodule
from torch_geometric.graphgym import cfg
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.retrieval import RetrievalMRR, RetrievalHitRate, RetrievalRecall
from torchmetrics.functional import retrieval_hit_rate


class SRGNN(LightningModule):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            net: torch.nn.Module,
            top_k: int = 10,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.top_k = top_k

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_acc = Accuracy()
        self.test_acc = Accuracy()
        # self.train_mrr = RetrievalMRR()
        # self.test_mrr = RetrievalMRR()
        self.train_hit = RetrievalHitRate(k=self.top_k)
        # self.val_hit = RetrievalHitRate(k=self.top_k)
        # self.train_recall = RetrievalRecall()
        # self.test_recall = RetrievalRecall()

        self.matric_hist = {
            "train/loss": [],
            "train/hit": [],
            "train/mrr": [],
            "test/loss": [],
            "test/hit": [],
            "test/mrr": []
        }

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def step(self, batch: Any):
        scores = self.net(batch)    # batch_size * n_nodes
        targets = batch.y - 1   #batch_size   # https://github.com/CRIPAC-DIG/SR-GNN/issues/6
        loss = self.criterion(scores, targets)
        return loss, scores, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, scores, targets = self.step(batch)
        sub_scores = scores.topk(self.top_k)[1]  # batch * top_k
        # log train metrics
        acc = self.train_acc(scores, targets)
        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
            self.matric_hist["train/hit"].append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                self.matric_hist["train/mrr"].append(0)
            else:
                self.matric_hist["train/mrr"].append(1 / (np.where(score == target)[0][0] + 1))
        hit = np.mean(self.matric_hist["train/hit"]) * 100
        mrr = np.mean(self.matric_hist["train/mrr"]) * 100
        # print("scores: ", scores)
        # print("targets: ", targets)
        print("acc: ", acc)
        print("mrr: ", mrr)
        print("hit: ", hit)
        # self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        # self.log("train/hit_rate", hit, prog_bar=False, on_step=True, on_epoch=False)
        # self.log("train/mrr", mrr, prog_bar=False, on_step=True, on_epoch=False)
        # hit_rate = self.train_hit(preds=scores, target=torch.where(scores == targets.unsqueeze(1), True, False),
        #                           indexes=torch.arange(scores.shape[0]).unsqueeze(1).repeat(1, scores.shape[1]))
        # for score, target in zip(scores, targets):
        #     hit_rate = self.train_hit(preds=score, target=torch.where(score == target.unsqueeze(0), True, False),
        #                               indexes=torch.arange(scores.shape[-1]))
        # hit_rate = retrieval_hit_rate(preds=scores, target=torch.isin(scores, targets), k=self.top_k)


        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "scores": scores, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        self.train_acc.reset()

    # def validation_step(self, batch: Any, batch_idx: int):
    #     loss, preds, targets = self.step(batch)
    #
    #     # log val metrics
    #     acc = self.val_acc(preds, targets)
    #     self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
    #     self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
    #
    #     return {"loss": loss, "preds": preds, "targets": targets}
    #
    # def validation_epoch_end(self, outputs: List[Any]):
    #     acc = self.val_acc.compute()  # get val accuracy from current epoch
    #     self.val_acc_best.update(acc)
    #     self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)
    #     self.val_acc.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, scores, targets = self.step(batch)
        sub_scores = scores.topk(self.top_k)[1]  # batch * top_k
        # log train metrics
        acc = self.train_acc(scores, targets)
        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
            self.matric_hist["test/hit"].append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                self.matric_hist["test/mrr"].append(0)
            else:
                self.matric_hist["test/mrr"].append(1 / (np.where(score == target)[0][0] + 1))
        hit = np.mean(self.matric_hist["test/hit"]) * 100
        mrr = np.mean(self.matric_hist["test/mrr"]) * 100
        # print("scores: ", scores)
        # print("targets: ", targets)
        print("acc: ", acc)
        print("mrr: ", mrr)
        print("hit: ", hit)
        # log train metrics
        # acc = self.train_acc(sub_scores, targets)
        # self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        #self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "scores": scores, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        self.test_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return {
            "optimizer": self.hparams.optimizer(params=self.parameters()),
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)

    from pytorch_lightning import LightningModule, Trainer

    data_cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "diginetica_sessions_graph.yaml")
    datamodule = hydra.utils.instantiate(data_cfg)
    model_cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "srgnn.yaml")
    model = hydra.utils.instantiate(model_cfg)
    trainer = Trainer(max_epochs=5, accelerator="mps", devices=1)
    trainer.fit(model, datamodule)




