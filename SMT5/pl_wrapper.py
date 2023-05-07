import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from SMT5 import SMT5CLModel, get_lr_linear_decay
import evaluate
import numpy as np
import re
from transformers.optimization import Adafactor

class LitSMT5(pl.LightningModule):
    def __init__(
        self, ckpt: str, lr: float, num_keep_steps: int, num_training_steps: int, name: str,
        mapper_ckpt: str = '', mode: str = 'train',
    ):
        super(LitSMT5, self).__init__()
        self.model = SMT5CLModel(ckpt, mapper_ckpt, mode)
        self.lr = lr
        self.num_keep_steps = num_keep_steps
        self.num_training_steps = num_training_steps
        self.name = name
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def training_step(self, batch, batch_idx):
        l = self.model(batch)

        loss = l.sum()
        self.log(f"{self.name}_train/loss", loss, sync_dist=True)
        return loss
        
    def configure_optimizers(self):
        optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=self.lr)
        lr_scheduler = get_lr_linear_decay(optimizer, num_keep_steps, num_training_steps)
        return [optimizer], [lr_scheduler]
