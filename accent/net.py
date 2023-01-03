import torch
from torch import nn
import torch.nn.functional as F

import lightning as pl
from lightning.pytorch.callbacks import LearningRateFinder
import torch.optim as optim

from torchmetrics import Accuracy

import models as ms
# import losses

# https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#reproducibility
def set_seed(random_state):
    pl.lite.utilities.seed.seed_everything(random_state, workers=True)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class Model(pl.LightningModule):
    def __init__(self, lr = 0.001):
        super().__init__()
        self.save_hyperparameters()

        self.model = ms.Gru_Pack_final()
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task='multiclass',top_k=1, average ='micro',num_classes=6)
        self.val_acc = Accuracy(task='multiclass',top_k=1,  average ='micro', num_classes=6)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5,threshold=0.0001, min_lr=0.00005,verbose=True)
        return {"optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_acc",
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },}

    def training_step(self, batch, batch_idx):
        y = batch['y']
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        y = batch['y']
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)

        self.val_acc(y_hat, y)
        self.log('val_acc',self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        return self(batch)

# https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#customizing-learning-rate-finder
class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)