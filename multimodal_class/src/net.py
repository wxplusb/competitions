import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from .models import BertModel, ImageModel, FTTModel
from typing import Union, Any, Optional,Set, Tuple,List, Dict
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed:int=34):
    pl.seed_everything(seed, workers=True)
    # устанавливается в trainer
    # torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class KazanModule(pl.LightningModule):
    @torch.inference_mode()
    def get_embs(self, dl):
        self.model.eval()
        embs = []
        for batch in tqdm(dl, desc='Getting embeddings', leave=False):
            batch = self.dict_to_device(batch)
            batch_embs = self.model.get_embs(batch).detach().cpu().numpy()
            embs.append(batch_embs)
        self.model.train()
        return np.concatenate(embs)

    def dict_to_device(self, batch:Dict[str,torch.tensor]) -> Dict[str,torch.tensor]:
        return {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in batch.items()}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch:Dict[str,Tensor], batch_idx):
        y = batch['label']
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        return {'loss': loss}

    def validation_step(self, batch:Dict[str,Tensor], batch_idx):
        y = batch['label']
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return (y.cpu().numpy(), torch.argmax(y_hat, dim=1).detach().cpu().numpy())
        
    def validation_epoch_end(self, outputs):
        ys = np.concatenate([o[0] for o in outputs])
        y_hats = np.concatenate([o[1] for o in outputs])

        val_score = f1_score(ys, y_hats, average='weighted')

        self.log('val_f1', val_score, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return nn.functional.softmax(self(batch),dim = 1)

class ImageModule(KazanModule):
    def __init__(self, lr:float = 0.001, freeze_first_layers:Optional[int]= None):
        super().__init__()
        self.save_hyperparameters()

        self.model = ImageModel(freeze_first_layers=freeze_first_layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3,threshold=0.0001, min_lr=0.00005,verbose=True)
        return {"optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_f1",
            "frequency": 1
        },}

class BertModule(KazanModule):
    def __init__(self, cfg, lr = 5e-5):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.model = BertModel(self.cfg)
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(get_optim_params(self.model), lr=self.hparams.lr)
        scheduler = get_scheduler(optimizer, self.cfg)

        return {"optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": 'step',
            "frequency": 1
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },}


class FTTModule(KazanModule):
    def __init__(self, lr:float = 0.001, name='mlp'):
        super().__init__()
        self.save_hyperparameters()

        self.name = name

        self.model = FTTModel(name)
        self.loss_fn = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        if self.name == 'ftt':
            optimizer = self.model.base.make_default_optimizer()
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=0.)
            
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5,threshold=0.0001, min_lr=0.00001,verbose=True)
        return {"optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_f1",
            "frequency": 1
        },}

def get_optim_params(model):
    model_params = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    opt_params = [
        {"params": [
        p for n, p in model_params if
        not any(nd in n for nd in no_decay)], "weight_decay": 0.001},

        {"params": [
        p for n, p in model_params if
        any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    return opt_params

def get_scheduler(optimizer, cfg):
    if cfg.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=cfg.num_train_steps)

    elif cfg.scheduler == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=cfg.num_train_steps, num_cycles=cfg.num_cycles)
    return scheduler