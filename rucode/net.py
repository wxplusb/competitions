import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig
# from sklearn.metrics import f1_score
from typing import Union, Any, Optional,Set, Tuple,List, Dict
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torchmetrics.classification import BinaryF1Score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed:int=34):
    pl.seed_everything(seed, workers=True)
    # устанавливается в trainer
    # torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


class BertDset(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer, max_length:int=400):
        
        self.inputs = tokenizer(df['text'].to_list(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

        if 'label' in df.columns:
            self.labels = torch.tensor(df['label'].to_numpy(), dtype=torch.float32)
        else:
            self.labels = torch.zeros(len(df), dtype=torch.float32)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "label": self.labels[idx],
            }
        if "token_type_ids" in self.inputs:
            item["token_type_ids"] = self.inputs["token_type_ids"][idx]
        return item

class BertDataModule(pl.LightningDataModule):
    def __init__(self, Xy:pd.DataFrame, X_test:pd.DataFrame, fold:int=0, batch_size:int=16, val_bs:int=16, n_cpu:int=8, cfg=None):
        super().__init__()

        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#hyperparameters-in-datamodules
        # если без аргументов то датасеты сохранит
        self.save_hyperparameters('batch_size','cfg')

        self.fold = fold
        self.val_bs = val_bs
        self.Xy = Xy
        self.X_test = X_test
        self.n_cpu = n_cpu
        self.max_length = cfg.max_length
        self.tokenizer = cfg.model_name

        self.len_train = (self.Xy['fold']!=self.fold).sum()
        
    # def prepare_data(self):
    # только скачиваем датасеты, эта ф-я только один раз исполняется в одном процессе
    #     # called only on 1 GPU
    #     download_dataset()
    #     tokenize()
    #     build_vocab()

    def setup(self,stage):
        if stage == 'fit':

            Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold]
            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold]


            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

            self.train = BertDset(Xy_train,self.tokenizer, max_length=self.max_length)
            self.val = BertDset(Xy_val,self.tokenizer, max_length=self.max_length)

            self.train_val = BertDset(self.Xy,self.tokenizer, max_length=self.max_length)
            self.test = BertDset(self.X_test,self.tokenizer, max_length=self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True,num_workers=self.n_cpu)

    def get_val_dl(self, ds:Dataset) -> DataLoader:
        return DataLoader(ds, batch_size=self.val_bs, num_workers=self.n_cpu)

    def predict_dataloader(self):
        return self.get_val_dl(self.test)

    def val_dataloader(self):
        return self.get_val_dl(self.val)

class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class BertModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.model = AutoModel.from_pretrained(self.cfg.model_name)
        # self.config = AutoConfig.from_pretrained(self.cfg.model_name)

        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.pool = MeanPooling()
        self.drop = nn.Dropout(0.3)
        self.clf = nn.Linear(self.model.config.hidden_size, 1)

    def features(self, inputs):
        inputs = {k:v for k,v in inputs.items() if k != 'label'}
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feats = self.features(inputs)
        output = self.clf(self.drop(feats))
        return output

class BertModule(pl.LightningModule):

    def __init__(self, cfg, lr = 5e-5):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.model = BertModel(self.cfg)
        self.loss_fn = nn.BCEWithLogitsLoss() # more stable
        self.val_metric = BinaryF1Score(threshold = cfg.threshold)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch:Dict[str,Tensor], batch_idx):
        y = batch['label'].reshape(-1,1)
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        return {'loss': loss}

    def validation_step(self, batch:Dict[str,Tensor], batch_idx):
        y = batch['label'].reshape(-1,1) # shape (16)
        y_hat = self(batch) # shape (16, 1)

        # print('y_shapes: ', y.shape, y_hat.shape)
        loss = self.loss_fn(y_hat, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True)

        self.val_metric(y_hat, y)

        self.log('val_f1', self.val_metric, on_step=False, on_epoch=True, prog_bar=True)

        # return (y.cpu().numpy(), torch.argmax(y_hat, dim=1).detach().cpu().numpy())
        
    # def validation_epoch_end(self, outputs):
    #     ys = np.concatenate([o[0] for o in outputs])
    #     y_hats = np.concatenate([o[1] for o in outputs])

    #     val_score = f1_score(ys, y_hats, average='weighted')

    #     self.log('val_f1', val_score, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        return nn.functional.sigmoid(self(batch))

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