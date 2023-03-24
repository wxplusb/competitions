import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Union, Any, Optional,Set, Tuple,List, Dict
from .models import MeanPooling
from .losses import BigLoss
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
import torch.optim as optim
from .net import get_scheduler, get_optim_params

class ImageTower(nn.Module):
    def __init__(self, emb_dim:int=512, freeze_first_layers:Optional[int]=3) -> None:
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT 
        self.base = torchvision.models.efficientnet_b0(weights=weights)

        if freeze_first_layers is not None:
            for name, param in self.base.features[:freeze_first_layers].named_parameters():
                param.requires_grad = False

        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True), 
            nn.Linear(in_features=1280, out_features=emb_dim))

    def forward(self, batch:Dict[str,Tensor]):
        x = self.base(batch['image'])
        return x
        # return F.normalize(x, p=2, dim=1)


class TextTower(nn.Module):
    def __init__(self, cfg=None) -> None:
        super().__init__()

        self.cfg = cfg

        self.base = AutoModel.from_pretrained(self.cfg.model_name)
        # self.config = AutoConfig.from_pretrained(self.cfg.model_name)

        if self.cfg.gradient_checkpointing:
            self.base.gradient_checkpointing_enable()

        self.pool = MeanPooling()
        self.drop = nn.Dropout(0.3)
        self.clf = nn.Linear(self.base.config.hidden_size, cfg.emb_dim)

    def features(self, inputs):
        inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask","token_type_ids"]}
        outputs = self.base(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feats = self.features(inputs)
        x = self.clf(self.drop(feats))
        return x
        # return F.normalize(x, p=2, dim=1)

class HUSE(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.ImageTower = ImageTower(cfg.emb_dim)
        self.TextTower = TextTower(cfg)

        # for shop_id (shop_emb_dim = 10)
        self.shop_emb = nn.Embedding(num_embeddings=11126+1, embedding_dim=cfg.shop_emb_dim, padding_idx=0)

        self.clf = nn.Sequential( 
            torch.nn.Linear(cfg.emb_dim * 2 + cfg.shop_emb_dim, out_features = cfg.num_classes),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2), 
            torch.nn.Linear(in_features=cfg.num_classes, out_features=cfg.num_classes))
        

    def forward(self, batch):        
        img_emb = self.ImageTower(batch)
        text_emb = self.TextTower(batch)
        cat_emb = self.shop_emb(batch['shop_id'])

        all_emb = torch.cat([img_emb,text_emb,cat_emb], dim=1)

        out = self.clf(all_emb)

        img_emb = F.normalize(img_emb, p=2, dim=1)
        text_emb = F.normalize(text_emb, p=2, dim=1)
        all_emb = F.normalize(all_emb, p=2, dim=1)
        return out, all_emb, img_emb, text_emb

class HUSEModule(pl.LightningModule):
    def __init__(self, Aij, cfg):
        super().__init__()

        self.save_hyperparameters('cfg')
        self.cfg = cfg
        self.model = HUSE(self.cfg)
        self.loss_fn = BigLoss(Aij, cfg=cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch:Dict[str,Tensor], batch_idx):
        y = batch['label']
        y_hat, all_emb, img_emb, text_emb  = self(batch)
        loss, _ = self.loss_fn(y_hat, all_emb, img_emb, text_emb, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        
        return {'loss': loss}

    def validation_step(self, batch:Dict[str,Tensor], batch_idx):
        y = batch['label']
        y_hat, all_emb, img_emb, text_emb  = self(batch)
        loss, ls_ = self.loss_fn(y_hat, all_emb, img_emb, text_emb, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        if batch_idx %50 == 0:
            print(ls_)

        return (y.cpu().numpy(), torch.argmax(y_hat, dim=1).detach().cpu().numpy())
        
    def validation_epoch_end(self, outputs):
        ys = np.concatenate([o[0] for o in outputs])
        y_hats = np.concatenate([o[1] for o in outputs])

        val_score = f1_score(ys, y_hats, average='weighted')

        self.log('val_f1', val_score, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(get_opt_params_lr(self.model, bert_lr = self.cfg.bert_lr), lr=self.cfg.lr)
        scheduler = get_scheduler(optimizer, self.cfg)

# https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers
        return {"optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": 'step',
            "frequency": 1
        },}


def get_opt_params_lr(model, bert_lr:float=5e-5):
    
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    other_name_params = []
    bert_name_params = []
    for n,p in model.named_parameters():
        if 'TextTower' in n:
        # if 'TextTower.base' in n:
            bert_name_params.append((n,p))
        else:
            other_name_params.append((n,p))

    bert_params = [
        {"params": [
        p for n, p in bert_name_params if
        not any(nd in n for nd in no_decay)], "weight_decay": 0.001, 'lr': bert_lr},

        {"params": [
        p for n, p in bert_name_params if
        any(nd in n for nd in no_decay)], "weight_decay": 0.0, 'lr': bert_lr},
        ]

    other_params = [
        {"params": [
        p for n, p in other_name_params if
        not any(nd in n for nd in no_decay)], "weight_decay": 0.001},

        {"params": [
        p for n, p in other_name_params if
        any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
    print(len(bert_params[0]["params"]),len(bert_params[1]["params"]),len(other_params[0]["params"]),len(other_params[1]["params"]))

    return bert_params + other_params