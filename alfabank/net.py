import numpy as np
import torch
from torch import nn, Tensor
import torch.optim as optim
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Union, Any, Optional,Set, Tuple,List, Dict
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from .data import tag_to_index, get_ents, extract_word, index_to_tag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed:int=34):
    pl.seed_everything(seed, workers=True)
    # устанавливается в trainer
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class CustomBertNer(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg
        self.model = AutoModel.from_pretrained(self.cfg.model_name)

        self.dropout = nn.Dropout(self.cfg.hidden_dropout_prob)
        # self.clf = nn.Linear(self.model.config.hidden_size, self.cfg.num_labels)
        self.clf = nn.Sequential( 
            torch.nn.Linear(self.model.config.hidden_size, out_features = 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.2), 
            torch.nn.Linear(64, self.cfg.num_labels))

    def features(self, inputs):
        inputs = {k:v for k,v in inputs.items() if k in ['input_ids','token_type_ids', 'attention_mask']}
        outputs = self.model(**inputs)
        return outputs['last_hidden_state']

    def forward(self, inputs):
        feats = self.features(inputs)
        logits = self.clf(self.dropout(feats))
        return logits


class NerModule(pl.LightningModule):
    def __init__(self, cfg, lr = 5e-5):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.model = CustomBertNer(self.cfg)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tag_to_index['PAD'])

        self.f1_good_train = F1Score()
        self.f1_brand_train = F1Score()
        self.f1_good_val = F1Score()
        self.f1_brand_val = F1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch:Dict[str,Tensor], batch_idx):

        logits = self.model(batch)
        loss = self.loss_fn(logits.view(-1, self.cfg.num_labels), batch['labels'].view(-1))

        pred_labels = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        g_preds, b_preds = labels_to_words(batch, pred_labels)

        for i in range(len(batch['goods'])):
            self.f1_good_train.update(g_preds[i], batch['goods'][i])
            self.f1_brand_train.update(b_preds[i], batch['brands'][i])

        self.log("train_loss", loss, on_step=True, on_epoch=False, batch_size=len(batch['texts']))
        
        return {'loss': loss}

    def validation_step(self, batch:Dict[str,Tensor], batch_idx):

        logits = self.model(batch)
        loss = self.loss_fn(logits.view(-1, self.cfg.num_labels), batch['labels'].view(-1))

        pred_labels = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        g_preds, b_preds = labels_to_words(batch, pred_labels)

        for i in range(len(batch['goods'])):
            self.f1_good_val.update(g_preds[i], batch['goods'][i])
            self.f1_brand_val.update(b_preds[i], batch['brands'][i])

        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=len(batch['texts']))

    def predict_step(self, batch:Dict[str,Tensor], batch_idx):

        logits = self.model(batch)

        pred_labels = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        g_preds, b_preds = labels_to_words(batch, pred_labels)

        results = []

        for i in range(len(g_preds)):
            results.append((g_preds[i],b_preds[i]))

        return results

    def on_train_epoch_end(self):
        f1_g,p_g,r_g = self.f1_good_train.get()
        f1_b,p_b,r_b = self.f1_brand_train.get()
        m = {"f1_G_train": f1_g, "f1_B_train": f1_b, 'f1_train': (2*f1_b + f1_g)/3}
        self.log_dict(m)
        print(f'TRAIN: f1: {m["f1_train"]:.3f} | f1_G: {m["f1_G_train"]:.3f} | f1_B: {m["f1_B_train"]:.3f} | PR_G: {p_g:.3f} | RE_G: {r_g:.3f} | PR_B: {p_b:.3f} | RE_B: {r_b:.3f}')

        self.f1_good_train.reset()
        self.f1_brand_train.reset()

    def on_validation_epoch_end(self):
        f1_g,p_g,r_g = self.f1_good_val.get()
        f1_b,p_b,r_b = self.f1_brand_val.get()
        m = {"f1_G_val": f1_g, "f1_B_val": f1_b, 'f1_val': (2*f1_b + f1_g)/3}
        self.log_dict(m)

        print(f'> VAL: f1: {m["f1_val"]:.3f} | f1_G: {m["f1_G_val"]:.3f} | f1_B: {m["f1_B_val"]:.3f} | PR_G: {p_g:.3f} | RE_G: {r_g:.3f} | PR_B: {p_b:.3f} | RE_B: {r_b:.3f}')

        self.f1_good_val.reset()
        self.f1_brand_val.reset()  

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


def labels_to_words(bt, pred_labels):

    good_preds = []
    brand_preds = []

    batch_size = len(pred_labels)

    for ex_i in range(batch_size):
        
        pads = bt['paddings'][ex_i]
        pred_tags = ['O'] * len(pads)

        for seq_idx in range(len(pads)):
            if pads[seq_idx] != 1:
                pred_tags[seq_idx] = index_to_tag[pred_labels[ex_i, seq_idx]]
        
        
        entities = get_ents(pred_tags)

        # print(pred_tags,entities)
        # print(entities)

        g_preds = [extract_word(bt['texts'][ex_i], bt['offsets'][ex_i], start, end) for t, start, end in entities if t == "GOOD"]
        good_preds.append(g_preds)

        b_preds = [extract_word(bt['texts'][ex_i], bt['offsets'][ex_i], start, end) for t, start, end in entities if t == "BRAND"]
        brand_preds.append(b_preds)

    return good_preds, brand_preds

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


class F1Score:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred, target):
        if len(pred)  == 0:
            pred = ['']
        if len(target)  == 0:
            target = ['']
        pred = frozenset(x for x in pred)
        target = frozenset(x for x in target)
        self.tp += len(pred & target)
        self.fp += len(pred - target)
        self.fn += len(target - pred)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def get(self):
        if self.tp == 0:
            return 0.0
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        return 2 / (1 / precision + 1 / recall), precision, recall
    

def calc_f1(y_pred, y_true):
    f1 = F1Score()

    y_pred = list(y_pred)
    y_true = list(y_true)

    for i in range(len(y_pred)):
        f1.update(y_pred[i], y_true[i])

    return f1.get()