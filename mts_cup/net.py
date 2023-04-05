import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os
import pandas as pd

import pytorch_lightning as pl
import torch.optim as optim

from torchmetrics.classification import MulticlassF1Score
from ptls.data_load.padded_batch import PaddedBatch
from ptls.frames.supervised import SequenceToTarget
from ptls.frames.coles import ColesSupervisedModule

import logging
logger = logging.getLogger(__name__)



def set_seed(seed:int=34):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    pl.utilities.seed.seed_everything(seed, workers=True)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

class UserEncoder(nn.Module):
    def __init__(self, seq_enc, cat_embs, float_const=None) -> None:
        super().__init__()
        
        self.seq_enc = seq_enc

        self.cat_embs = nn.ModuleDict({name:nn.Embedding(num_embeddings=v['in'],embedding_dim=v['emb_dim'], padding_idx=0) for name,v in cat_embs.items() if v['emb_dim'] > 0})

        self.output_size = self.seq_enc.embedding_size + sum(v['emb_dim'] for v in cat_embs.values())

    def forward(self, x):
        const_feats = []
        if len(self.cat_embs) > 0:
            const_feats = [e(x.payload[name]) for name, e in self.cat_embs.items()]

        seq_feats = self.seq_enc(x)

        x = torch.cat(const_feats + [seq_feats], dim=1)
        return x



class mySequenceToTarget(SequenceToTarget):
    def predict_step(self, batch, batch_idx):
        return self(batch[0]).detach().cpu().numpy()

class myColesSupervisedModule(ColesSupervisedModule):
    def predict_step(self, batch, batch_idx):
        return self(batch[0]).cpu().numpy(), batch[1].squeeze().cpu().numpy()
