import os 
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
import lightning as pl

import my
from my import char2int, LETTERS

N_CPU = os.cpu_count()

word2list = lambda x: [char2int[char] for char in x]
token_seqs = lambda words: [torch.tensor(word2list(w),dtype=torch.int64) for w in words]

class WordDataset(Dataset):
    def __init__(self, df:pd.DataFrame, p_augment:float = -1., seed=34):

        self.df = df.reset_index(drop=True).copy()
        self.word_seqs = token_seqs(df['word'])
        self.lemma_seqs = token_seqs(df['lemma'])

        self.p_augment = p_augment
        self.rng = np.random.default_rng(seed)

        if self.p_augment > 0:
            self.prepare_aug()

        self.labels = None
        if 'stress' in df.columns:
            self.labels = torch.tensor(df['stress'].to_numpy()-1, dtype=torch.int64)

    def __len__(self):
        return len(self.word_seqs)

    def prepare_aug(self):
        self.df[['stem','ending']]=self.df.apply(my.stem,axis=1)
        self.df['ids_cons_in_stem'] = self.df['stem'].apply(my.ids_cons)


    def get_augmented(self, idx:int):
        row = self.df.iloc[idx]
        aug_word, aug_lemma = my.augment(row, self.rng, p=0.3)

        return {'X':torch.tensor(word2list(aug_word),dtype=torch.int64),'lemma':torch.tensor(word2list(aug_lemma),dtype=torch.int64)}

    def __getitem__(self, idx:int):

        if self.rng.random() <= self.p_augment:
            item = self.get_augmented(idx)
        else:
            item = {'X':self.word_seqs[idx],'lemma':self.lemma_seqs[idx]}

        if self.labels is not None: 
                item['y'] = self.labels[idx]

        return item


def collate_for_pack(batch):
    word_list, lemma_list, label_list, len_x, len_lem = [], [], [], [], []

    for item in batch:
        word_list.append(item['X'])
        lemma_list.append(item['lemma'])

        label_list.append(item.get('y',0))

        len_x.append(len(item['X']))
        len_lem.append(len(item['lemma']))

    padded_word_list = nn.utils.rnn.pad_sequence(
    word_list, batch_first=True)
    padded_lemma_list = nn.utils.rnn.pad_sequence(
    lemma_list, batch_first=True)

    label_list = torch.tensor(label_list)
    len_x = torch.tensor(len_x)
    len_lem = torch.tensor(len_lem)
    
    return {'X':padded_word_list, 'lemma':padded_lemma_list, 'y':label_list, 'len_x':len_x, 'len_lem':len_lem}


def collate_for_pad(batch):
    word_list, lemma_list, label_list, lengths = [], [],[], []

    max_seq_len = 0
    for item in batch:
        word_list.append(item['X'])
        lemma_list.append(item['lemma'])
        max_seq_len = max(max_seq_len,len(item['X']),len(item['lemma']))

        label_list.append(item.get('y',0))

        lengths.append(len(item['X']))

    word_list[0] = nn.ConstantPad1d((0, max_seq_len - len(word_list[0])), 0)(word_list[0])
    lemma_list[0] = nn.ConstantPad1d((0, max_seq_len - len(lemma_list[0])), 0)(lemma_list[0])

    padded_word_list = nn.utils.rnn.pad_sequence(
    word_list, batch_first=True)
    padded_lemma_list = nn.utils.rnn.pad_sequence(
    lemma_list, batch_first=True)

    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    
    return {'X':padded_word_list, 'lemma':padded_lemma_list, 'y':label_list, 'lens':lengths}

class DataModule(pl.LightningDataModule):
    def __init__(self, Xy, X_test, fold=0, collate_type='pad', batch_size=256, val_bs=1024):
        super().__init__()

        self.save_hyperparameters('batch_size')

        self.fold = fold
        self.val_bs = val_bs
        self.Xy = Xy
        self.X_test = X_test

        if collate_type == 'pad':
            self.collate_fn = collate_for_pad
        elif collate_type == 'pack':
            self.collate_fn = collate_for_pack
        else:
            raise Exception(f'Unknown collate type {collate_type}')

    def setup(self,stage):

        if stage == 'fit':

            Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold]
            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold]

            self.train = WordDataset(Xy_train)
            self.val = WordDataset(Xy_val)

        if stage == "predict":
            self.test = WordDataset(self.X_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True, collate_fn=self.collate_fn,num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.val_bs, collate_fn=self.collate_fn,num_workers=1)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.val_bs, collate_fn=self.collate_fn,num_workers=1)
