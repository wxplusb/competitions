import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
import torch
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar, ModelCheckpoint
import lightning as pl
import net, data

import my

dir_out = "out/"
dir_data = 'data/'

os.makedirs(dir_out, exist_ok = True)

RANDOM_STATE = 34
np.random.seed(RANDOM_STATE)
N_CPU = os.cpu_count()

def main(args):

    N_FOLDS = args.n_folds
    RANDOM_STATE = args.seed
    EXP = args.experiment
    FOLD = args.fold
    CHECKPOINT_DIR = args.ckpt_dir
    os.makedirs(CHECKPOINT_DIR, exist_ok = True)
    NAME_CKPT = f'exp{EXP}_f{FOLD}'

    print('[train fold]: ', FOLD)

    net.set_seed(RANDOM_STATE)

    df = pd.read_csv(dir_data+'train.csv')
    df_test = pd.read_csv(dir_data+'test.csv')

    # удаляем точно известные случаи ударений
    known_cases = df['word'].str.contains('ё') | (df['num_syllables'] == 1)
    df = df[~known_cases].reset_index(drop=True).copy()

    df = my.add_folds(df,n_folds=N_FOLDS, random_state=RANDOM_STATE)

    # тренируем модель
    es = EarlyStopping('val_acc',patience=10,verbose=True,mode='max')
    tq = TQDMProgressBar(refresh_rate=10)
    chpt = ModelCheckpoint(dirpath=CHECKPOINT_DIR,filename=NAME_CKPT,  monitor="val_acc",mode='max')

    dm = data.DataModule(df,df_test,fold=FOLD,collate_type='pack', batch_size=64)
    model = net.Model()
    trainer = pl.Trainer(callbacks=[tq,es,chpt],max_epochs=1000,deterministic = True)
    trainer.fit(model, datamodule=dm)

    logit_preds = trainer.predict(datamodule=dm, ckpt_path=chpt.best_model_path)
    logit_preds = torch.cat(logit_preds).numpy()

    np.save(dir_out+f'logits_exp{EXP}_f{FOLD}.npy',logit_preds)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--experiment", type=int, default=1)
    parser.add_argument("--ckpt_dir", type=str, default='ckpts/')
    parser.add_argument("--make_sub", type=str, default='')
    args = parser.parse_args()

    if args.make_sub:
        my.make_sub(dir_data + 'test.csv', dir_out, args.make_sub)
    else:
        main(args)