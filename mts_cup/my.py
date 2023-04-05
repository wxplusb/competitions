import pickle
import numpy as np
import pandas as pd
import os
import json
import math
import bisect
import torch
from ptls.data_load.utils import collate_feature_dict

try:
    from IPython.display import display
except:
    pass

RANDOM_STATE = 34
N_CPU = os.cpu_count()

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*least populated class.*")
warnings.filterwarnings("ignore", ".*one of them to enable TensorBoard support.*")

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True,nb_workers=5)

def get_train_val_ids(df, fold, target='target_age'):

    train = df.loc[(df[target]!=-1) & (df.fold!=fold),['user_id', 'len_bucket']]
    val = df.loc[(df[target]!=-1) & (df.fold==fold),'user_id'].to_numpy()

    return train, val

def get_val_plts(all_plts, val_ids):
    val_plts = []
    for i in val_ids:
        val_plts.append(all_plts[i])

    val_plts.sort(key=lambda x:len(x['url_host'])) 

    return val_plts



seq_keys = ['region_name','city_name','event_time','dayofweek','diff_time','part_of_day','url_host','request_cnt']

class SeqToTargetDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 target_col_name,
                 target_dtype=None,
                 real_len=None,
                 seed=34,
                 aug=False,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)

        self.data = data
        self.target_col_name = target_col_name
        if type(target_dtype) is str:
            self.target_dtype = getattr(torch, target_dtype)
        else:
            self.target_dtype = target_dtype

        self.real_len = real_len
        self.rng = np.random.default_rng(seed)
        self.aug = aug


    def __len__(self):
        if self.real_len:
            return self.real_len
        return len(self.data)

    def __getitem__(self, ix):
        d = self.data[ix]
        seq_len = len(d['url_host'])

        if (not self.aug) or (seq_len < 30):
            return d
        
        if seq_len < 100:
            new_size = self.rng.integers(int(seq_len*0.9),seq_len,endpoint=True)
        elif seq_len < 300:
            new_size = self.rng.integers(100,seq_len,endpoint=True)
        else:
            new_size = self.rng.integers(300,seq_len,endpoint=True)

        new_ids = np.sort(self.rng.choice(seq_len,size=new_size,replace=False))

        new_d = d.copy()
        for k in seq_keys:
            new_d[k] = new_d[k][new_ids]

        return new_d

    def __iter__(self):
        for feature_arrays in self.data:
            yield feature_arrays


    def collate_fn(self, padded_batch):
        padded_batch = collate_feature_dict(padded_batch)
        target = padded_batch.payload[self.target_col_name]
        del padded_batch.payload[self.target_col_name]
        if self.target_dtype is not None:
            target = target.to(dtype=self.target_dtype)
        return padded_batch, target

TIMEtoINT = {'morning':1,'day':2,'evening':3,'night':4}

def emb_sz_rule(n_cat:int)->int: return min(512, round(1.6 * n_cat**0.56))

def label_encode(df, cols, encoders):
    for col in cols:
        if col in df.columns:
            if len(encoders[col].classes_)>120: col_type = np.int32
            else: col_type = np.int8

            df[col] = encoders[col].transform(df[col]).astype(col_type) + 1   
    return df


def age_bucket(x):
    return max(bisect.bisect_left([18,25,35,45,55,65], x) - 1,0)

def cat_age(x):
    if x == -1:
        return -1 
    elif x <= 25:
        return 0
    elif x <= 35:
        return 1
    elif x <= 45:
        return 2
    elif x <= 55:
        return 3
    elif x <= 65:
        return 4
    return 5 

def add_folds(df:pd.DataFrame, strat_col = 'stratify', n_folds:int=5, random_state:int=34) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    df['fold'] = -1
    for fold, (trn_, val_) in enumerate(skf.split(df,df[strat_col])):
        df.loc[val_,'fold'] = fold

    return df

def binze(se:pd.Series, ql = (0.05,0.95), num=2, pre='f') -> pd.Series:
    a1 = se.quantile(ql[0])
    a2 = se.quantile(ql[1])

    edges = np.linspace(a1,a2, num+1)
    assert len(edges) >= 3, 'len(edges) < 3'
    edges[0] = float('-inf')
    edges[-1] = float('inf')

    time_lb = [f'{pre}{i}' for i in range(num)]

    se, bins = pd.cut(se, edges, labels=time_lb, retbins=True)
    print('bins: ', bins)
    return se

def to_sub(recs, name_sub):
    sub = recs.groupby('user_id')['item_id'].apply(lambda x:list(x)[:20]).reset_index()
    sub.columns = ['user_id','predictions']
    sub.to_parquet(name_sub, compression='gzip')

#########

def uniqize_cols(df):
    new_cols = []
    for i, col in enumerate(df.columns):
        if i < len(df.columns)/2:
            col = 'a_' + col
        new_cols.append(col)
    df.columns = new_cols

####################


def get_scores(list_models):
    res = []
    for model in list_models:
         res.append((model.get_best_iteration(), model.get_best_score()['validation']['RMSLE_val']))
    res = pd.DataFrame(res, columns=['iters','RMSLE_val'])
    res['mean_on_folds'] = res['RMSLE_val'].mean()
    res['mean_iters'] = res['iters'].mean()
    return res

def get_importance(list_models):
    fi = list_models[0].get_feature_importance(prettified=True).set_index('Feature Id')

    if len(list_models) == 1:
        return fi

    for model in list_models[1:]:
        fi = fi + model.get_feature_importance(prettified=True).set_index('Feature Id')

    return fi.sort_values('Importances',ascending=False)  


def p(*args):
    for i, a in enumerate(args):
        if isinstance(a, (pd.Series, pd.DataFrame)):
            display(a)
        else:
            print(a, end='')

        if i < len(args) - 1:
            try:
                len(a)
                print("\n ~")
            except:
                print(" | ", end='')
    print()


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_pickle(file_name, data, verbose=False):
    if verbose:
        print('save: ', file_name)
    with open(file_name, 'wb') as f:
        # , protocol=pickle.HIGHEST_PROTOCOL
        pickle.dump(data, f)


def select_cols(df, names):
    if not isinstance(names, list):
        names = [names]
    new_cols = []
    for col in df.columns:
        for name in names:
            if col.startswith(name):
                new_cols.append(col)
    return new_cols


def flat_cols(df, pre='k', columns=True):

    def f(se):
        return [
            pre + '_' + '_'.join(map(str, col)) if type(col) is tuple else pre + '_' +
            str(col) for col in se.to_numpy()
        ]

    if columns:
        df.columns = f(df.columns)
    else:
        df.index = f(df.index)


def mem(df):
    memory = df.memory_usage().sum() / 1024**2
    print(f'Память: {round(memory)} Мб')


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if str(col_type)[:5] == "float":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.finfo("f2").min and c_max < np.finfo("f2").max:
                # np.float16 не принимает бывает
                df[col] = df[col].astype(np.float32)
            elif c_min > np.finfo("f4").min and c_max < np.finfo("f4").max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
        elif str(col_type)[:3] == "int":
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo("i1").min and c_max < np.iinfo("i1").max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo("i2").min and c_max < np.iinfo("i2").max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo("i4").min and c_max < np.iinfo("i4").max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo("i8").min and c_max < np.iinfo("i8").max:
                df[col] = df[col].astype(np.int64)
        # elif col == "timestamp":
        #     df[col] = pd.to_datetime(df[col])
        # elif str(col_type)[:8] != "datetime":
        #     df[col] = df[col].astype("category")
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Память ДО: {round(start_mem,1)} Мб')
    print(f'Память ПОСЛЕ: {round(end_mem,1)} Мб')
    print('Уменьшилось на', round(start_mem - end_mem, 2), 'Мб (минус',
          round(100 * (start_mem - end_mem) / start_mem, 1), '%)')
    return
