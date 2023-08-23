import pickle
import numpy as np
import pandas as pd
import os
import json
import math
from tqdm.notebook import tqdm
import torch
import random

try:
    from IPython.display import display
except:
    pass

# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=True,nb_workers=5)

SEED = 34
N_CPU = os.cpu_count()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 34) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def len_in_tokens(se, tokenizer, plot=True):
    encs = tokenizer(se.to_list(), padding=False, truncation=False)
    t_len = pd.Series([len(x) for x in encs["input_ids"]])
    if plot:
        t_len.hist(bins=50)
    return t_len


def get_scores(list_models, final=False):
    res = []
    for model in list_models:
        if final:
            score = model.get_evals_result()['validation']['AUC'][-1]
        else:
            score = model.get_best_score()['validation']['AUC']
        res.append((model.get_best_iteration(), score))
    res = pd.DataFrame(res, columns=['iters','AUC'])
    res['mean_on_folds'] = res['AUC'].mean()
    res['std_on_folds'] = res['AUC'].std()
    res['mean_iters'] = res['iters'].mean()
    return res



def get_importance(list_models):
    if not isinstance(list_models,list):
        list_models = [list_models]

    fi = list_models[0].get_feature_importance(prettified=True).set_index('Feature Id')

    if len(list_models) == 1:
        return fi

    for model in list_models[1:]:
        fi = fi + model.get_feature_importance(prettified=True).set_index('Feature Id')

    return fi.sort_values('Importances',ascending=False) 

def cb_predict_proba(list_models, X_test):

    X_test = X_test.copy()
    for col in list_models[0].feature_names_:
        if col not in X_test.columns:
            X_test[col] = 0

    X_test = X_test[list_models[0].feature_names_]

    pred = list_models[0].predict_proba(X_test)[:,1]

    if len(list_models) == 1:
        return pred

    for model in list_models[1:]:
        pred = pred + model.predict_proba(X_test)[:,1]

    return pred/len(list_models)


def add_folds(df:pd.DataFrame, strat_col = None, group_col=None, n_folds:int=5, seed:int=34) -> pd.DataFrame:
    if (strat_col is not None) and (group_col is None):
        print('StratifiedKFold')
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    elif (strat_col is not None) and (group_col is not None):
        print('StratifiedGroupKFold')
        from sklearn.model_selection import StratifiedGroupKFold

        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    if type(group_col) == str:
        group_col = df[group_col]

    if type(strat_col) == str:
        strat_col = df[strat_col]

    ixs_gen = skf.split(df,strat_col,group_col)

    df['fold'] = -1
    for fold, (trn_, val_) in enumerate(ixs_gen):
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


# def get_importance(list_models):
#     fi = list_models[0].get_feature_importance(prettified=True).set_index('Feature Id')

#     if len(list_models) == 1:
#         return fi

#     for model in list_models[1:]:
#         fi = fi + model.get_feature_importance(prettified=True).set_index('Feature Id')

#     return fi.sort_values('Importances',ascending=False)  


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
