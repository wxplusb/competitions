import numpy as np
import pandas as pd
import gc
import pickle
import collections
import itertools
from catboost import CatBoostRanker

RANDOM_STATE = 34

all_days = set(range(25, 213))
max_active_days = len(all_days)


def select_cols(df, names):
    new_cols = []
    for col in df.columns:
        for name in names:
            if col.startswith(name):
                new_cols.append(col)
    return new_cols


def not_active_days(x):
    return list(all_days.difference(x))


def mean_intr_days(x):
    if len(x) < 2:
        return 0
    return np.diff(np.sort(np.array(x))).mean()


def num_share_active_days(x):
    return max_active_days - len(set(x['p3b_days']).union(set(x['p3r_days'])))


def process_not_active(df):
    se = df.loc[:, ['p3b_days', 'p3r_days']].apply(
        num_share_active_days, axis=1)
    df['p3b_days'] = se / df['p3b_num_active_days']
    df['p3r_days'] = se / df['p3r_num_active_days']

    return df


def rtk_stats(col):
    ls_ids = list(col)
    rtk_no = collections.Counter(itertools.chain(*ls_ids))
    return pd.DataFrame(rtk_no.most_common(), columns=["id", "count"])


def probs_to_preds(probs, bank_ids, n=100):
    def trim_list(se):
        return se[:n].tolist()
    probs = probs.sort_values(
        ['bank', 'pred'], ascending=False)
    probs = probs.groupby('bank', observed=True, sort=False)[
        'rtk'].agg(trim_list)
    probs = probs.reindex(bank_ids)
    probs.name = 'rtk'

    probs = probs.to_frame()
    probs.reset_index(inplace=True)
    probs.sort_values('bank', inplace=True)

    return probs


def calc_weights(probs, target, k=1):
    q = probs.groupby('rtk').agg({'pred':'count','rank':'sum'})
    q = (q['rank'].sum()/q['rank'])*k
    
    mean_q = q.mean()
    q = q.reindex(target).fillna(mean_q)
    
    weights = np.log(q)
    weights = weights/weights.max()
    weights.name = 'weights'
    return weights


def predict_probs(test_bank, test_rtk, model, batch=50, n=100, zeros_prob=0.83):

    preds = []

    Npart = len(test_bank)//batch + 1

    for i in range(Npart):
        # print(i, i*batch,(i+1)*batch)

        if i*batch >= len(test_bank):
            break

        part = test_bank[i*batch:(i+1)*batch].merge(test_rtk,
                                                    how='cross')
        part = process_not_active(part)

        x = part.loc[:, part.columns[~part.columns.isin(['bank', 'rtk'])]]
        if isinstance(model, CatBoostRanker):
            part['pred'] = model.predict(x)
        else:
            part['pred'] = model.predict_proba(x)[:, 1]

        del x
        gc.collect()

        bank_ids = test_bank['bank'][i*batch:(i+1)*batch]
        zeros_part = pd.DataFrame(bank_ids, columns=['bank'])
        zeros_part['rtk'] = 0.
        zeros_part['pred'] = zeros_prob

        part = pd.concat((part, zeros_part))

        part = part.loc[:, ['bank', 'rtk', 'pred']].sort_values(
            ['bank', 'pred'], ascending=False)

        part['rank'] = part.groupby('bank')['pred'].rank(
            'first', ascending=False)

        part = part[part['rank'] < (n+1)]

        preds.append(part)

    preds = pd.concat(preds)

    preds['rank'] = preds['rank'].max() - preds['rank'] + 1

    return preds


def predict(test_bank, test_rtk, model, batch=25, n=100, zeros_prob=0.83):

    def trim_list(se):
        return se[:n].tolist()

    preds = []

    Npart = len(test_bank)//batch + 1

    for i in range(Npart):

        if i*batch >= len(test_bank):
            break

        part = test_bank[i*batch:(i+1)*batch].merge(test_rtk,
                                                    how='cross')
        part = process_not_active(part)

        x = part.loc[:, part.columns[~part.columns.isin(
            ['bank', 'rtk'])]]

        if isinstance(model, CatBoostRanker):
            part['pred'] = model.predict(x)
        else:
            part['pred'] = model.predict_proba(x)[:, 1]

        del x
        gc.collect()

        part = part.loc[:, ['bank', 'rtk', 'pred']]

        bank_ids = test_bank['bank'][i*batch:(i+1)*batch]
        zeros_part = pd.DataFrame(bank_ids, columns=['bank'])
        zeros_part['rtk'] = 0.
        zeros_part['pred'] = zeros_prob

        part = pd.concat((part, zeros_part))

        part = part.sort_values(
            ['bank', 'pred'], ascending=False)

        part = part.groupby('bank', observed=True, sort=False)[
            'rtk'].agg(trim_list)

        preds.append(part)

    preds = pd.concat(preds)
    preds.name = 'rtk'

    preds = preds.to_frame()
    preds.reset_index(inplace=True)
    preds.sort_values('bank', inplace=True)

    return preds


def flat_cols(df, pre='k', columns=True):

    def f(se): return [pre+'_'.join(map(str, col)) if type(col)
                       is tuple else pre+'_'+str(col) for col in se.to_numpy()]

    if columns:
        df.columns = f(df.columns)
    else:
        df.index = f(df.index)


def full_cols(emb, file_name):
    all_cols = load_pickle(file_name)
    emb = emb.reindex(columns=all_cols)
    for col in emb.columns:
        col_type = emb[col].dtypes
        if str(col_type)[:5] == "float":
            emb[col] = emb[col].fillna(0)
    return emb


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj
