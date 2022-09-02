import gc
import pickle
import collections
import itertools
import datetime

import numpy as np
from numpy.random import default_rng
import pandas as pd

from tqdm.notebook import tqdm
from IPython.display import HTML, display

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold



RANDOM_STATE = 34

dir_data = './data/'

def fillna_dom_url(t):
    t["referrer_num"] = t["referrer_num"].replace("target", 10).astype(np.int8)
    t.loc[t.domain.isna(),'domain'] = t.loc[t.domain.isna(),'url'].str.split('/').str[0].str[-9:]
    
    t.rename(columns={'referrer_domain':'ref', "referrer_url": 'url_ref', 'referrer_num':'pos', "is_referrer": 'y'}, inplace=True)

    if "ref" in t.columns:
        t['y'] = t['y'].astype(np.int8)
        t["ref"] = t["ref"].fillna("n")
        t["url_ref"] = t["url_ref"].fillna("n")

    t.sort_values(['event_group_id', 'pos'], inplace=True)
    t.reset_index(drop=True, inplace=True)

    return t
        
def split_domain(t, name_col):
    t['root_' + name_col] = t[name_col].str[-9:-3].replace('','n')
    t['ltd_'+ name_col] = t[name_col].str[-2:]
    
def split_path(t, name_col):
    t['path_' + name_col] = t[name_col].str[-6:]

def encode_cols(t, mapping):
    encoders = load_pickle(dir_data+"encs.pickle")

    for (col, enc) in mapping:
        t[col] = encoders[enc].transform(
         t[col]).astype(np.int32 if enc != 'ltd_enc' else np.int16) + 1

def count_parts_url(t):
  t['ff_count_slash'] = t['url'].str.count('/')
  t.loc[t['ff_count_slash']>7,'ff_count_slash'] =7

  t['ff_count_subdomain'] = t['url'].str.count('\.')
  t.loc[t['ff_count_subdomain']>3,'ff_count_subdomain'] =3

def add_cats_1(stat,cols,path_cats):
    cats = pd.read_parquet(path_cats,columns=['domain','category_1']).set_index('domain')
    for col in cols:
        # может заменить на -1        
        cat_domain = cats.reindex(stat[col],fill_value=0)
        cat_domain.index = stat.index
        stat['cat_'+col] = cat_domain

def make_pairs(t, col_pairs):
    k = 10_000_000
    for (col1, col2) in col_pairs:
        t["pair_" + col1] = t[col1].astype(np.int64) * k + t[col2].astype(np.int64)        

def calc_stats(stat):
    stats = {}

    # как часто пара (domain , ref) встречается
    for col in ['domain', 'root_domain', 'ltd_domain', 'cat_domain','url','path_url']:
        stats['pair_count_' + col] = stat.groupby('pair_' +
                                                  col)['device_id'].nunique()

    # как часто рефер выступает рефом (1 - по девайсам, 2 - по доменам)
    for col in ['ref', 'root_ref', 'ltd_ref','cat_ref','url_ref','path_url_ref']:
        stats['count_dev_' + col] = stat.groupby(col)['device_id'].nunique()
        stats['count_dom_' + col] = stat.groupby(col)['domain'].nunique()

        
    between_cols = [('root_ref','root_domain'),
                   ('ltd_ref','ltd_domain'),
                   ('cat_ref','cat_domain'),
                   ('url_ref','url'),
                   ('path_url_ref','path_url')]
    
    for (c1,c2) in between_cols:
        stats['count_' + c1 +'_'+c2] = stat.groupby(c1)[c2].nunique()

    return stats       


adjust_map = [
    ('pair_count_domain','pair_target'),
               ('pair_count_root_domain','pair_root_target'),
              ('pair_count_ltd_domain','pair_ltd_target'),
              ('pair_count_cat_domain','pair_cat_target'),
               ('pair_count_url','pair_url_target'),
              ('pair_count_path_url','pair_path_url_target'),
              
             ('count_dev_ref', 'domain'),
              ('count_dom_ref', 'domain'),
              
              ('count_dev_root_ref', 'root_domain'),
              ('count_dom_root_ref', 'root_domain'),
              
              ('count_dev_ltd_ref', 'ltd_domain'),
              ('count_dom_ltd_ref', 'ltd_domain'),
              
              ('count_dev_cat_ref', 'cat_domain'),
              ('count_dom_cat_ref', 'cat_domain'),
              
              ('count_dev_url_ref', 'url'),
              ('count_dom_url_ref', 'url'),
              
              ('count_dev_path_url_ref', 'path_url'),
              ('count_dom_path_url_ref', 'path_url'),
              
              ('count_root_ref_root_domain', 'root_domain'),
             ('count_ltd_ref_ltd_domain', 'ltd_domain'),
             ('count_cat_ref_cat_domain', 'cat_domain'),
             ('count_url_ref_url', 'url'),
             ('count_path_url_ref_path_url', 'path_url')]

def add_stats_features(t, sts):
    
    def adjust(stat_col, col):
        ff = sts[stat_col].reindex(t[col],fill_value=0).astype(np.float32)
        ff.index = t.index
        return ff
    
    for stat_col, col in adjust_map:
        t['ff_'+stat_col] = adjust(stat_col, col)
        
    t['ff_prob_ref'] = (t['ff_pair_count_domain'] / t['ff_count_dev_ref']).fillna(0)
    t['ff_prob_ref_root'] = (t['ff_pair_count_root_domain'] / t['ff_count_dev_root_ref']).fillna(0)
    t['ff_prob_ref_ltd'] = (t['ff_pair_count_ltd_domain'] / t['ff_count_dev_ltd_ref']).fillna(0)
    t['ff_prob_ref_cat'] = (t['ff_pair_count_cat_domain'] / t['ff_count_dev_cat_ref']).fillna(0)
    t['ff_prob_ref_url'] = (t['ff_pair_count_url'] / t['ff_count_dev_url_ref']).fillna(0)
    t['ff_prob_ref_path_url'] = (t['ff_pair_count_path_url'] / t['ff_count_dev_path_url_ref']).fillna(0)
    return t

def time_features(t):
    # FFF1
    t["ff_delta_time"] = t.groupby("event_group_id")["timestamp"].transform(
            "last") - t["timestamp"]

    # FFF2
    t["ff_ratio_time"] = (t["ff_delta_time"]/t.groupby("event_group_id")\
                ["ff_delta_time"].transform("max")).fillna(0)

    # FFF3
    q = (t.groupby(["device_id",'event_group_id'])["ff_delta_time"].agg(
        'max')/10).groupby('device_id').mean()
    q = q.reindex(t['device_id'])
    q.index = t.index
    t["ff_device_ratio_time"] = (t["ff_delta_time"]/q).fillna(0)
    return t

def fit_catboost(X_train,
                 y_train,
                 X_val,
                 y_val,
                 lr=0.1,
                 iters=3500,
                 v=500,
                 auto_class_weights=None,
                 weights=None,
                 random_state=RANDOM_STATE,
                 dir_out=''):

    from catboost import CatBoostRanker, CatBoostClassifier

    cb = CatBoostClassifier(iterations=iters,
                            random_seed=random_state,
                            early_stopping_rounds=100,
                            auto_class_weights=auto_class_weights,
                            
                            learning_rate=lr)

    cb.fit(X_train, y_train, eval_set=(X_val, y_val), sample_weight=weights,verbose=v)
    name_cb = f"model_LR_{lr}_SEED_{random_state}.cbm"
    print(name_cb)
    cb.save_model(dir_out + name_cb)
    config = {}
    config['NAME_CB'] = name_cb
    config['LR'] = lr
    config['BEST_ITER'] = cb.get_best_iteration()

    return cb, config


def stratify_device_group_split(t,
                                n_splits_level1=3,
                                n_splits_level2=3,
                                random_state=34):
    data = t.loc[t.y == 1,
                 ['event_group_id', 'device_id', 'pos']]

    def split_(data,n_splits):
      cv = StratifiedGroupKFold(n_splits=n_splits,
                                shuffle=True,
                                random_state=random_state)
      X = data['event_group_id']
      y = data['pos']
      groups = data['device_id']

      return enumerate(cv.split(X, y, groups))

    device_folds = []

    for i,(train_ids, val_ids) in split_(data, n_splits_level1):
      train_dev = data['device_id'].iloc[train_ids].unique()
      val_dev = data['device_id'].iloc[val_ids].unique()
      device_folds.append({'level_1':(train_dev,val_dev),\
                                      'level_2':[]})

      if n_splits_level2:
        data2 = data.iloc[val_ids]
        for _,(train_ids, val_ids) in split_(data2, n_splits_level2):
          train_dev = data2['device_id'].iloc[train_ids].unique()
          val_dev = data2['device_id'].iloc[val_ids].unique()
          device_folds[-1]['level_2'].append((train_dev,val_dev))


    return device_folds

def get_devices_from_folds(folds, ix_level1, ix_level2=None):
  l1 = folds['Fold_'+str(ix_level1)]['level_1']
  if ix_level2 is None:
    return l1[0],l1[1]
  l2 = folds['Fold_'+str(ix_level1)]['level_2'][ix_level2]
  return l1[0],l2[0],l2[1]

def pair_counts(t, col):
    targets = t.loc[t.pos==10,col].unique()
    results = []
    col_tar = col + '_tar'

    for i in range(11):

        target_rows = t.loc[t.pos == i, ['event_group_id', col, 'pos', 'timestamp']].set_index('event_group_id')
        target_rows.columns = [col_tar, 'pos_tar', 'timestamp_tar']
        # print(1,t.shape)
        t1 = t.merge(target_rows, left_on='event_group_id', right_index=True)
        # print(2,t1.shape)
        t1['diff_timestamp'] = t1['timestamp_tar'] - t1['timestamp']
        t1['diff_pos'] = t1['pos_tar'] - t1['pos']
        
        make_pairs(t1, [(col_tar, col)])

        t1 = t1.loc[t1[col_tar].isin(targets) & (t1['diff_pos'] != 0),\
         ['device_id', 'pair_'+col_tar, 'diff_timestamp', 'diff_pos']]
        results.append(t1)
        # print(3,t1.shape)
        del t1
        gc.collect()
    return pd.concat(results)

def stat_pair_counts(q):
    pair_col = q.columns[1]
    print(pair_col)
    q = q.pivot_table(index=pair_col,
         columns=(q['diff_pos'] >= 0).astype(np.int8), values=['diff_timestamp', 'diff_pos'],aggfunc={
                                'diff_timestamp': ['mean', 'sum'],
                                'diff_pos': ['mean', 'sum', 'count']
                            },
                            fill_value=0,
                            sort=False)
    q['diff_pos','count'] = q['diff_pos','count'].div(q['diff_pos','count'].sum(axis=1),axis=0)
    
    q['diff_pos','sum',0] = - q['diff_pos','sum',0]
    q['diff_pos','sum'] = q['diff_pos','sum'].\
                        div(q['diff_pos','sum'].sum(axis=1),axis=0)
    
    q['diff_timestamp','sum',0] = - q['diff_timestamp','sum',0]
    q['diff_timestamp','sum'] = q['diff_timestamp','sum']\
                       .div(q['diff_timestamp','sum'].sum(axis=1),axis=0)
    flat_cols(q,'ff')
    return q.fillna(0).astype(np.float32)

def merge_stat_pair_counts(t,col='domain',merge_col='pair_target'):
    results = pair_counts(t,col)
    stats = stat_pair_counts(results)
    t1 = t.merge(stats,left_on=merge_col,right_index=True)
    t1.sort_values(['event_group_id','pos'],inplace=True)
    t1.reset_index(drop=True,inplace=True)
    return t1

def save_results(data, clear=False):

    data = [datetime.datetime.now()] + data

    cols = [
        'tm', 'f1', 'precision', 'recall', 'desc', 'lr', 'iters', 'split_stat',
        'split_train', 'len_stat', 'len_train', 'len_val'
    ]

    assert len(data) == len(cols)
    if clear:
        results = pd.DataFrame(columns=cols)
    else:
        results = pd.read_csv('results.csv')
    results.loc[len(results)] = data
    results.sort_values('f1', ascending=False, inplace=True)
    results.to_csv('results.csv', index=False)


def calc_preds(val_df):
    y_true = val_df.groupby('event_group_id')['y'].agg(
        lambda x: x.argmax())
    y_pred = val_df.groupby('event_group_id')['probs'].agg(
        lambda x: x.argmax())
    return y_true, y_pred


def scores(val_df, print_score=True, no_avg=True):
    y_true, y_pred = calc_preds(val_df)

    precision = round(precision_score(y_true, y_pred, average='macro'), 3)
    recall = round(recall_score(y_true, y_pred, average='macro'), 3)
    f1 = round(f1_score(y_true, y_pred, average='macro'),3)
    if print_score:
        print(f'f1: {f1} | precision: {precision} | recall: {recall} |')
    if no_avg:
        precision = np.round(precision_score(y_true, y_pred, average=None), 3)
        recall = np.round(recall_score(y_true, y_pred, average=None),3)
        f1 = np.round(f1_score(y_true, y_pred, average=None), 3)

        results = pd.DataFrame({'f1': f1, 'precision': precision, 'recall': recall})
        display(results)
        
    return f1, precision, recall

def to_sub(test, probs, name):
    test_df = test[['event_group_id']].copy()
    test_df['probs'] = probs
    test_df = test_df.groupby('event_group_id')['probs'].agg(
        lambda x: x.argmax())
    test_df.name = 'referrer_num'
    test_df.to_csv(f'subs/{name}')
    print(f'Сохранено в файл: subs/{name}')


def p(*args):
    for a in args:
        print(a, "\n ~")


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_pickle(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def select_cols(df, names):
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
                df[col] = df[col].astype(np.float32) # np.float16 не принимает бывает
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