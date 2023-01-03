import pickle
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from typing import Tuple,List 

try:
    from IPython.display import display
except:
    pass

RANDOM_STATE = 34
N_CPU = os.cpu_count()

VOWELS = set('аеёиоуыэюя')
CONSONANTS = set('бвгджзйклмнпрстфхцчшщьъ')
LETTERS = VOWELS.union(CONSONANTS)

GLU = set('кпстфхцшчщ')
ZVON = set('бвгджзлмнрй')
PAR = set('бвгджзкпстф')
SHIP = set('шжщч')
MIAG = set('иеяюёь')

AUG_GLU = list('кпстфхцшч')
AUG_ZVON = list('бвгджзлмнр')
AUG_CONS = AUG_GLU + AUG_ZVON

MAX_WORD_LEN = 20

char2int = {l:(i+1) for i,l in enumerate(sorted(LETTERS))}

# make such table for embeddings
# a 000101
# б 101101
# в 010101
def get_letter_feats() -> List[List[int]]:
    groups = [VOWELS, CONSONANTS, GLU, ZVON, PAR, SHIP, MIAG]
    feats_table = [[0]*len(groups)]
    let_emb = []
    for letter in sorted(LETTERS):
        for group in groups:
            flag = 1 if letter in group else 0
            let_emb.append(flag)
        feats_table.append(let_emb)
        # print(letter,let_emb)
        let_emb = []
    return feats_table

# for augment
def stem(row:pd.Series) -> pd.Series:

    lemma = row.lemma

    if lemma.endswith('й') or lemma.endswith('ть') or lemma.endswith('ие'):
        lemma = lemma[:-3]
    elif lemma.endswith('ик'):
        lemma = lemma[:-2]

    i = 0
    for i,(l1,l2) in enumerate(zip(row.word,lemma)):
        if l1!=l2:
            i -= 1
            break
    if i <= 0:
        return pd.Series(['',''])
    return pd.Series([row.word[:i+1],row.word[i+1:]])

def ids_cons(s:str) -> List[int]:
    ids = []
    for i, l in enumerate(s):        
        if l in AUG_CONS:
            ids.append(i)
    return ids

def augment(row:pd.Series, rng:np.random.Generator, p:float=0.3) -> Tuple[str,str]:
    word = list(row.word)
    lemma = list(row.lemma)
    for i in row.ids_cons_in_stem:
        if rng.random() > p:
            continue

        if word[i] in AUG_GLU:
            new_letter = rng.choice(AUG_GLU)
            word[i] = new_letter
            lemma[i] = new_letter

        elif word[i] in AUG_ZVON:
            new_letter = rng.choice(AUG_ZVON)
            word[i] = new_letter
            lemma[i] = new_letter

    return ''.join(word), ''.join(lemma)
# for augment


def logits2labels(logits:np.ndarray) -> np.ndarray:
    return logits.argmax(axis=1) + 1

def stress_on_io(w:str)-> int:
    pos = 0
    for l in w:
        if l in VOWELS:
            pos += 1
        if l == 'ё':
            return pos
    raise Exception('not io in word: ',w)

def correct_preds(df:pd.DataFrame) -> pd.DataFrame:
    df['stress'] = df['stress'].clip(1,df['num_syllables'])

    filtr_io = df['word'].str.contains('ё')

    df.loc[filtr_io,'word'] = df.loc[filtr_io,'word'].apply(stress_on_io)

    return df

def make_sub(test_path:str, dir_out:str, name_sub:str) -> None:
    pred_paths = sorted(Path(dir_out).rglob('*.npy'))
    logit_preds = [np.load(p_) for p_ in pred_paths]
    logit_preds = np.stack(logit_preds).mean(axis=0)

    df_test = pd.read_csv(test_path)

    labels = logits2labels(logit_preds)
    df_test['stress'] = labels
    df_test = correct_preds(df_test)
    df_test[['id','stress']].to_csv(f'{dir_out}{name_sub}',index=False)

def add_folds(df:pd.DataFrame, n_folds:int=5, random_state:int=34) -> pd.DataFrame:
    df['stratify'] = df.groupby(['num_syllables','stress'])['id'].transform('count')

    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    df['fold'] = -1
    for fold, (trn_, val_) in enumerate(skf.split(df,df['stratify'],groups=df['lemma'])):
        df.loc[val_,'fold'] = fold

    return df

def get_remain(s:str) -> str:
    n_con = 0
    for i in s:
        if i in CONSONANTS:
            n_con += 1
        else:
            break

    if n_con == 0:
        return ''

    cons = s[:n_con]

    k = cons.find('ь')
    if k != -1:
        return cons[:k+1]

    k = cons.find('ъ')
    if k != -1:
        return cons[:k+1]

    if n_con == 1:
        return ''
    elif n_con == 2:
        return cons[:1]

    return cons[:2]

def count_vowels(w:str) -> int:
    n = 0
    for l in w:
        if l in VOWELS:
            n += 1
    return n

def split_by_syls(w:str):

    n = count_vowels(w)
    syls = []
    sl = ''

    i = 0
    while i < len(w):
        s = w[i]
        
        sl += s

        if s in VOWELS:
            n -= 1
            if n == 0:
                sl += w[i+1:]
                syls.append(sl)
                break

            letters = get_remain(w[i+1:])
            sl += letters
            syls.append(sl)
            sl = ''
            i += len(letters)

        i+=1

    return syls



def word_to2d(s:str) -> np.ndarray:
    w = np.zeros((len(LETTERS),MAX_WORD_LEN),dtype=np.float32)

    x_i = np.arange(len(s))
    y_i = []
    
    for i,l in enumerate(s):
        y_i.append(char2int[l])

    w[y_i,x_i] = 1.

    return w


def uniq_cols(df:pd.DataFrame):
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

