import re
import json
import pymorphy2
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pymorphy2
from pandarallel import pandarallel
from typing import Union, Any, Optional,Set, Tuple,List

pandarallel.initialize(progress_bar=False)

nltk.download("stopwords")
stopws = stopwords.words("russian")

clean_tags = re.compile('<.*?>')

def remove_html_tags(text: str) -> str:
    """Remove html tags from a string"""    
    return re.sub(clean_tags, ' ', text).replace('&nbsp;',' ')

def unpack(d:dict) -> List[str]:
    texts = []
    for k, v in d.items():
        texts.append(k)
        if k.lower() == 'цвет':
            continue
        texts.extend(set(v))
    return texts

def join_text_fields(fields:str) -> str:
    fields = json.loads(fields)

    feats = {}
    feats.update(fields['custom_characteristics'])
    feats.update(fields['defined_characteristics'])
    feats.update(fields['filters'])

    all_text = [fields['title'], remove_html_tags(fields['description'])]
    all_text.extend(fields['attributes'])
    all_text.extend(unpack(feats))

    return ' '.join(all_text)

morph = pymorphy2.MorphAnalyzer()

def lemmatize(token:str) -> str:
    return morph.normal_forms(token)[0]

repls = {
    "м": "метр",
    "v": "вольт",
    "в": "вольт",
    "w": "ватт",
    "см": "см",
    "sm": "см",
    "mm": "мм",
    "мм": "мм",
    "ml": "мл",
    "мл": "мл",
    "h": "час",
    "г": "гр",
    "g": "гр",
    "л": "литр",
    "mah": "мач",
    "ah": "ач",
    "гб": "гб",
    "gb": "гб",
    "mb": "мб",
    "мб": "мб",
    "kb": "кб",
    "кб": "кб",
    "tb": "тб",
    "тб": "тб",
}

def clean_by_word(text:str, do_lemma:bool=False, min_sim:int=2) -> str:
    """удаляет стоп-слова, короткие слова, лематизация"""
    tokens = []
    for token in text.split():
        if len(token) < min_sim:
            continue

        if do_lemma and ("а" <= token[0] <= "я") and ("а" <= token[-1] <= "я"):
            token = lemmatize(token)

        parts = re.split("\d+", token)
        if len(parts) > 1:
            w = repls.get(parts[-1])
            if w:
                token = w

        if token not in stopws:
            tokens.append(token)

    return " ".join(tokens)

def clean_text(text_col:pd.Series, low:bool=True, do_lemma:bool=True, min_sim:int=2):
    # понижение регистра, удаление ссылок и знаков препинания
    if low:
        text_col = text_col.str.lower()
        
    text_col = text_col.str.replace("[^A-Za-z0-9А-Яа-я]+", " ", regex=True)

    # удаление отдельно стоящих цифр
    text_col = text_col.str.replace("(?<=\s)[\d\s]+(?=\s)", "", regex=True)
    text_col = text_col.str.replace("^\d+(?=\s)|(?<=\s)\d+$", "", regex=True)

    # удаление таких последовательностей 50х50x50
    text_col = text_col.str.replace("\d+[x,х,×][\d,x,х,×]+", " 777", regex=True)
    text_col = text_col.str.replace("(?<=\s)777(?=\s)", "", regex=True)

    # удаление стоп-слов, опечаток, лематизация
    text_col = text_col.parallel_apply(clean_by_word, args=(do_lemma,min_sim))

    return text_col

def clean_cat_name(text_col:pd.Series) -> pd.Series:
    # убираем одинаковое у всех "Все категории-"
    text_col = text_col.str[15:]
    text_col = text_col.str.lower()
    text_col = text_col.str.replace("[^А-Яа-я]+", " ", regex=True)
    text_col = text_col.str.replace("\s+", " ", regex=True)
    text_col = text_col.str.replace("(\sи\s|\sдля\s)", " ", regex=True)

    return text_col


def number_words(text):
    n = 0
    for word in text.split():
        if len(word)>1:
            n+=1
    return n