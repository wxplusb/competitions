import re

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader,Dataset
from typing import Union, Any, Optional,Set, Tuple,List
import pytorch_lightning as pl

import string
from .goodz import chs_g
from .brandz import chs_b
import cyrtranslit as tr

def my_translit(s):
    ru_s = tr.to_cyrillic(s, "ru")

    ru_s = ru_s.replace('х','кс').replace('h','х').replace('лукс','люкс').replace('еи','ей').replace('ь',"'").replace('ы',"'").replace('ай',"эй")

    if 'ee' in s:
        ru_s = ru_s.replace('ее','и')

    if not s.endswith('ee') and s.endswith('e'):
        ru_s = ru_s[:-1]

    if ru_s.endswith('еа'):
        ru_s = ru_s[:-2] + 'и'

    ru_s = ru_s.replace('ж','дж').replace('хн','н').replace('nj','нж')

    if ru_s.startswith('й'):
        ru_s = 'дж' + ru_s[1:]

    return ru_s

def rus_2words_variants(s):
    vrs = []
    ws = re.findall('[а-яё]+',s)
    if len(ws) != 2:
        return []
    
    for i in range(2,15):
        for j in range(3,15):
            vr = ws[0][:i] + ws[1][:j]
            if len(vr) >= 6:
                vrs.append(vr)

    return vrs

bs2variants = [
 'fruit-tella',
 'office clean',
]

def generate_brands(s):
    # print(s)
    gen_variants = []
    if (s[0] >= 'a') and (s[0] <= 'z') and (len(s) > 4):
        vr = my_translit(s)
        gen_variants.append(vr)
        gen_variants.append(vr.replace('е','э'))
        gen_variants.append(vr.replace('е','э', 1))
        gen_variants.append(vr.replace('э','е'))
        gen_variants.append(vr.replace('э','е', 1))
        gen_variants.append(vr.replace('дж','ж'))
        gen_variants.append(vr.replace('лл','л'))
        gen_variants.append(vr.replace('сс','с'))
        gen_variants.append(vr.replace('нн','н'))
        gen_variants.append(vr.replace('а','э'))
        # gen_variants.append(vr.replace('&',''))
        gen_variants.append(vr.replace('&',' & '))
        gen_variants.append(vr.replace('q','g'))
        gen_variants.append(re.sub('(?<!л)л(?!л)','ль',vr))

        # gen_variants.append(vr.replace('у','ю'))
        
        if len(s) > 7 and (s not in bs2variants):
            gen_variants.append(vr.replace('ч','х'))
            gen_variants.append(vr.replace('х','ч'))
            gen_variants.append(vr.replace(' ',''))
            gen_variants.append(vr.replace('-',''))
            gen_variants.append(re.sub('[\.\-\,\s\'\"]','',vr))

    if len(s) > 8 and (s not in bs2variants):
        vr = re.sub('[\.\-\,\s\'\"]','',s)
        if vr != s:
            gen_variants.append(vr)
        vr = s.replace(' ','')
        if vr != s:
            gen_variants.append(vr)

    if (s[0] >= 'а') and (s[0] <= 'я'):
        gen_variants.append(s.replace('и','й'))
        gen_variants.append(s.replace('&',' & '))
        gen_variants.extend(rus_2words_variants(s))

    return list(set(gen_variants))

def punc_split_tokens(s):
    s = re.split('(,|\s+|\(|-|\)|\.|\d+|\/|\"|\'|\`|:)',s)
    return [el for el in s if el.strip() != '']

def replace_misc(s, repl_e = True):
    if repl_e:
        s = s.replace('ё', 'е').replace('Ё', 'Е')

    s = s.replace('_', ' ').replace('\n', ' ').replace('`',"'")
    # '2) ZB027-01X-LR ФУТБОЛКА ЖЕНСКАЯ'
    res = re.findall('^\d{1,2}\) [A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+ ',s)
    if res:
        s = s.replace(res[0], '')
        s = s + ' @a'

    # '7 2456609958774 майка муж/Orbi/1400'
    res = re.findall('^\d{1,2} [0-9]{10,15} [0-9\.\s]*',s)
    if res:
        s = s.replace(res[0], '')
        if not re.findall('[А-я//]+',s[-4:]):
            s = s[:-4]
        s = s + ' @b'

    # '4.Тканые шорты-бермуды S1ER23Z8 8681380011095'
    res = re.findall(' [A-Z0-9]{8} \d{9,15}$',s)
    if res:
        s = s.replace(res[0], ' @f')
        res = s.find('.')
        if s[1] == '.':
            s = s[2:]
        elif s[2] == '.':
            s = s[3:]

    # 'Круг отрезной 115*1.0*22 мет ИП ФИЛИП'
    res = re.findall('ИП [А-Я][а-я]{5,15}\s?[А-Я]?\.?[А-Я]?\.?',s)
    if res:
        s = s.replace(res[0], ' ')
        s = s + ' @g'

    # СЁМГА Г/К ИП ШУСТОВА
    res = re.findall('ИП [А-Я][А-Я]{6,15}\s?[А-Я]?\.?[А-Я]?\.?',s)
    if res:
        s = s.replace(res[0], ' ')
        s = s + ' @g'

    # [M]!Напиток 1л
    res = re.findall('^\[M\]!?',s)
    if res:
        s = s.replace(res[0], '')
        s = re.sub(' [^\s]{8,20}\s[^\s]{10,20}$', ' ', s)
        s = s + ' @j'

    # 00193658720670 : SHORT N
    res = re.findall('^\d{8,20} : ',s)
    if res:
        s = s.replace(res[0], '')
        s = s + ' @k'

    res = re.findall('^\d{8} \d{2} [\dA-Z]{2} [I\- ]{0,2}',s)
    if res:
        s = s.replace(res[0], '')
        s = s + ' @e'

    # 'мeдь Артикул 25028745882'
    s = re.sub(' [Арти \*&]+кул [0-9]+$', ' @d', s)

    # Пельмени Бульмени 373г Го<13081>
    s = re.sub('<\d+>$', ' @l', s)

    # 9315551 ПОЛОТЕНЦЕ 48x75 Красный 802762969299874166261131098
    res = re.findall('\d{27}$',s)
    if res:
        s = s[8:-28] + ' @m'

    # Чай Зеленый Вкус 3шт ШК=8884031460337
    s = re.sub('ШК=\d{13}$', ' @n', s)

    # LM84031460337
    s = re.sub('(LM|REP|WR)\d{8,9}$', ' @o', s)

    # [М] Шампунь.(04054839472596P2TW1SP30D50Z)
    res = re.findall('\([0-9A-Z]{27}\)$',s)
    if res:
        s = s[:-29] + ' @p'
        if s.startswith('[М] '):
            s = s[4:]

    # 2236206 [М] Обувь весна-лето NEW BALANC
    s = re.sub('^\d{7} [М] ', '', s)

    # 587122ХКБ Туфли мужские, 44, Синие
    res = re.findall('\([0-9A-Z]{27}\)$',s)
    if res:
        s = s[:-29] + ' @p'
        if s.startswith('[М] '):
            s = s[4:]

    # 000000НЧ
    res = re.findall('^\d{6}[А-Я]{2,3} ',s)
    if res and not s.isupper():
        s = s.replace(res[0], '')
        s = s + ' @q'
        
    # NIKE / 107867-00 50-52 Футболка женская белый р. 50-52
    res = re.findall(' / [0-9A-Z]{5,7}-[0-9A-Z]{2,4} [0-9A-Z]{1,3}-*[0-9]{0,2}',s)
    if res:
        s = s.replace(res[0], ' / ')
        s = s + ' @r'

    # 1/1 Пеленки впитывающие
    res = re.findall('^\d+/\d+',s)
    if res:
        s = s.replace(res[0], '', 1)
        s = s + ' @s'

    s = re.sub('[0-9][0-9\,\-]*\s*(кг|КГ)\.?(?![А-я])', ' 0\\1 ', s)
    
    s = re.sub('[0-9][0-9\,\-]*\s*(гр|ГР)\.?(?![А-я])', ' 0\\1 ', s)
    s = re.sub('[0-9][0-9\,\-]*\s*(г|Г)\.?(?![А-я])', ' 0\\1 ', s)

    s = re.sub('[0-9][0-9\,\-]*\s*(мл|МЛ)\.?(?![А-я])', ' 0\\1 ', s)
    s = re.sub('[0-9][0-9\,\-]*\s*(л|Л)\.?(?![А-я])', ' 0\\1 ', s)

    s = re.sub('[0-9][0-9\,\-xXхХ]*\s*(мм|ММ)\.?(?![А-я])', ' 0\\1 ', s)
    s = re.sub('[0-9][0-9\,\-xXхХ]*\s*(см|СМ)\.?(?![А-я])', ' 0\\1 ', s)
    s = re.sub('[0-9][0-9\,\-xXхХ]*\s*(м|М)\.?(?![А-я])', ' 0\\1 ', s)

    s = re.sub('[0-9][0-9\,\-xXхХ]*\s*(шт|ШТ)\.?(?![А-я])', ' 0\\1 ', s)

    # цифры
    s = re.sub('\d+(\.|\-|\,)\d+', '000', s)
    s = re.sub('\d+(\.|\-|\,)\d+', '000', s)

    s = re.sub('\([0-9\.\- \,]*\)', ' ', s)

    if s.startswith('яя') or s.startswith('ЯЯ'):
        s = s[2:]

    if s.startswith('КН '):
        s = 'Книга ' + s[3:]

    # 'д/' на для
    s = re.sub('[^A-Za-zА-я]д[/\\\]', ' для ', s)
    s = re.sub('[^A-Za-zА-я]Д[/\\\]', ' ДЛЯ ', s)

    s = del_space(s)
    return s

def del_space(s):
    s = re.sub('\s+', ' ', s)
    s = s.strip(' -')
    return s

def get_num(se):
    nums = set()
    for b in se:
        num = re.findall('\d+',b)
        nums.update(num)
    return nums

def replace_num_0(s, exclude=[]):
    nums = re.findall('\d+',s)
    # print(nums)
    if not nums:
        return s
    
    for n in sorted(nums, key=len, reverse=True):
        if n not in exclude:
            s = s.replace(n,'0')

    s = re.sub('\d+\s*%', ' %', s)
    s = re.sub('0/+0', '0', s)

    # цифры вначале и вконце
    s = re.sub('^[0\-\s\.\/\,]+', '', s)
    s = re.sub('(?<![A-Za-zА-я])[0\-\s\.]+$', '', s)

    s = re.sub('0[0\*\,\s=xXхХ\.]*0', '0', s)

    s = re.sub('(?<![А-яA-Za-z])[A-Z\d-]+\-[A-Z\d-]+\-[A-Z\d-]+\-[A-Z\d-]+(?![А-яA-Za-z])', '0', s)

    return s

def split_long_words(s):
    res = re.findall('[А-яA-Za-z]{12,50}', s)
    if res:
        new_s = re.sub('([A-ZА-ЯЁ][a-zа-яё]+)', r' \1', re.sub('([A-ZА-ЯЁ]+)', r' \1', res[0]))
        return s.replace(res[0], ' ' + new_s + ' ')
    return s

####
chs_G = {
    'мороженое': ['мороженое пломбир', 'биомороженое'],
    'щетка': ['зубная щетка'],
    'паста': ['зубная паста', 'томатная паста','паста зубная'],
    # 'чистящее средство': ['средство чистящее'],
    "хлопья": ['овсяные хлопья'],
    'лак': ['лaк'],
    'шлифовальный круг': ['круг шлифовальный'],
}

prav_G = {}
for k,v in chs_G.items():
    prav_G[k] = k
    for el in v:
        prav_G[el] = k
        
chs_B = {
    'фрутоняня': ['фруто няня'],
    "o'stin": ['o`stin','ostin'],
    "coca-cola": ['coca cola','кока кола','кока'],
    "milka": ['милка'],
}

prav_B = {}
for k,v in chs_B.items():
    prav_B[k] = k
    for el in v:
        prav_B[el] = k

def correct_train(df):
    df.good = df.good.str.replace('ё','е')
    df.brand = df.brand.str.replace('ё','е')

    df.loc[3901,'good'] = 'автокормушка,автопоилка'
    df.loc[1058,'good'] = 'йогурт'
    df.loc[2945,'good'] = 'роза'
    df.loc[5236,'good'] = 'булка'
    df.loc[4576,'good'] = 'косметика'

    df.loc[df.good.str.contains('напиток'),'good'] = 'напиток'

    df['good'] = df['good'].apply(lambda x:prav_G.get(x,x))
    df['brand'] = df['brand'].map(lambda x:prav_B.get(x,x))
    return df

index_to_tag = ["O", "B-GOOD", "I-GOOD", "B-BRAND", "I-BRAND", "PAD"]
tag_to_index = {tag: index for index, tag in enumerate(index_to_tag)}

def get_entity(text, word, label):
    start = text.find(word)
    entity = {'word': word, 'label': label, 'start':start, 'end':start + len(word)} if start != -1 else None
    return entity

def extract_targets(s):
    return [s_.strip() for s_ in s.split(',')] if s.strip() else []

def extract_word(text, offsets, start, end):
    return text[offsets[start][0]:offsets[end][1]]

def get_ents(seq):

    ents = []
    curr_ent = None

    for i, el in enumerate(seq):
        if el == 'O' and curr_ent:
            ents.append(curr_ent)
            curr_ent = None

        if el.startswith('B-'):
            if curr_ent:
                ents.append(curr_ent)
                
            curr_ent = [el[2:], i, i]

        if el.startswith('I-'):
            if not curr_ent:
                curr_ent = [el[2:], i, i]
            elif el[2:] == curr_ent[0]:
                curr_ent[2] = i
            else:
                ents.append(curr_ent)
                curr_ent = [el[2:], i, i]

    if curr_ent:
        ents.append(curr_ent)

    return ents


def tokenize(df, tokenizer, is_test = True, max_length = None):

    examples = []

    for t in df.itertuples():

        text = t.name.lower()

        tok_seq = tokenizer(text, padding=False, truncation=False, return_length=True, return_offsets_mapping=True, return_special_tokens_mask=True, max_length=max_length, return_tensors='pt')

        ex = {k:v.squeeze() for k,v in tok_seq.items()}
        ex['tokens'] = tok_seq.tokens()
        ex['text'] = text

        if is_test:
            examples.append(ex)
            continue

        goods = extract_targets(t.good)
        brands = extract_targets(t.brand)

        targets = list(zip(goods, ['GOOD']*len(goods))) + list(zip(brands, ['BRAND']*len(brands)))

        entities = []
        for word, label in targets:
            entity = get_entity(text, word, label)
            if entity:
                entities.append(entity)

        tags = ['PAD'] + ['O'] * (tok_seq.input_ids.shape[1]-2) + ['PAD']
    
        for entity in entities:
            start_token_pos = tok_seq.char_to_token(entity['start'])
            # -1 чтобы на пробел не указало, так как он None
            try:
                end_token_pos = tok_seq.char_to_token(entity['end'] - 1) + 1
            except:
                print(text, goods, brands, entities, entity)
                raise Exception()

            for pos in range(start_token_pos, end_token_pos):
                # print(pos)
                label = entity['label']
                tags[pos] = f'B-{label}' if pos == start_token_pos else f'I-{label}'

        # paddings = [tag != 'PAD' for t in tags]    
        # ex['tags'] = tags
        ex['labels'] = torch.tensor([tag_to_index[tag] for tag in tags],dtype=torch.long)
        ex['goods'] = goods
        ex['brands'] = brands
        examples.append(ex)
        
    return examples

EN_CHARS = list(string.ascii_lowercase + '0' + '   ')
RU_CHARS = list('абвгдеёжзиклмнопрстуфхцчшщьъэюя' + '0' + '   ')

def rnd_str(rng, max_words=1):
    if rng.random() <=0.9:
        s = ''.join(rng.choice(RU_CHARS, max_words*4)) # ru
    else:  
        s = ''.join(rng.choice(EN_CHARS, max_words*4)) # en
    return s

def augment_text(row, rng):
    text1 = ['','','','','']
    p0 = rng.random()
    if p0 <= 0.8:
        text1[1] = row['good_1']
        text1[3] = row['brand_1']
    else:
        text1[3] = row['good_1']
        text1[1] = row['brand_1']

    p0 = rng.random()

    if p0 <= 0.6:
        text1[2] = rnd_str(rng, 2)
        text1[4] = rnd_str(rng, 5)
    elif p0 <= 0.8:
        text1[0] = rnd_str(rng, 1)
        text1[4] = rnd_str(rng, 2)
    else:
        text1[0] = rnd_str(rng, 1)
        text1[2] = rnd_str(rng, 2)
        text1[4] = rnd_str(rng, 4)

    text = ' '.join(text1).strip()
    text = re.sub('\s+', ' ', text)
    return text

class ReceiptDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=None, p_aug = 0.6, seed=34):
        self.examples = tokenize(df, tokenizer, is_test = 'good' not in df.columns, max_length = max_length)

        self.tokenizer = tokenizer
        self.rng = np.random
        self.p_aug = p_aug
        self.df = df

    def get_augmented(self, t):
        ex = {}
        text = augment_text(t, self.rng)

        tok_seq = self.tokenizer(text, padding=False, truncation=False, return_length=True, return_offsets_mapping=True, return_special_tokens_mask=True, max_length=None, return_tensors='pt')


        ex = {k:v.squeeze() for k,v in tok_seq.items()}
        ex['tokens'] = tok_seq.tokens()
        ex['text'] = text

        goods = extract_targets(t.good)
        brands = extract_targets(t.brand)

        targets = list(zip(goods, ['GOOD']*len(goods))) + list(zip(brands, ['BRAND']*len(brands)))

        entities = []
        for word, label in targets:
            entity = get_entity(text, word, label)
            if entity:
                entities.append(entity)


        tags = ['PAD'] + ['O'] * (tok_seq.input_ids.shape[1]-2) + ['PAD']
    
        for entity in entities:
            start_token_pos = tok_seq.char_to_token(entity['start'])
            # -1 чтобы на пробел не указало, так как он None
            try:
                end_token_pos = tok_seq.char_to_token(entity['end'] - 1) + 1
            except:
                print(text, goods, brands, entities, entity)
                raise Exception()

            for pos in range(start_token_pos, end_token_pos):
                # print(pos)
                label = entity['label']
                tags[pos] = f'B-{label}' if pos == start_token_pos else f'I-{label}'

        ex['labels'] = torch.tensor([tag_to_index[tag] for tag in tags],dtype=torch.long)
        ex['goods'] = goods
        ex['brands'] = brands
        
        return ex
    
    def __getitem__(self, idx):
        if (self.p_aug > 0) and self.rng.random() <= self.p_aug:
            return self.get_augmented(self.df.iloc[idx])
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)

class NerDataModule(pl.LightningDataModule):
    def __init__(self, Xy:pd.DataFrame, X_test:pd.DataFrame, fold:int=0, n_cpu:int=10, cfg=None):
  
        super().__init__()

        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#hyperparameters-in-datamodules
        # если без аргументов то датасеты сохранит
        self.save_hyperparameters('cfg')

        self.Xy = Xy
        self.X_test = X_test
        self.fold = fold

        self.n_cpu = n_cpu

        self.cfg = cfg

        self.len_train = (self.Xy['fold']!=self.fold).sum()
        if self.cfg.train_full:
            self.len_train = len(self.Xy)

        self.G = torch.Generator()
        self.G.manual_seed(34)

    def setup(self,stage):
        if stage == 'fit':
            if self.cfg.train_full:
                Xy_train = self.Xy
            else:
                Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold]

            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold]

            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)

            self.train = ReceiptDataset(Xy_train, self.tokenizer, max_length=self.cfg.max_length, p_aug = self.cfg.p_aug)
            self.val = ReceiptDataset(Xy_val, self.tokenizer, max_length=self.cfg.max_length, p_aug = 0)
        elif stage == 'predict':
            self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
            self.test = ReceiptDataset(self.X_test, self.tokenizer, max_length=self.cfg.max_length, p_aug = 0)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, collate_fn=collate_fn,  shuffle=True, num_workers=self.n_cpu, generator=self.G, worker_init_fn=seed_worker)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.val_bs, collate_fn=collate_fn, num_workers=self.n_cpu)

    def predict_dataloader(self):
        return DataLoader(self.test, batch_size=self.cfg.val_bs, collate_fn=test_collate_fn, num_workers=self.n_cpu)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def collate_fn(batch):
    # print(batch, len(batch), type(batch[0]))
    input_ids, token_type_ids, attention_mask, offsets, lengths, labels, goods, brands, texts, paddings = [], [], [], [], [], [], [], [], [], []

    # max_seq_len = 0
    for item in batch:
        input_ids.append(item['input_ids'])
        token_type_ids.append(item['token_type_ids'])
        attention_mask.append(item['attention_mask'])
        offsets.append(item['offset_mapping'].numpy())
        paddings.append(item['special_tokens_mask'].numpy())
        lengths.append(item['length'].item())
        labels.append(item['labels'])
        goods.append(item['goods'])
        brands.append(item['brands'])
        texts.append(item['text'])

        # max_seq_len = max(max_seq_len,int(item['length']))

    # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=tag_to_index["PAD"])

    return {'input_ids': input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask,'offsets': offsets, 'paddings': paddings, 'length': lengths,'labels':labels, 'goods':goods, 'brands':brands, 'texts': texts }

def test_collate_fn(batch):
    # print(batch, len(batch), type(batch[0]))
    input_ids, token_type_ids, attention_mask, offsets, lengths, texts, paddings = [], [], [], [], [], [], []

    for item in batch:
        input_ids.append(item['input_ids'])
        token_type_ids.append(item['token_type_ids'])
        attention_mask.append(item['attention_mask'])
        offsets.append(item['offset_mapping'].numpy())
        paddings.append(item['special_tokens_mask'].numpy())
        lengths.append(item['length'].item())
        texts.append(item['text'])

    # https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizerFast
    input_ids = pad_sequence(input_ids, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)

    return {'input_ids': input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask,'offsets': offsets, 'paddings': paddings, 'length': lengths, 'texts': texts }


# 'ёэз'  'еес'
ch1 = 'etyopahkxcbm'
ch1 += ch1.upper()
ch2 = 'етуоранкхсвм'
ch2 += ch2.upper()
en2ru_trans_dict = str.maketrans(dict(zip(ch1, ch2)))
ru2en_trans_dict = str.maketrans(dict(zip(ch2, ch1)))

def en2ru(w):
    return w.translate(en2ru_trans_dict)

def ru2en(w):
    return w.translate(ru2en_trans_dict)

def is_ru_word(w):
    for ch in w:
        if ord(ch) < 1040 or ord(ch) > 1103:
            return False
    return True

def is_en_word(w):
    for ch in w.lower():
        if ch < 'a' or ch > 'z':
            return False
    return True

def mult_replace(text, dic):
    for i, j in dic:
        text = text.replace(i, j)
    return text

# https://stackoverflow.com/questions/227459/how-to-get-the-ascii-value-of-a-character
# https://stackoverflow.com/questions/48255244/python-check-if-a-string-contains-cyrillic-characters
def replace_to_ru(s:str) -> str:
    ru_en_words = []
    words = re.findall('[a-zA-Zа-яА-я]+', s)
    for w in words:
        for ch in w:
            if ord(ch) >= 1040 and ord(ch) <= 1103:
                new_w = en2ru(w)
                if new_w != w and is_ru_word(new_w):
                    # print(1,(w, new_w))
                    ru_en_words.append((w, new_w))
                elif new_w != w:
                    new_w = ru2en(w)
                    if is_en_word(new_w):
                        # print(2,(w, new_w))
                        ru_en_words.append((w, new_w))
                break

    # print(words, ru_en_words)

    return mult_replace(s, ru_en_words)

def caps_to_low(s):
    try:
        if s.isupper() or (len(s)>3 and s[-2]=='@' and s[:-2].isupper()):
            return s.lower() + ' @c'
    except:
        print(1, s)
    return s

# извлекает весь бренд до запятой возможно из нескольких слов
def get_en_word(x):
    x = x.split(',')
    x = map(str.strip,x)
    en_b = []
    for el in x:
        if re.fullmatch('[^А-я]+',el):
            en_b.append(el)
    return en_b

def get_en_brands(li):
    en_brands = []
    for x in li:
        en_brands.extend(get_en_word(x))
    return en_brands

# извлекает отдельные русские слова (не двойные)
def get_sep_ru_word(li):
    ru_words = []
    for x in li:
        ru_words.extend(re.findall('[А-я]+',x))
    return set(ru_words)


def get_long_goods(se):
    long_goods = {}
    all_words_in_goods = set(se.str.split(',|-| ').sum())
    all_words_in_goods.add('трикота')

    for g in sorted(all_words_in_goods,reverse=True):
        if len(g)>7 and (g[:-1] not in all_words_in_goods):
            long_goods[g[:-1]] = g

    return long_goods

def get_replaces_goods(long_goods):
    chs = {}
    for k,v in chs_g():
        for el in v:
            chs[el] = k

    chs.update(long_goods)
    
    return chs


def get_replaces_brands(train_brands, exclude=None):
    chs = {}
    for k,v in chs_b():
        for el in v:
            chs[el] = k

    for k,v in chs.copy().items():
        if len(k) > 6:
            vr = re.sub('[\.\-\,\s\'\"]','', k)
            if vr not in chs:
                chs[vr] = v
            vr = k.replace(' ','')
            if vr not in chs:
                chs[vr] = v

    for b in train_brands:
        for vr in generate_brands(b):
            if (vr not in exclude) and (vr not in chs) and len(vr)>4:
                chs[vr] = b
    
    return chs


def find_in_dict(s, chs):
    word = chs.get(s,'')
    if not word and len(s) > 6:
        word = chs.get(re.sub('[\.\-\,\s\'\"\/]','',s),'')
    return word

def dekor_fn(name):
    if name == 'upper':
        dekor = lambda x:x.upper()
    elif name == 'title':
        dekor = lambda x:x.title()
    elif name == 'capitalize':
        dekor = lambda x:x.capitalize()
    else:
        dekor = lambda x:x
    return dekor

def correct_sent(s, chs, dekor=None):

    dekor = dekor_fn(dekor)
    
    new_s = ''

    while s:
        s_low = s.lower()

        ms = list(re.finditer('[А-яA-Za-z]+|\d+|\+',s))
        L = len(ms)
        word = ''

        for i in range(L):
            n1 = ms[i].start()
            
            for j in range(3,-1,-1):
                if i+j >= L:
                    continue
                n2 = ms[i+j].end()
                # print(s_low[n1:n2])
                word = find_in_dict(s_low[n1:n2], chs) or find_in_dict(s[n1:n2], chs)

                if word:
                    new_s += s[:n1] + f' {dekor(word)} '
                    s = s[n2:]
                    break

            if word:
                break
        
        if word == '':
            new_s += s
            break

    return new_s

def stat_err(df):
    ng = 0
    nb = 0
    ne_g = []
    ne_b = []
    for t in df.itertuples():
        if t.good != '' and (t.good not in t.name.lower()):
            # print(f'GOOD: {t}\n')
            ng += 1
            ne_g.append(t.Index)
        if t.brand != '' and (t.brand not in t.name.lower()):
            # print(f'BRAND: {t}\n')
            nb += 1
            ne_b.append(t.Index)
        # print(t)
        # break
    return ng, nb, ne_g, ne_b




