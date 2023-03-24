import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader,Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image
from typing import Union, Any, Optional,Set, Tuple,List
import pytorch_lightning as pl
from transformers import AutoTokenizer

OUTPUT_SHAPE = (224, 224)

train_transform = T.Compose([
    T.RandomCrop(size=(490, 490)),
    T.Resize(OUTPUT_SHAPE),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

val_transform = T.Compose([
    T.Resize(OUTPUT_SHAPE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ImageDset(Dataset):
    def __init__(self, df:pd.DataFrame, dir_images:str='data/images/train',augmentation:Optional[T.Compose]=None):
        
        self.paths = df['product_id'].apply(lambda x:f'{dir_images}/{x}.jpg').reset_index(drop=True)

        if 'category_id' in df.columns:
            self.labels = torch.tensor(df['category_id'].to_numpy(), dtype=torch.int64)
        else:
            self.labels = torch.zeros(len(self.paths), dtype=torch.int64)

        self.aug = augmentation

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):

        img_path = self.paths.iloc[idx]
        image = Image.open(img_path)

        if self.aug:
            image = self.aug(image)

        return {
            "image": image,
            "label": self.labels[idx],
            }


class BertDset(Dataset):
    def __init__(self, df:pd.DataFrame, tokenizer, max_length:int=150):
        
        self.inputs = tokenizer(df['text'].to_list(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

        if 'category_id' in df.columns:
            self.labels = torch.tensor(df['category_id'].to_numpy(), dtype=torch.int64)
        else:
            self.labels = torch.zeros(len(df), dtype=torch.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "label": self.labels[idx],
            }
        if "token_type_ids" in self.inputs:
            item["token_type_ids"] = self.inputs["token_type_ids"][idx]
        return item

class HuseDset(Dataset):
    def __init__(self, df:pd.DataFrame, dir_images:str='data/images/train',augmentation:Optional[T.Compose]=None, tokenizer=None, max_length:int=150):
        
        # TEXT
        self.inputs = tokenizer(df['text'].to_list(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

        # IMAGES
        self.paths = df['product_id'].apply(lambda x:f'{dir_images}/{x}.jpg').reset_index(drop=True)
        self.aug = augmentation

        # CAT FEATS
        self.shops = torch.tensor(df['shop_id'].to_numpy(), dtype=torch.int64)

        if 'category_id' in df.columns:
            self.labels = torch.tensor(df['category_id'].to_numpy(), dtype=torch.int64)
        else:
            self.labels = torch.zeros(len(df), dtype=torch.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.paths.iloc[idx]
        image = Image.open(img_path)

        if self.aug:
            image = self.aug(image)

        item = {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "image": image,
            'shop_id': self.shops[idx],
            "label": self.labels[idx],
            }

        if "token_type_ids" in self.inputs:
            item["token_type_ids"] = self.inputs["token_type_ids"][idx]

        return item


class EmbedDataModule(pl.LightningDataModule):
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size, shuffle=True,num_workers=self.n_cpu)

    def get_val_dl(self, ds:Dataset) -> DataLoader:
        return DataLoader(ds, batch_size=self.val_bs, num_workers=self.n_cpu)

    def predict_dataloader(self):
        return self.get_val_dl(self.test)

    def val_dataloader(self):
        return self.get_val_dl(self.val)

    def emb_dataloaders(self):
        return {
        'train_val':(self.Xy[['product_id']],self.get_val_dl(self.train_val)),
        'test':(self.X_test[['product_id']],self.get_val_dl(self.test))
        }

class HuseDataModule(EmbedDataModule):
    def __init__(self, Xy:pd.DataFrame, X_test:pd.DataFrame, fold:int=0, batch_size:int=16, val_bs:int=16, n_cpu:int=8, cfg=None, train_img_dir:str='data/images/train',test_img_dir:str='data/images/test'):
        super().__init__()

        self.save_hyperparameters('batch_size','fold','cfg')

        self.fold = fold
        self.val_bs = val_bs
        self.Xy = Xy
        self.X_test = X_test
        self.n_cpu = n_cpu

        # for txt
        self.max_length = cfg.max_length
        self.tokenizer = cfg.model_name
        self.len_train = (self.Xy['fold']!=self.fold).sum()

        # for img
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        
    def setup(self,stage):
        if stage == 'fit':

            Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold]
            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold]

            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

            self.train = HuseDset(Xy_train, self.train_img_dir, train_transform,self.tokenizer, max_length=self.max_length)
            self.val = HuseDset(Xy_val, self.train_img_dir, val_transform,self.tokenizer, max_length=self.max_length)
            
            self.test = HuseDset(self.X_test, self.test_img_dir, val_transform,self.tokenizer, max_length=self.max_length)


class BertDataModule(EmbedDataModule):
    def __init__(self, Xy:pd.DataFrame, X_test:pd.DataFrame, fold:int=0, batch_size:int=16, val_bs:int=16, n_cpu:int=8, cfg=None):
        super().__init__()

        # https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html#hyperparameters-in-datamodules
        # если без аргументов то датасеты сохранит
        self.save_hyperparameters('batch_size','cfg')

        self.fold = fold
        self.val_bs = val_bs
        self.Xy = Xy
        self.X_test = X_test
        self.n_cpu = n_cpu
        self.max_length = cfg.max_length
        self.tokenizer = cfg.model_name

        self.len_train = (self.Xy['fold']!=self.fold).sum()
        
    # def prepare_data(self):
    # только скачиваем датасеты, эта ф-я только один раз исполняется в одном процессе
    #     # called only on 1 GPU
    #     download_dataset()
    #     tokenize()
    #     build_vocab()

    def setup(self,stage):
        if stage == 'fit':

            Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold]
            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold]


            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)

            self.train = BertDset(Xy_train,self.tokenizer, max_length=self.max_length)
            self.val = BertDset(Xy_val,self.tokenizer, max_length=self.max_length)

            self.train_val = BertDset(self.Xy,self.tokenizer, max_length=self.max_length)
            self.test = BertDset(self.X_test,self.tokenizer, max_length=self.max_length)


class ImageDataModule(EmbedDataModule):
    def __init__(self, Xy:pd.DataFrame, X_test:pd.DataFrame, fold:int=0, batch_size:int=128, val_bs:int=128, n_cpu:int=8, train_img_dir:str='data/images/train',test_img_dir:str='data/images/test'):
        super().__init__()

        self.save_hyperparameters('batch_size')

        self.fold = fold
        self.val_bs = val_bs
        self.Xy = Xy
        self.X_test = X_test
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir

        self.n_cpu = n_cpu


    def setup(self,stage):

        if stage == 'fit':

            Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold]
            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold]

            self.train = ImageDset(Xy_train, self.train_img_dir, train_transform)
            self.val = ImageDset(Xy_val,self.train_img_dir, val_transform)

            # embeds !!! val_transform
            self.train_val = ImageDset(self.Xy, self.train_img_dir, val_transform)
            self.test = ImageDset(self.X_test, self.test_img_dir, val_transform)

def save_all_embs(module:pl.LightningModule, dm:pl.LightningDataModule, prefix:str = 'img', dir_out:str = 'out/image/'):

    for part, (ids, dl) in dm.emb_dataloaders().items():
        embs = module.get_embs(dl)
        print(f'save {part}: {embs.shape}')
        df_emb = pd.DataFrame(embs)
        df_emb.columns = [prefix+str(col) for col  in df_emb.columns]
        df_emb = pd.concat([ids,df_emb], axis=1)
        df_emb.to_parquet(dir_out+f'{prefix}_{part}_embs_f{dm.fold}.pq',index=False)

class FTTDset(Dataset):
    def __init__(self, df:pd.DataFrame):
        self.df = df.iloc[:, 3:].to_numpy()

        if 'category_id' in df.columns:
            self.labels = torch.tensor(df['category_id'].to_numpy(), dtype=torch.int64)
        else:
            self.labels = torch.zeros(len(df), dtype=torch.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "num_feats": self.df[idx],
            "label": self.labels[idx],
            }
        return item

class FTTDataModule(EmbedDataModule):
    def __init__(self, Xy:pd.DataFrame, X_test:pd.DataFrame, fold:int=0, batch_size:int=128, val_bs:int=128, n_cpu:int=8, img_emb_dir:str='out/image_model/',bert_emb_dir:str='out/bert_model/'):
        super().__init__()
        self.save_hyperparameters('batch_size')

        self.fold = fold
        self.val_bs = val_bs
        self.Xy = Xy
        self.X_test = X_test
        self.img_emb_dir = img_emb_dir
        self.bert_emb_dir = bert_emb_dir

        self.n_cpu = n_cpu

    def setup(self,stage):
        if stage == 'fit':

            Xy_train = self.Xy.loc[self.Xy['fold']!=self.fold,['product_id','shop_id', 'category_id']]
            Xy_val = self.Xy.loc[self.Xy['fold']==self.fold,['product_id','shop_id', 'category_id']]

            bert_train_val_embs = pd.read_parquet(f'{self.bert_emb_dir}bert_train_val_embs_f{self.fold}.pq')
            img_train_val_embs = pd.read_parquet(f'{self.img_emb_dir}img_train_val_embs_f{self.fold}.pq')

            Xy_train = Xy_train.merge(bert_train_val_embs,on='product_id').merge(img_train_val_embs,on='product_id')

            Xy_val = Xy_val.merge(bert_train_val_embs,on='product_id').merge(img_train_val_embs,on='product_id')

            self.train = FTTDset(Xy_train)
            self.val = FTTDset(Xy_val)