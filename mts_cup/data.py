import numpy as np
import pandas as pd
from typing import Optional

class BucketBatchSampler:
    def __init__(self, df:pd.DataFrame, batch_size:int=3, seed:int=34) -> None:
        
        self.bucket_list = df.sort_values('user_id').groupby('len_bucket')['user_id'].agg(list).to_list()
        self.bucket_list = [np.array(_) for _ in self.bucket_list]

        self.batch_size = batch_size

        self.rng = np.random.default_rng(seed) 

        self._len = len(df)//self.batch_size + len(self.bucket_list)

    def new_epoch(self) -> None:
        # print('new_epoch')
        self.rng.shuffle(self.bucket_list)
        [self.rng.shuffle(self.bucket_list[i]) for i in range(len(self.bucket_list))]
        
    def __iter__(self) -> np.ndarray:
        bs = self.batch_size
        self.new_epoch()

        for b in self.bucket_list:
            for i in range(len(b)//bs + 1):
                if i*bs >= len(b):
                    break
                yield b[i*bs:(i+1)*bs]
        
    def __len__(self) -> int:
        return self._len

from ptls.frames.coles import ColesSupervisedDataset
import torch

class myColesSupervisedDataset(ColesSupervisedDataset):
    def get_splits(self, feature_arrays):
        local_date = feature_arrays[self.col_time]
        indexes = self.splitter.split(local_date)
        return [{k: self.get_value(k, v, ix) for k, v in feature_arrays.items()} for ix in indexes]

    @staticmethod
    def get_value(k: str, x, ix):
        if k == 'event_time':
            return x[ix]
        if k.startswith('target'):
            return x
        if type(x) in (np.ndarray, torch.Tensor):
            return x[ix]
        return x