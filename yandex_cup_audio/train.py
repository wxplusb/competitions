import numpy as np
import pandas as pd
import os
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from tqdm import tqdm
import random
import annoy
import gc
import torch.optim as optim

import math
# from resnet import ResNet
RANDOM_STATE = 34
N_CPU = os.cpu_count()
rng = np.random.default_rng(RANDOM_STATE)

dir_out = "out/"
os.makedirs(dir_out, exist_ok = True)

# Data Loader 

def train_val_split(dataset, val_size = 0.2): # Сплит по artistid
    artist_ids = dataset['artistid'].unique()
    train_artist_ids, val_artist_ids = train_test_split(artist_ids, test_size = val_size, random_state=RANDOM_STATE)
    trainset = dataset[dataset['artistid'].isin(train_artist_ids)].copy()
    valset = dataset[dataset['artistid'].isin(val_artist_ids)].copy()
    return trainset, valset

class FeaturesLoader: 
    def __init__(self, feats, meta_info, device='cpu', id_correction=1):
        self.feats = feats
        self.meta_info = meta_info
        self.trackid2path = meta_info.set_index('trackid')['archive_features_path'].to_dict()
        self.id_correction = id_correction
        self.device = device
        
    def load_batch(self, tracks_ids):
        if isinstance(tracks_ids, list):
            tracks_ids = np.array(tracks_ids)
        tracks_ids = tracks_ids - self.id_correction
        batch = self.feats[tracks_ids]
        return torch.as_tensor(batch, device=self.device)

class TrainLoader:
    def __init__(self, features_loader, batch_size = 256, features_size = (512,60)):
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.features_size = features_size
        self.artist_track_ids = self.features_loader.meta_info.groupby('artistid').agg(list)
        
    def _generate_pairs(self, track_ids):
        np.random.shuffle(track_ids)
        pairs = [track_ids[i-2:i] for i in range(2, len(track_ids)+1, 2)]
        return pairs
        
    def _get_pair_ids(self):
        artist_track_ids = self.artist_track_ids.copy()
        artist_track_pairs = artist_track_ids['trackid'].map(self._generate_pairs)
        for pair_ids in artist_track_pairs.explode().dropna():
            yield pair_ids
            
    def _get_batch(self, batch_ids):
        batch_ids = np.array(batch_ids).reshape(-1)
        batch_features = self.features_loader.load_batch(batch_ids)
        batch_features = batch_features.reshape(self.batch_size, 2, *self.features_size)
        return batch_features
        
    def __iter__(self):
        batch_ids = []
        for pair_ids in self._get_pair_ids():
            batch_ids.append(pair_ids)
            if len(batch_ids) == self.batch_size:
                batch = self._get_batch(batch_ids)
                yield batch
                batch_ids = []

class TestLoader:
    def __init__(self, features_loader, batch_size = 256, features_size = (512,60)):
        self.features_loader = features_loader
        self.batch_size = batch_size
        self.features_size = features_size
        
    def __iter__(self):
        batch_ids = []
        for track_id in tqdm(self.features_loader.meta_info['trackid'].values):
            batch_ids.append(track_id)
            if len(batch_ids) == self.batch_size:
                yield batch_ids, self.features_loader.load_batch(batch_ids) 
                batch_ids = []
        if len(batch_ids) > 0:
            yield batch_ids, self.features_loader.load_batch(batch_ids) 

# Loss & Metrics

class NT_Xent(nn.Module):
    def __init__(self, temperature):
        super(NT_Xent, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
 
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        with torch.no_grad():
            top1_negative_samples, _ = negative_samples.topk(1)
            avg_rank = logits.argsort(descending=True).argmin(dim=1).float().mean().cpu().numpy()

        return loss, avg_rank

def get_ranked_list(embeds, top_size, annoy_num_trees = 32):
    annoy_index = None
    annoy2id = []
    id2annoy = dict()
    for track_id, track_embed in embeds.items():
        id2annoy[track_id] = len(annoy2id)
        annoy2id.append(track_id)
        if annoy_index is None:
            annoy_index = annoy.AnnoyIndex(len(track_embed), 'angular')
        annoy_index.add_item(id2annoy[track_id], track_embed)
    annoy_index.build(annoy_num_trees)
    ranked_list = dict()
    for track_id in embeds.keys():
        candidates = annoy_index.get_nns_by_item(id2annoy[track_id], top_size+1)[1:] # exclude trackid itself
        candidates = list(filter(lambda x: x != id2annoy[track_id], candidates))
        ranked_list[track_id] = [annoy2id[candidate] for candidate in candidates]
    return ranked_list

def position_discounter(position):
    return 1.0 / np.log2(position+1)   

def get_ideal_dcg(relevant_items_count, top_size):
    dcg = 0.0
    for result_indx in range(min(top_size, relevant_items_count)):
        position = result_indx + 1
        dcg += position_discounter(position)
    return dcg

def compute_dcg(query_trackid, ranked_list, track2artist_map, top_size):
    query_artistid = track2artist_map[query_trackid]
    dcg = 0.0
    for result_indx, result_trackid in enumerate(ranked_list[:top_size]):
        assert result_trackid != query_trackid
        position = result_indx + 1
        discounted_position = position_discounter(position)
        result_artistid = track2artist_map[result_trackid]
        if result_artistid == query_artistid:
            dcg += discounted_position
    return dcg

def eval_submission(submission, gt_meta_info, top_size = 100):
    track2artist_map = gt_meta_info.set_index('trackid')['artistid'].to_dict()
    artist2tracks_map = gt_meta_info.groupby('artistid').agg(list)['trackid'].to_dict()
    ndcg_list = []
    for query_trackid in tqdm(submission.keys()):
        ranked_list = submission[query_trackid]
        query_artistid = track2artist_map[query_trackid]
        query_artist_tracks_count = len(artist2tracks_map[query_artistid])
        ideal_dcg = get_ideal_dcg(query_artist_tracks_count-1, top_size=top_size)
        dcg = compute_dcg(query_trackid, ranked_list, track2artist_map, top_size=top_size)
        try:
            ndcg_list.append(dcg/ideal_dcg)
        except ZeroDivisionError:
            continue
    return np.mean(ndcg_list)

# class BasicNet(nn.Module):
#     def __init__(self, output_features_size):
#         super().__init__()
#         self.output_features_size = output_features_size *2
#         self.gru = nn.GRU(
#             512, 256, 2, batch_first=True, dropout=0.05
#         )
#         self.cl1 = nn.Linear(256, 256)

#         self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)
#         self.conv_2 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         self.mp_1 = nn.MaxPool1d(2, 2)
#         self.conv_3 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         self.conv_4 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)

#     def forward(self, x):
#         x1 = x.permute(0,2,1)
#         out,_ = self.gru(x1)
#         x1 = F.relu(self.cl1(out[:,-1]))

#         x2 = F.relu(self.conv_1(x))
#         x2 = F.relu(self.conv_2(x2))
#         x2 = self.mp_1(x2)

#         x3 = F.relu(self.conv_3(x2))
#         x3 = self.conv_4(x3).mean(axis = 2)

#         x4 = torch.cat([x1,x3],axis=-1)

#         return x4

# 294
# class BasicNet(nn.Module):
#     def __init__(self, output_features_size):
#         super().__init__()
#         self.output_features_size = output_features_size *2
#         self.gru = nn.GRU(
#             512, 256, 3, batch_first=True, dropout=0.05
#         )
#         # self.cl1 = nn.Linear(256, 256)

#         self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)
#         self.conv_2 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         self.mp_1 = nn.MaxPool1d(2, 2)
#         self.conv_3 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         self.conv_4 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)

#     def forward(self, x):
#         x1 = x.permute(0,2,1)
#         out,_ = self.gru(x1)
#         # x1 = F.relu(self.cl1(out[:,-1]))
#         x1 = out.permute(0,2,1).mean(axis = 2)

#         x2 = F.relu(self.conv_1(x))
#         x2 = F.relu(self.conv_2(x2))
#         x2 = self.mp_1(x2)

#         x3 = F.relu(self.conv_3(x2))
#         x3 = self.conv_4(x3).mean(axis = 2)

#         x4 = torch.cat([x1,x3],axis=-1)

#         return x4

# class BasicNet(nn.Module):
#     def __init__(self, output_features_size):
#         super().__init__()
#         self.output_features_size = output_features_size
#         self.gru_1 = nn.GRU(
#             512, 256, 1, batch_first=True, dropout=0.05
#         )
#         self.gru_2 = nn.GRU(
#             256, 256, 1, batch_first=True, dropout=0.05
#         )
#         self.gru_3 = nn.GRU(
#             256, 256, 1, batch_first=True, dropout=0.05
#         )
#         self.gru_4 = nn.GRU(
#             256, 256, 1, batch_first=True, dropout=0.05
#         )
#         # self.cl1 = nn.Linear(256, 256)

#         # self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)
#         # self.conv_2 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         # self.mp_1 = nn.MaxPool1d(2, 2)
#         # self.conv_3 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         # self.conv_4 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = x.permute(0,2,1)

#         out,_ = self.gru_1(x)
#         x = F.relu(out)
#         out,_ = self.gru_2(x)
#         x = F.relu(out)

#         out,_ = self.gru_3(x)
#         x = F.relu(out)
#         out,_ = self.gru_4(x)

#         # x1 = F.relu(self.cl1(out[:,-1]))
#         x = out.permute(0,2,1).mean(axis = 2)

#         # x2 = F.relu(self.conv_1(x))
#         # x2 = F.relu(self.conv_2(x2))
#         # x2 = self.mp_1(x2)

#         # x3 = F.relu(self.conv_3(x2))
#         # x3 = self.conv_4(x3).mean(axis = 2)

#         # x4 = torch.cat([x1,x3],axis=-1)

#         return x

# class BasicNet(nn.Module):
#     def __init__(self, output_features_size):
#         super().__init__()
#         self.output_features_size = 64
#         self.gru_1 = nn.LSTM(
#             512, 256, 1, batch_first=True, dropout=0.1
#         )
#         self.gru_2 = nn.LSTM(
#             256, 128, 1, batch_first=True, dropout=0.1
#         )
#         self.gru_3 = nn.LSTM(
#             128, 64, 1, batch_first=True, dropout=0.1
#         )
#         self.gru_4 = nn.LSTM(
#             64, 64, 1, batch_first=True, dropout=0.1
#         )
#         # self.cl1 = nn.Linear(256, 256)

#         # self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)
#         # self.conv_2 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         # self.mp_1 = nn.MaxPool1d(2, 2)
#         # self.conv_3 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)
#         # self.conv_4 = nn.Conv1d(output_features_size, output_features_size, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = x.permute(0,2,1)

#         out,_ = self.gru_1(x)
#         x = F.relu(out)
#         out,_ = self.gru_2(x)
#         x = F.relu(out)

#         out,_ = self.gru_3(x)
#         x = F.relu(out)
#         out,_ = self.gru_4(x)

#         # x1 = F.relu(self.cl1(out[:,-1]))
#         x = out.permute(0,2,1).mean(axis = 2)

#         # x2 = F.relu(self.conv_1(x))
#         # x2 = F.relu(self.conv_2(x2))
#         # x2 = self.mp_1(x2)

#         # x3 = F.relu(self.conv_3(x2))
#         # x3 = self.conv_4(x3).mean(axis = 2)

#         # x4 = torch.cat([x1,x3],axis=-1)

#         return x

# class BasicNet(nn.Module):
#     def __init__(self, output_features_size):
#         super().__init__()
#         self.output_features_size = 256
#         self.gru_1 = nn.LSTM(
#             512, 256, 2, batch_first=True, dropout=0.1
#         )
#         self.gru_3 = nn.LSTM(
#             256, 256, 2, batch_first=True, dropout=0.1
#         )

#     def forward(self, x):
#         x = x.permute(0,2,1)

#         x,_ = self.gru_1(x)
#         x,_ = self.gru_3(x)

#         x = x.permute(0,2,1).mean(axis = 2)

#         return x


from omni import torch_OS_CNN

class BasicNet(nn.Module):
    def __init__(self, output_features_size):
        super().__init__()
        self.output_features_size = 512
        # self.omni = ResNet(60,256)
        self.omni = torch_OS_CNN
        self.gru1 = nn.GRU(
            512, 512, 2, batch_first=True, dropout=0.05
        )

        self.gru2 = nn.GRU(
            256, 512, 1, batch_first=True, dropout=0.05
        )

        self.conv_1 = nn.Conv1d(512, output_features_size, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.omni(x) # 512-512

        x2 = x.permute(0,2,1)
        out,_ = self.gru1(x2)
        x2 = out[:,-1] # 512-512

        x3 = F.relu(self.conv_1(x))
        x3 = x3.permute(0,2,1)
        out,_ = self.gru2(x3)
        x3 = out[:,-1] # 512-512

        # x2 = out.permute(0,2,1).mean(axis = 2)

        x = x1 + x2 + x3

        # x = torch.cat([x1,x2],axis=-1)

        return x




class SimCLR(nn.Module):
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.n_features = encoder.output_features_size
        self.projection_dim = projection_dim
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.projection_dim, bias=False),
        )
        
    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

def inference(model, loader):
    embeds = dict()
    for tracks_ids, tracks_features in loader:
        with torch.no_grad():
            tracks_embeds = model(tracks_features)
            for track_id, track_embed in zip(tracks_ids, tracks_embeds):
                embeds[track_id] = track_embed.cpu().numpy()
    return embeds

def train(module, train_loader, val_loader, valset_meta, optimizer,scheduler, criterion, num_epochs, checkpoint_path, top_size = 100):

    max_ndcg = None
    max_tol = 5
    tol = 0

    for epoch in range(num_epochs):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            module.train()
            x_i, x_j = batch[:, 0, :, :], batch[:, 1, :, :]
            h_i, h_j, z_i, z_j = module(x_i, x_j)
            loss, avg_rank = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{num_epochs}. loss: {round(loss.item(),2)}. avg_rank: {round(avg_rank.item())}")
        
        with torch.no_grad():
            model_encoder = module.encoder
            embeds_encoder = inference(model_encoder, val_loader)
            ranked_list_encoder = get_ranked_list(embeds_encoder, top_size)
            val_ndcg_encoder = eval_submission(ranked_list_encoder, valset_meta)
            
            model_projector = nn.Sequential(module.encoder, module.projector)
            embeds_projector = inference(model_projector, val_loader)
            ranked_list_projector = get_ranked_list(embeds_projector, top_size)
            val_ndcg_projector = eval_submission(ranked_list_projector, valset_meta)
            
            print(f"Validation nDCG on epoch {epoch}. Encoder: {round(val_ndcg_encoder,3)}. Projector: {round(val_ndcg_projector,3)}")
            
            if (max_ndcg is None) or (val_ndcg_encoder > max_ndcg):
                max_ndcg = val_ndcg_encoder
                torch.save(module.state_dict(), checkpoint_path)
                tol = 0
            else:
                tol += 1
                if tol >= max_tol:
                    break  
        scheduler.step(val_ndcg_encoder)

def save_submission(submission, submission_path):
    with open(submission_path, 'w') as f:
        for query_trackid, result in submission.items():
            f.write("{}\t{}\n".format(query_trackid, " ".join(map(str, result))))

def main():
    parser = ArgumentParser(description='Simple naive baseline')
    parser.add_argument('--base-dir', dest='base_dir', action='store', required=False)
    args = parser.parse_args()


    # Seed
    seed = RANDOM_STATE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    TRAINSET_FILE = 'train_6.npy'
    TESTSET_FILE = 'test_6.npy'
    TRAINSET_META_FILENAME = 'train_meta.tsv'
    TESTSET_META_FILENAME = 'test_meta.tsv'
    SUBMISSION_FILENAME = 'submission.txt'
    MODEL_FILENAME = 'model.pt'
    CHECKPOINT_FILENAME = 'best.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    BATCH_SIZE = 512
    N_CHANNELS = 256
    PROJECTION_DIM = 256
    NUM_EPOCHS = 50
    LR = 1e-3
    TEMPERATURE = 0.1

    args.base_dir = 'data/yan2022'
    CHECKPOINT_PATH = dir_out + 'best_model.pt'

    TRAINSET_PATH = os.path.join(args.base_dir, TRAINSET_FILE)
    TESTSET_PATH = os.path.join(args.base_dir, TESTSET_FILE)
    TRAINSET_META_PATH = os.path.join(args.base_dir, TRAINSET_META_FILENAME)
    TESTSET_META_PATH = os.path.join(args.base_dir, TESTSET_META_FILENAME)
    SUBMISSION_PATH = os.path.join(args.base_dir, SUBMISSION_FILENAME)
    MODEL_PATH = os.path.join(args.base_dir, MODEL_FILENAME)

    

    sim_clr = SimCLR(
        encoder = BasicNet(N_CHANNELS),
        projection_dim = PROJECTION_DIM
    ).to(device)
    
    train_meta_info = pd.read_csv(TRAINSET_META_PATH, sep='\t')
    test_meta_info = pd.read_csv(TESTSET_META_PATH, sep='\t')
    train_meta_info, validation_meta_info = train_val_split(train_meta_info, val_size=0.1)

    # print("Loaded data")
    print("Train set size: {}".format(len(train_meta_info)))
    print("Validation set size: {}".format(len(validation_meta_info)))
    print("Test set size: {}".format(len(test_meta_info)))
    # print()

    train_feats = np.load(TRAINSET_PATH, mmap_mode='r')

    optimizer = torch.optim.Adam(sim_clr.parameters(), lr = LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=2,threshold=0.0005, min_lr=0.0005)

    print("Train")
    train(
        module = sim_clr,
        train_loader = TrainLoader(FeaturesLoader(train_feats, train_meta_info, device, id_correction=1), batch_size = BATCH_SIZE),
        val_loader = TestLoader(FeaturesLoader(train_feats, validation_meta_info, device, id_correction=1), batch_size = BATCH_SIZE),
        valset_meta = validation_meta_info,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = NT_Xent(temperature = TEMPERATURE),
        num_epochs = NUM_EPOCHS,
        checkpoint_path = CHECKPOINT_PATH
    )

    sim_clr.load_state_dict(torch.load(CHECKPOINT_PATH))

    del train_feats
    gc.collect()

    test_feats = np.load(TESTSET_PATH, mmap_mode='r')

    print("Submission encoder")
    test_loader = TestLoader(FeaturesLoader(test_feats, test_meta_info, device, id_correction=167197), batch_size = BATCH_SIZE)
    model = sim_clr.encoder
    embeds = inference(model, test_loader)
    submission = get_ranked_list(embeds, 100, annoy_num_trees=64)
    save_submission(submission, dir_out + 'sub.txt')

    # def save_pickle(file_name, data, verbose=False):
    #     import pickle
    #     if verbose:
    #         print('save: ', file_name)
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(data, f)

    # save_pickle('data/embeds', embeds)
    # return
    


if __name__ == '__main__':
    main()
