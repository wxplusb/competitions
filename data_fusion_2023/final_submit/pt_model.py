import numpy as np
import pandas as pd
import pickle
import pytorch_lightning as pl
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, default_collate
import numpy as np
from tqdm.notebook import tqdm

from torchmetrics.classification import BinaryAUROC

embedding_projections = {
    "hour": (26, 12),
    "mcc_code": (403, 150),
    "currency_rk": (5, 3),
    "transaction_amt": (103, 50),
    "day": (9, 4),
    "month": (14, 6),
    "number_day": (33, 15),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransactionsRnn(pl.LightningModule):
    def __init__(self, rnn_units=128, top_classifier_units=64, lr=3e-4, out_features=2):
        super().__init__()
        self._transaction_cat_embeddings = nn.ModuleList(
            [
                self._create_embedding_projection(*embedding_projections[feature])
                for feature in embedding_projections.keys()
            ]
        )

        self._spatial_dropout = nn.Dropout2d(0.5)
        concat_embed = sum([embedding_projections[x][1] for x in embedding_projections.keys()])
        self._gru = nn.GRU(input_size=concat_embed, hidden_size=rnn_units, batch_first=True, bidirectional=True,)

        self._hidden_size = rnn_units

        pooling_result_dimension = self._hidden_size * 2
        self._top_classifier = nn.Sequential(
            nn.Linear(in_features=pooling_result_dimension * 3, out_features=top_classifier_units,),
            nn.ReLU(),
            nn.Linear(in_features=top_classifier_units, out_features=out_features),
        )

        self.loss = nn.functional.binary_cross_entropy
        self.learning_rate = lr

        self.val_metric = BinaryAUROC()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(lr=self.learning_rate, params=self.parameters())
        return opt

    def forward(self, transactions_cat_features, *, return_embedding=False):
        batch_size = transactions_cat_features.shape[0]

        last_hidden, states = self.get_emb(transactions_cat_features)
        probs = self.classify_emb(batch_size, last_hidden, states)

        if return_embedding:
            return probs, states
        else:
            return probs

    def classify_emb(self, batch_size, last_hidden, states):
        rnn_max_pool = states.max(dim=1)[0]
        rnn_avg_pool = states.sum(dim=1) / states.shape[1]
        last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, self._hidden_size * 2))
        combined = torch.cat([rnn_max_pool, rnn_avg_pool, last_hidden], dim=-1)
        drop = nn.functional.dropout(combined, p=0.5)
        logit = self._top_classifier(drop)
        probs = nn.functional.softmax(logit, dim=1)
        return probs

    def get_emb(self, transactions_cat_features):
        embeddings = [
            embedding(transactions_cat_features[:, i]) for i, embedding in enumerate(self._transaction_cat_embeddings)
        ]
        concated_embeddings = torch.cat(embeddings, dim=-1).permute(0, 2, 1).unsqueeze(3)
        dropout_embeddings = self._spatial_dropout(concated_embeddings)
        dropout_embeddings = dropout_embeddings.squeeze(3).permute(0, 2, 1)
        states, last_hidden = self._gru(dropout_embeddings)
        return last_hidden, states

    def training_step(self, train_batch, batch_idx):
        x, y_label = train_batch
        batch_size = len(y_label)
        y_pred = self.forward(x)

        y_true = torch.zeros(batch_size, 2)
        y_true[range(y_true.shape[0]), y_label.long()] = 1
        y_true_dev = y_true.to(self.device)

        batch_loss = self.loss(y_pred, y_true_dev)


        self.log("train_loss", batch_loss, on_step=True, on_epoch=False)

        # self.log("loss", batch_loss)
        # self.log("train_ROC-AUC", rocauc)
        # self.log("train_Accuracy", accuracy)

        return {
            "loss": batch_loss,
            # "log": {"train_ROC-AUC": rocauc, "train_Accuracy": accuracy},
        }

    def validation_step(self, val_batch, batch_idx):
        x, y_label = val_batch
        batch_size = len(y_label)
        y_hat = self.forward(x)

        y_true = torch.zeros(batch_size, 2)
        y_true[range(y_true.shape[0]), y_label.long()] = 1
        y_true_dev = y_true.to(self.device)

        val_loss = self.loss(y_hat, y_true_dev)
        self.val_metric(y_hat.detach()[:,1], y_label)

        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_roc_auc',self.val_metric, on_step=False, on_epoch=True, prog_bar=True)
        

    @classmethod
    def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0):
        add_missing = 1 if add_missing else 0
        return nn.Embedding(
            num_embeddings=cardinality + add_missing, embedding_dim=embed_size, padding_idx=padding_idx,
        )


class TransactionsDataset(Dataset):
    def __init__(self, data, dftarget=None):
        if dftarget is not None:
            d = data.merge(dftarget, on="user_id")
            self.labels = d.target.copy().values
            self.features = d.sequences.copy().values
            del d
        else:
            self.labels = data.user_id.copy().values
            self.features = data.sequences.copy().values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x, y = self.features[idx], self.labels[idx]
        return x, y


def process_for_nn(transactions_frame, features, nn_bins, *, need_padding=True, num_transactions = 300):

    if not need_padding:
        num_transactions = 1
    return (
        transactions_frame.groupby(["user_id"])[features]
        # take last 300 transactions
        .apply(lambda x: x.values.transpose()[:, -num_transactions:].tolist())
        # additional padding to 300
        .apply(lambda x: np.array([list(i) + [0] * int(num_transactions - len(x[0])) for i in x]))
        .reset_index()
        .rename(columns={0: "sequences"})
    )



def get_dataloader(dataset, device, batch_size=128, is_validation=False):
    def collate_loader(x):
        return tuple(x_.to(device) for x_ in default_collate(x))

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_loader, shuffle=not is_validation,)


def read_trans(path, sort=False, add_pos=True):
    df = pd.read_csv(path,parse_dates=["transaction_dttm"], dtype={"user_id": int, "mcc_code": int, "currency_rk": int, "transaction_amt": float})

    if sort:
        df = df.sort_values(['user_id',"transaction_dttm"]).reset_index(drop=True)

    df = df.dropna().assign(
            hour=lambda x: x.transaction_dttm.dt.hour,
            day=lambda x: x.transaction_dttm.dt.dayofweek,
            month=lambda x: x.transaction_dttm.dt.month,
            number_day=lambda x: x.transaction_dttm.dt.day)

    if add_pos:
        df['pos'] = df.groupby('user_id').cumcount()

    return df


def pos_truncate(df, min_pos, max_pos):
    df['pos'] = df.groupby('user_id').cumcount()
    df = df[(df['pos']>min_pos) & (df['pos']<=max_pos)].reset_index(drop=True).copy()
    return df

def pt_predict(source_file, bins_path, model_path, random_seed=34):
    
    pl.seed_everything(random_seed)
    df = read_trans(source_file, sort=False, add_pos=True)

    df = pos_truncate(df, 15, 285)
    
    with open(bins_path, "rb") as f:
        bins = pickle.load(f)

    features = bins.pop("features")

    def digitize_cat(df, nn_bins):
        for dense_col in nn_bins.keys():
            if dense_col == "transaction_amt":
                df[dense_col] = pd.cut(df[dense_col], bins=nn_bins[dense_col], labels=False).astype(int)
            else:
                df[dense_col] = pd.cut(
                    df[dense_col].astype(float).astype(int), bins=nn_bins[dense_col], labels=False,
                ).astype(int)

        return df

    df = digitize_cat(df, bins)

    models = []
    for fold in range(5):
        model = TransactionsRnn.load_from_checkpoint(f'best_fold_{fold}.ckpt')
        model.to(device)
        model.eval()
        models.append(model)

    REPETITIONS = 30
    rng = np.random.default_rng(random_seed)

    results = []
    for i in range(REPETITIONS):
        df_sample = df.copy()

        df_sample = df_sample.groupby('user_id').sample(n=180, random_state=180+i, replace=True)

        df_sample = df_sample.groupby(["user_id"])[features].apply(lambda x: x.values.transpose().tolist()).apply(lambda x: np.array(x,dtype=np.int32)).reset_index().rename(columns={0: "sequences"})

        dataset = TransactionsDataset(df_sample)
        dataloader = get_dataloader(dataset, device, batch_size=256, is_validation=True)
        preds = []
        users = []
        for data, target in dataloader:
            y_pred = np.zeros(len(target))
            for model in models:
                y_pred = y_pred + model(data).detach().cpu().numpy()[:,1]
            y_pred = y_pred/len(models)

            preds.append(y_pred)
            users.append(target.detach().cpu().numpy())
        preds = np.concatenate(preds)
        users = np.concatenate(users)
        results.append(pd.DataFrame({"user_id": users, "target": preds}))

    results[0]['target'] = np.mean([x.target.values for x in results], axis=0)
    return results[0]


def main():
    source_file, output_path = sys.argv[1:]
    bins_path="nn_bins.pickle"
    model_path="nn_weights_1.ckpt"
    result = reliable_predict(source_file, bins_path, model_path)
    result.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
