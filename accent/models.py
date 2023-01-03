import torch
from torch import nn
import torch.nn.functional as F
from my import LETTERS, get_letter_feats

class Gru_Pack_final(nn.Module):
	def __init__(self, input_dim:int=len(LETTERS)+1, emb_dim:int=128, hid_dim:int=128, n_layers:int=1, dropout:float=0.2, num_classes:int=6):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		letter_feats = torch.tensor(get_letter_feats(),dtype=torch.float32)

		self.feats_emb = nn.Embedding.from_pretrained(letter_feats,padding_idx=0)

		self.rnn1 = nn.GRU(input_size = emb_dim+letter_feats.shape[1], hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.rnn2 = nn.GRU(input_size = emb_dim+letter_feats.shape[1], hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.rnn3 = nn.GRU(input_size = hid_dim*4, hidden_size = hid_dim*4, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.fc1 = nn.Linear(8*hid_dim, 4*hid_dim)
		self.fc2 = nn.Linear(4*hid_dim, num_classes)
		
		self.dt1 = nn.Dropout(dropout)

		self.dt2 = nn.Dropout1d(p=0.1)
		self.dt3 = nn.Dropout1d(p=0.1)
	
	@staticmethod
	def pack_rnn(rnn, x, len_x, pad_len):
		x = nn.utils.rnn.pack_padded_sequence(x, len_x, batch_first=True,enforce_sorted=False)
		out, _ = rnn(x)
		out, seq_sizes = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=pad_len)

		return out, seq_sizes

	def forward(self, x): 
		x, lemma, len_x, len_lem = x['X'],x['lemma'],x['len_x'],x['len_lem']

		len_x_lem = torch.maximum(len_x, len_lem)
		max_len_x_lem = len_x_lem.max()

		x1 = self.dt2(self.emb(x))
		x2 = self.feats_emb(x)
		x = torch.cat([x1, x2],axis=2)

		out1,_ = self.pack_rnn(self.rnn1, x, len_x, max_len_x_lem)

		lemma1 = self.dt3(self.emb(lemma))
		lemma2 = self.feats_emb(lemma)
		lemma = torch.cat([lemma1, lemma2],axis=2)

		out2,_ = self.pack_rnn(self.rnn2, lemma, len_lem, max_len_x_lem)
		
		x = torch.cat([out1, out2],axis=2)
		out, seq_sizes = self.pack_rnn(self.rnn3, x, len_x_lem, max_len_x_lem)

		seq_idx = torch.arange(seq_sizes.size(0))
		x = out[seq_idx, seq_sizes-1]

		x = self.dt1(x)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

class Model_Gru_Transformer2(nn.Module):
	def __init__(self, input_dim:int=len(LETTERS)+1, emb_dim:int=64, hid_dim:int=64, n_layers:int=1, dropout:float=0.1, num_classes:int=6):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.pe1 =PositionalEncoding(d_model=64, dropout=0.1)
		self.pe2 =PositionalEncoding(d_model=64, dropout=0.1)

		self.rnn1 = nn.TransformerEncoderLayer(d_model=64, nhead=2, batch_first=True,dim_feedforward=256)
		self.rnn1 = nn.TransformerEncoder(self.rnn1, num_layers=2)

		self.rnn2 = nn.TransformerEncoderLayer(d_model=64, nhead=2, batch_first=True,dim_feedforward=256)
		self.rnn2 = nn.TransformerEncoder(self.rnn2, num_layers=2)

		self.rnn3 = nn.GRU(input_size = hid_dim*2, hidden_size = hid_dim*2, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.fc1 = nn.Linear(4*hid_dim, 4*hid_dim)
		self.fc2 = nn.Linear(4*hid_dim, num_classes)
		
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x): 
		x, lemma = x['X'],x['lemma']

		x = self.pe1(self.emb(x))
		out1 = self.rnn1(x)

		lemma = self.pe2(self.emb(lemma))
		out2 = self.rnn2(lemma)

		x = torch.cat([out1,out2],axis=2)

		out, _ = self.rnn3(x)
		x = out.mean(axis=1)

		# x = self.dropout(x)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x


class Gru1(nn.Module):
	def __init__(self, input_dim=len(LETTERS)+1, emb_dim=64, hid_dim=64, n_layers=1, dropout=0.05, num_classes=6):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.rnn1 = nn.GRU(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.rnn2 = nn.GRU(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.rnn3 = nn.GRU(input_size = hid_dim*4, hidden_size = hid_dim*4, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.fc1 = nn.Linear(8*hid_dim, 4*hid_dim)
		self.fc2 = nn.Linear(4*hid_dim, num_classes)
		
		self.dt1 = nn.Dropout(dropout)
		self.dt2 = nn.Dropout(dropout)
		
	def forward(self, x): 
		x, lemma = x['X'],x['lemma']

		x = self.emb(x)
		out1, _ = self.rnn1(x)

		lemma = self.emb(lemma)
		out2, _ = self.rnn2(lemma)

		x = torch.cat([out1,out2],axis=2)

		out, _ = self.rnn3(x)
		x = out.mean(axis=1)

		x = self.dt1(x)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x

class Gru2(nn.Module):
	def __init__(self, input_dim=len(LETTERS)+1, emb_dim=64, hid_dim=64, n_layers=1, dropout=0.05, num_classes=6):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.rnn3 = nn.GRU(input_size = hid_dim*2, hidden_size = hid_dim*2, num_layers = 2, batch_first=True, bidirectional=True,dropout=0.01)

		self.fc1 = nn.Linear(4*hid_dim, 4*hid_dim)
		self.fc2 = nn.Linear(4*hid_dim, num_classes)
		
		self.dt1 = nn.Dropout(dropout)
		self.dt2 = nn.Dropout(dropout)
		
	def forward(self, x): 
		x, lemma = x['X'],x['lemma']

		x = self.emb(x)

		lemma = self.emb(lemma)
		x = torch.cat([x,lemma],axis=2)

		out, _ = self.rnn3(x)
		x = out.mean(axis=1)

		x = self.dt1(x)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x


# 0.936
class Model_Gru_Lemma5(nn.Module):
	def __init__(self, input_dim=len(LETTERS)+1, emb_dim=128, hid_dim=128, n_layers=1, dropout=0.05, num_classes=6):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.rnn1 = nn.GRU(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.rnn2 = nn.GRU(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.rnn3 = nn.GRU(input_size = hid_dim*4, hidden_size = hid_dim*4, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.fc1 = nn.Linear(8*hid_dim, 4*hid_dim)
		self.fc2 = nn.Linear(4*hid_dim, num_classes)
		
		self.dt1 = nn.Dropout(dropout)
		self.dt2 = nn.Dropout(dropout)
		
	def forward(self, x): 
		x, lemma = x

		x = self.emb(x)
		out1, _ = self.rnn1(x)

		lemma = self.emb(lemma)
		out2, _ = self.rnn2(lemma)

		x = torch.cat([out1,out2],axis=2)

		out, _ = self.rnn3(x)
		x = out.mean(axis=1)

		x = self.dt1(x)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x


# ниже модели, которые не учитывают леммы
class Model_Gru(nn.Module):
	def __init__(self, input_dim=len(LETTERS)+1, emb_dim=64, hid_dim=64, n_layers=2, dropout=0.1, num_classes=6):
		super().__init__()
		
		self.hid_dim = hid_dim
		self.n_layers = n_layers

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.rnn = nn.GRU(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, batch_first=True, bidirectional=True,dropout=dropout)

		self.fc1 = nn.Linear(2*hid_dim, 2*hid_dim)
		self.fc2 = nn.Linear(2*hid_dim, num_classes)
		
		# self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):        
		x = self.emb(x)
		out, _ = self.rnn(x)
		# x = out[:,-1]
		x = out.mean(axis=1)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x


class Model_CNN(nn.Module):
	def __init__(self, input_dim=len(LETTERS)+1, emb_dim=64, hid_dim=128):
		super().__init__()
		
		self.hid_dim = hid_dim

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.conv_1 = nn.Conv1d(in_channels=emb_dim, out_channels = hid_dim, kernel_size=7, padding=3)
		self.conv_2 = nn.Conv1d(in_channels=hid_dim, out_channels = hid_dim, kernel_size=7, padding=3)

		self.fc1 = nn.Linear(hid_dim, hid_dim)
		self.fc2 = nn.Linear(hid_dim, 6)
		
		
	def forward(self, x):        
		x = self.emb(x)
		x = x.permute(0,2,1)
		x = F.relu(self.conv_1(x))
		x = F.relu(self.conv_2(x))
		x = x.mean(axis=2)

		x = F.relu(self.fc1(x))
		x = self.fc2(x)        
		return x


# attention

import math

RNNS = ['LSTM', 'GRU']

class Encoder(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, nlayers=1, dropout=0.,bidirectional=True, rnn_type='GRU'):
		super(Encoder, self).__init__()
		self.bidirectional = bidirectional
		assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
		rnn_cell = getattr(nn, rnn_type)
		self.rnn = rnn_cell(embedding_dim, hidden_dim, nlayers, 
							dropout=dropout, bidirectional=bidirectional)

	def forward(self, input, hidden=None):
		return self.rnn(input, hidden)


class Attention(nn.Module):
	def __init__(self, query_dim, key_dim, value_dim):
		super(Attention, self).__init__()
		self.scale = 1. / math.sqrt(query_dim)

	def forward(self, query, keys, values):
		# Query = [BxQ]
		# Keys = [TxBxK]
		# Values = [TxBxV]
		# Outputs = a:[TxB], lin_comb:[BxV]

		# Here we assume q_dim == k_dim (dot product attention)

		query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
		keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
		energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
		energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

		values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
		linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
		return energy, linear_combination

class Model_Rnn_with_attention(nn.Module):
	def __init__(self, input_dim=len(LETTERS)+1, emb_dim=64,hid_dim=64, n_layers=2, dropout=0.1, num_classes=6):
		super(Model_Rnn_with_attention, self).__init__()

		self.emb = nn.Embedding(num_embeddings=input_dim,embedding_dim=emb_dim, padding_idx=0)

		self.encoder = nn.GRU(input_size = emb_dim, hidden_size = hid_dim, num_layers = n_layers, bidirectional=True,dropout=dropout)

		self.attention = Attention(2*hid_dim,2*hid_dim,2*hid_dim)
		self.decoder = nn.Linear(2*hid_dim, num_classes)

		self.fc1 = nn.Linear(2*hid_dim, 2*hid_dim)
		self.fc2 = nn.Linear(2*hid_dim, 6)

	def forward(self, x):
		x = self.emb(x)
		x = x.permute(1,0,2)

		outputs, hidden = self.encoder(x)
		if isinstance(hidden, tuple): # LSTM
			hidden = hidden[1] # take the cell state

		# print('----',hidden.shape)
		hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
		# print('----',hidden.shape, outputs.shape)

		# if self.encoder.bidirectional: 
		#   hidden = torch.cat([hidden[-1], hidden[-2]], dim=1)
		# else:
		#   hidden = hidden[-1]

		energy, linear_combination = self.attention(hidden, outputs, outputs) 

		x = F.relu(self.fc1(linear_combination))
		x = self.fc2(x) 

		return x