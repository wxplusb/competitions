import torch
from torch import nn, Tensor
import torchvision
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Union, Any, Optional,Set, Tuple,List, Dict
import rtdl

class FTTModel(nn.Module):
    def __init__(self, name='resnet', n_num_features:int=2048, out_dim:int=874) -> None:
        super().__init__()

        self.name = name

        if self.name == 'mlp':
            self.base = rtdl.MLP.make_baseline(n_num_features, [out_dim, out_dim], 0.2, out_dim)
        elif self.name == 'ftt':
            self.base = rtdl.FTTransformer.make_default(
            n_num_features=n_num_features,
            cat_cardinalities=None,
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=out_dim)
        elif self.name == 'resnet':
            self.base = rtdl.ResNet.make_baseline(
            d_in=n_num_features,
            d_main=128,
            d_hidden=256,
            dropout_first=0.2,
            dropout_second=0.0,
            n_blocks=2,
            d_out=out_dim)

    def forward(self, batch:Dict[str,Tensor]):
        if self.name == 'ftt':
            return self.base(batch['num_feats'], x_cat=None)
        else:
            return self.base(batch['num_feats'])

class ImageModel(nn.Module):
    def __init__(self, hid_dim:int = 10, out_dim:int=874, freeze_first_layers:Optional[int]= None) -> None:
        super().__init__()

        # self.feats_extractor = torchvision.models.efficientnet_b0(pretrained=True)

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT 
        self.base = torchvision.models.efficientnet_b0(weights=weights)

        # https://discuss.pytorch.org/t/is-there-any-difference-between-calling-requires-grad-method-and-manually-set-requires-grad-attribute/122971/2
        if freeze_first_layers is not None:
            for name, param in self.base.features[:freeze_first_layers].named_parameters():
                param.requires_grad = False

        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True), 
            nn.Linear(in_features=1280, out_features=out_dim))

    def forward(self, batch:Dict[str,Tensor]):
        return self.base(batch['image'])

    def get_embs(self, batch:Dict[str,Tensor]):
        x = self.base.features(batch['image'])
        return self.base.avgpool(x).squeeze()


class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class BertModel(nn.Module):
    def __init__(self, cfg, out_dim:int=874) -> None:
        super().__init__()

        self.cfg = cfg

        self.model = AutoModel.from_pretrained(self.cfg.model_name)
        # self.config = AutoConfig.from_pretrained(self.cfg.model_name)

        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.pool = MeanPooling()
        self.drop = nn.Dropout(0.3)
        self.clf = nn.Linear(self.model.config.hidden_size, out_dim)

    def features(self, inputs):
        inputs = {k:v for k,v in inputs.items() if k != 'label'}
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = self.pool(last_hidden_states, inputs['attention_mask'])
        return feature

    def forward(self, inputs):
        feats = self.features(inputs)
        output = self.clf(self.drop(feats))
        return output

    def get_embs(self, inputs):
        return self.features(inputs)