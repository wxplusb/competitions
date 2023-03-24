import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Union, Any, Optional,Set, Tuple,List, Dict

class CrossModalGapLoss(nn.Module):  
    def __init__(self): super().__init__()
        
    def forward(self, img_embs:Tensor, text_embs:Tensor):
        return torch.mean(1 - F.cosine_similarity(img_embs, text_embs))

class SemanticSimilarityLoss(nn.Module):  
    def __init__(self, Aij:Tensor, margin:float=0.7): 
        super().__init__()
        self.margin = margin
        self.Aij = Aij
        
    def forward(self, all_embs:Tensor, labels:Tensor):
        N = all_embs.shape[0] # batch size
        
        embs_m = all_embs.repeat_interleave(N, -2)
        embs_n = all_embs.repeat(N,1)
        pairwise_embedding_distance = 1 - F.cosine_similarity(embs_m, embs_n) # d (U_m , U_n )
        
        L_m = labels.repeat_interleave(N)
        L_n = labels.repeat(N)
        pairwise_class_distance = self.Aij[[L_m, L_n]]

        sigma = self._calc_sigma(pairwise_class_distance, pairwise_embedding_distance)
        # loss = Σ ( σ * (d(U_m,U_n) - Aij)² ) / N²
        loss = (sigma * (pairwise_embedding_distance - pairwise_class_distance).pow(2)).mean() 
        return loss
    
    def _calc_sigma(self, pairwise_class_distance, pairwise_embedding_distance):
        # calculate sigma as described in eq(10) in the HUSE paper.
        sigma = ( pairwise_class_distance < self.margin ) & ( pairwise_embedding_distance < self.margin )
        return sigma.to(torch.float32)

class BigLoss(nn.Module):
    
    def __init__(self, Aij, cfg):
        super().__init__()
        self.cfg = cfg

        self.Loss1 = nn.CrossEntropyLoss()
        self.Loss2 = SemanticSimilarityLoss(Aij, self.cfg.margin_semantic)
        self.Loss3 = CrossModalGapLoss()

    def forward(self, y_hat, all_embs, img_embs, text_embs, labels):
        loss1 = self.cfg.weight_class_L * self.Loss1(y_hat, labels)
        loss2 = self.cfg.weight_semantic_L  * self.Loss2(all_embs, labels)
        loss3 = self.cfg.weight_gap_L * self.Loss3(img_embs, text_embs)
        loss = loss1 + loss2 + loss3
        return loss, (loss1.item(),loss2.item(),loss3.item())