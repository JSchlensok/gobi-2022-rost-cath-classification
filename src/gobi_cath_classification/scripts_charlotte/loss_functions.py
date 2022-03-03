from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BinomialDevianceLoss(nn.Module):
    def __init__( self, alpha=2, beta=0.5, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.sim = F.cosine_similarity

    def forward( self, x, m, w, device):
        # computer similarity matrix
        s = Variable(self.sim(x, x), requires_grad=True).to(device)
        m = Variable(m, requires_grad=True).to(device)
        w = Variable(w, requires_grad=True).to(device)

        # calculate loss using the function defined in the paper
        loss = torch.mul(w, torch.log(1 + torch.exp(torch.mul(-self.alpha*(s-self.beta), m)))).sum()
        return loss


class HierarchicalLoss(nn.Module):
    def __init__( self, class_weights: List[float]):
        super().__init__()
        self.sim = F.cosine_similarity

    def forward( self, x, m, w, device):
        # computer similarity matrix
        s = Variable(self.sim(x, x), requires_grad=True).to(device)
        m = Variable(m, requires_grad=True).to(device)
        w = Variable(w, requires_grad=True).to(device)

        # calculate loss using the function defined in the paper
        loss = torch.mul(w, torch.log(1 + torch.exp(torch.mul(-self.alpha*(s-self.beta), m)))).sum()
        return loss
