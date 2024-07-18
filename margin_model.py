from torch.nn import functional as F
import torch
import torch.nn as nn
import  math

class MarginModel(nn.Module):
    def __init__(self,embedding, numclass):
        super(MarginModel,self).__init__()
        self.embedding = embedding
        self.numclass = numclass
        self.weight = nn.Parameter(torch.Tensor(numclass,embedding))
        nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward (self, input ):
        input =  F.normalize(input)
        x = F.normalize(self.weight)
        logits=  F.linear(input ,x)
        logits = logits.clamp(-1,1)
        return logits


