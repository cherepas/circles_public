import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class Decoder(torch.nn.Module):
    """simple decoder with two layers"""
    def __init__(self, nim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(nim*32, 1500)
    def forward(self, x):
        x = self.linear(x)
        return x
