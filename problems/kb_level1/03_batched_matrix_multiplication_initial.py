import torch
import torch.nn.functional
import math

def run(A, B):
    return torch.bmm(A, B)