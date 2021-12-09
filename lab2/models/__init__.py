"""Init file"""
import numpy as np

from .bow import BOW
from .cbow import CBOW, DeepCBOW, PTDeepCBOW
from .lstm import LSTMClassifier
from .tree_lstm import TreeLSTMClassifier


# Here we print each parameter name, shape, and if it is trainable.
def print_parameters(model):
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
    print("\nTotal number of parameters: {}\n".format(total))
