import collections
import json
from collections import namedtuple
from scipy.misc import imread, imresize, imshow
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
import os
import sys
import socket

import pickle
import numpy as np
import _sqlite3 as sqlite3
from array import array
import redis

server_name = socket.gethostname()
python_version = sys.version[0]
isGPU = torch.cuda.is_available()

dataset_root = '/scratch/datasets/word2vec/'

print(isGPU)

if python_version == '3':
    import pickle as pkl
else:
    import cPickle as pkl
