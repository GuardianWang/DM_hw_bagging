import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.config import arguments
from model.model import BaseNet
from dataset.dataset import FlowerData

import matplotlib.pyplot as plt

# model = BaseNet(num_class=2)
# model = model.half()
# stat_dict = torch.load('./checkpoint/base_epoch%.4d.pth' % 100)
# model.load_state_dict(stat_dict, strict=False)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

ax1.plot([1, 2], [3, 4])
ax2.plot([3, 4], [1, 2])

# fig1.show()
# fig2.show()

plt.show()

print()
