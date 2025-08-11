import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import random, math

# openweb = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

# for rec in openweb:      
#     process(rec["text"])