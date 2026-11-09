"""Opt-in exact-FP32 execution helpers; bounded caches never share model trials."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, BatchSampler, RandomSampler

ARRAYS = OrderedDict()
TENSORS = OrderedDict()
SESSIONS = OrderedDict()


def clear_caches():
    ARRAYS.clear()
    TENSORS.clear()
    SESSIONS.clear()


def load_arrays(directory, split):
    path = Path(directory) / f'{split}.npz'
    info = path.stat()
    key = (str(path.resolve()), info.st_mtime_ns, info.st_size)
    if key not in ARRAYS:
        with np.load(path) as data:
            ARRAYS[key] = (data['x'], data['y'])
        while len(ARRAYS) > 2:
            ARRAYS.popitem(last=False)
    return ARRAYS[key]


def tensors(x, y, key, device):
    cache_key = (key, str(device))
    if cache_key not in TENSORS:
        TENSORS[cache_key] = (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device))
        while len(TENSORS) > 2:
            TENSORS.popitem(last=False)
    return TENSORS[cache_key]


class BatchDataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, indices):
        return self.x[indices], self.y[indices]


def vector_loader(x, y, batch_size, generator, pin_memory=False):
    dataset = BatchDataset(x, y)
    sampler = BatchSampler(RandomSampler(dataset, generator=generator), batch_size, drop_last=False)
    # The loader and RandomSampler consume the same generator calls as the
    # original automatically batched loader, preserving sample order and RNG.
    return DataLoader(dataset, batch_size=None, sampler=sampler, generator=generator,
                      num_workers=0, pin_memory=pin_memory)
