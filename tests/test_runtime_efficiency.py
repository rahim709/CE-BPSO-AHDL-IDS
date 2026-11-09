import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.training.runtime import vector_loader, clear_caches
from src.training.engine import train_to


def test_vectorized_sample_order():
    x = torch.arange(93).reshape(31, 3)
    y = torch.arange(31)
    a, b = torch.Generator().manual_seed(42), torch.Generator().manual_seed(42)
    legacy = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=a)
    fast = vector_loader(x, y, 8, b)
    for _ in range(3):
        assert all(torch.equal(u[0], v[0]) and torch.equal(u[1], v[1]) for u, v in zip(legacy, fast))
        assert torch.equal(a.get_state(), b.get_state())


def test_fast_exact_weights_and_resume(fixture_data, architecture, tmp_path):
    data, _ = fixture_data
    (data / 'test.npz').unlink()
    cfg = {'device': os.environ.get('BPSO_TEST_DEVICE', 'cpu'), 'threads': 1, 'class_weighting': 'balanced'}
    runtime = {'vectorized': True, 'resident': True, 'cache_data': True, 'cache_model': True,
               'defer_scalar_sync': True, 'validation_epochs': [1, 3]}
    clear_caches()
    train_to(architecture, data, tmp_path / 'legacy', 3, 42, cfg)
    train_to(architecture, data, tmp_path / 'fast', 3, 42, cfg, runtime=runtime)
    train_to(architecture, data, tmp_path / 'resume', 1, 42, cfg, runtime=runtime)
    train_to(architecture, data, tmp_path / 'resume', 3, 42, cfg, runtime=runtime)
    states = [torch.load(tmp_path / name / 'last.pt', map_location='cpu', weights_only=False) for name in ['legacy', 'fast', 'resume']]
    assert all(torch.equal(states[0]['model'][k], s['model'][k]) for s in states for k in states[0]['model'])
    assert states[1]['rows'][1]['macro_f1'] is None
    assert states[1]['rows'][-1]['macro_f1'] == states[0]['rows'][-1]['macro_f1']
    clear_caches()
