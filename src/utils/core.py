"""Configuration, atomic artifacts and reproducibility helpers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any
import numpy as np
import torch
import yaml


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=_json, allow_nan=False) + '\n')
    os.replace(tmp, path)


def _json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read_config(path: str | Path) -> dict:
    path = Path(path)
    class ConfigLoader(yaml.SafeLoader):
        pass
    ConfigLoader.add_implicit_resolver('tag:yaml.org,2002:float',
        re.compile(r'^[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)[eE][-+]?[0-9]+$'), list('-+0123456789.'))
    config = json.loads(path.read_text()) if path.suffix == '.json' else yaml.load(path.read_text(), Loader=ConfigLoader)
    if not isinstance(config, dict):
        raise ValueError('Experiment config must be a mapping')
    if 'extends' in config:
        base = read_config(path.parent / config.pop('extends'))
        config = merge(base, config)
    return config


def merge(base: dict, overrides: dict) -> dict:
    result = dict(base)
    for key, val in overrides.items():
        result[key] = merge(result[key], val) if isinstance(val, dict) and isinstance(result.get(key), dict) else val
    return result


def seed_all(seed: int, threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def environment() -> dict:
    import platform
    import importlib.metadata as md
    return {'python': platform.python_version(), 'packages': {p: md.version(p) for p in
            ['torch', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'dehb']},
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
