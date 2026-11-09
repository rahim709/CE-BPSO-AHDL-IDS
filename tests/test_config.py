import json
from src.utils.core import read_config, save_json


def test_json_and_yaml_scientific_numbers_roundtrip(tmp_path):
    cfg = {'training': {'l1': 1e-5, 'l2': 1e-5}, 'label': '1e-5'}
    path = tmp_path / 'resolved.json'
    save_json(path, cfg)
    assert read_config(path) == cfg
    path = tmp_path / 'config.yaml'
    path.write_text("training: {l1: 1e-5, l2: 0.00001}\nlabel: '1e-5'\n")
    assert read_config(path) == cfg
