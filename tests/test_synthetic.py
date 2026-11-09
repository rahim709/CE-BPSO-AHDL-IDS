"""Software fixtures must remain reproducible, segregated and explicitly synthetic."""
import os
import json
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import prepare
from src.data.synthetic import generate_datasets, SWAT_SENSORS
from src.training.engine import final_evaluate, load_partition
from src.utils.core import read_config


@pytest.mark.parametrize('name,classes', [('ton_iot', 10), ('bot_iot', 5), ('swat', 2)])
def test_synthetic_generation_and_profile_pipeline(tmp_path, architecture, name, classes):
    generated = generate_datasets(tmp_path / 'synthetic', rows=600, seed=31)
    directory = tmp_path / 'synthetic' / name
    manifest = json.loads((directory / 'SYNTHETIC_MANIFEST.json').read_text())
    assert manifest['synthetic'] and not manifest['official_data']
    assert sum(f['rows'] for f in manifest['files']) == 600
    cfg = read_config(f'configs/synthetic/{name}.yaml')
    cfg['data']['files'] = [str(directory / f['name']) for f in manifest['files'] if 'train' in f['name'] or 'sensor' in f['name']]
    cfg['data']['test_files'] = [str(directory / f['name']) for f in manifest['files'] if 'test' in f['name']] or None
    output = tmp_path / 'processed' / name
    metadata = prepare(cfg['data'], output)
    assert metadata['config']['synthetic'] is True
    assert len(metadata['classes']) == classes
    for part in ['train', 'val', 'test']:
        x, y = load_partition(output, part)
        assert np.isfinite(x).all()
        assert len(np.unique(y)) == classes
    assert not set(metadata['features']) & {cfg['data']['target'], 'label', 'attack', 'subcategory', 'Timestamp'}
    if name == 'ton_iot':
        assert len(pd.read_csv(directory / 'SYNTHETIC_train.csv').columns) == 44
    if name == 'swat':
        assert len(SWAT_SENSORS) == 51
        assert metadata['features'] == list(SWAT_SENSORS)
    result = final_evaluate(architecture, output, tmp_path / 'run' / name, 1, 7, {'device': os.environ.get('BPSO_TEST_DEVICE', 'cpu')})
    assert result['synthetic_data'] is True
    assert len(result['per_class']) == classes


def test_synthetic_seed_and_checksums_are_reproducible(tmp_path):
    generate_datasets(tmp_path / 'a', rows=500, seed=17)
    generate_datasets(tmp_path / 'b', rows=500, seed=17)
    generate_datasets(tmp_path / 'c', rows=500, seed=18)
    for name in ['ton_iot', 'bot_iot', 'swat']:
        a = json.loads((tmp_path / 'a' / name / 'SYNTHETIC_MANIFEST.json').read_text())
        b = json.loads((tmp_path / 'b' / name / 'SYNTHETIC_MANIFEST.json').read_text())
        c = json.loads((tmp_path / 'c' / name / 'SYNTHETIC_MANIFEST.json').read_text())
        assert a['files'] == b['files']
        assert a['files'][0]['sha256'] != c['files'][0]['sha256']
