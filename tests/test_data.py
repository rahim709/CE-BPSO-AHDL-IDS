import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import prepare
from src.training.engine import load_partition


def test_predictor_overlap_detects_conflicting_labels_and_all_pairs():
    from src.data.preprocessing import overlap_diagnostics
    parts = [pd.DataFrame({'x': [1, 2], 'class': ['a', 'b']}),
             pd.DataFrame({'x': [1, 3], 'class': ['b', 'a']}),
             pd.DataFrame({'x': [3, 4], 'class': ['a', 'b']})]
    result = overlap_diagnostics(parts, ['x'], 'class')
    assert result['exact_row_overlap_train_val'] == 0
    assert result['predictor_overlap_train_val'] == 1
    assert result['exact_row_overlap_val_test'] == 1
    assert result['predictor_overlap_train_test'] == 0
    assert result['predictor_hashes_with_conflicting_labels'] == 1


def test_saved_disjoint_reproducible_splits(fixture_data, tmp_path):
    directory, cfg = fixture_data
    indices = json.loads((directory / 'split_indices.json').read_text())
    assert not set(indices['train']) & set(indices['test'])
    assert not set(indices['train']) & set(indices['val'])
    assert not set(indices['val']) & set(indices['test'])
    first = json.loads((directory / 'metadata.json').read_text())
    second = prepare(cfg, tmp_path / 'repeat')
    assert first['fingerprint'] == second['fingerprint']


def test_official_test_not_used_in_fit(tmp_path):
    train = tmp_path / 'train.csv'
    test = tmp_path / 'test.csv'
    pd.DataFrame({'x': np.arange(30), 'label': ['a', 'b']*15, 'proto': ['known']*30}).to_csv(train, index=False)
    pd.DataFrame({'x': [10000, -999], 'label': ['a', 'b'], 'proto': ['unseen']*2}).to_csv(test, index=False)
    cfg = {'name': 'gas_pipeline', 'target': 'label', 'files': [str(train)], 'test_files': [str(test)],
           'normalization': 'minmax'}
    output = tmp_path / 'processed'
    prepare(cfg, output)
    x, _ = load_partition(output, 'test')
    assert x[0, 0] > 100  # train-only range; test not silently clipped
    assert x[0, 1] == -1  # unknown category0 transformed by train-constant MinMax range1
    artifact = joblib.load(output / 'preprocessor.joblib')
    assert artifact['normalizer'].data_max_[0] <= 29
    cfg['features'] = ['label']
    with pytest.raises(ValueError):
        prepare(cfg, tmp_path / 'bad')


def test_column_l2_train_only(tmp_path):
    train, test = tmp_path / 'train.csv', tmp_path / 'test.csv'
    pd.DataFrame({'x': [3.]*30, 'class': ['n', 'a']*15}).to_csv(train, index=False)
    pd.DataFrame({'x': [1000.], 'class': ['n']}).to_csv(test, index=False)
    cfg = {'name': 'gas_pipeline', 'target': 'class', 'files': [str(train)], 'test_files': [str(test)]}
    out = tmp_path / 'processed'
    prepare(cfg, out)
    x, _ = load_partition(out, 'train')
    tx, _ = load_partition(out, 'test')
    assert np.allclose(np.linalg.norm(x, axis=0), 1)
    assert tx[0, 0] > 1


def test_missing_files_not_substituted(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare({'name': 'gas_pipeline', 'files': [str(tmp_path / 'missing.csv')]}, tmp_path)


def test_each_source_pattern_must_match(tmp_path):
    from src.data.preprocessing import load_files
    source = tmp_path / 'present.csv'
    pd.DataFrame({'x': [1]}).to_csv(source, index=False)
    with pytest.raises(FileNotFoundError, match='missing'):
        load_files([str(source), str(tmp_path / 'missing*.csv')])


def test_source_aliases_do_not_duplicate_rows(tmp_path):
    from src.data.preprocessing import load_files
    source = tmp_path / 'source.csv'
    alias = tmp_path / 'alias.csv'
    pd.DataFrame({'x': [1, 2]}).to_csv(source, index=False)
    alias.symlink_to(source)
    frame, sources = load_files([str(source), str(alias), str(tmp_path / '*.csv')])
    assert len(frame) == 2
    assert frame.index.is_unique
    assert len(sources) == 1


def test_saved_transform_reuse(fixture_data):
    from src.data.preprocessing import transform_new
    directory, cfg = fixture_data
    indices = json.loads((directory / 'split_indices.json').read_text())['test']
    raw = pd.read_csv(cfg['files'][0])
    rows = [int(i.rsplit(':', 1)[1]) for i in indices]
    actual = transform_new(directory / 'preprocessor.joblib', raw.iloc[rows])
    expected, _ = load_partition(directory, 'test')
    assert np.array_equal(actual, expected)


def test_group_split_keeps_capture_groups_disjoint(tmp_path):
    raw = tmp_path / 'groups.csv'
    frame = pd.DataFrame({'x': np.arange(120), 'capture': np.repeat(np.arange(20), 6),
                          'class': ['normal', 'attack']*60})
    frame.to_csv(raw, index=False)
    out = tmp_path / 'processed'
    prepare({'name': 'gas_pipeline', 'files': [str(raw)], 'target': 'class', 'split': 'group',
             'group_column': 'capture'}, out)
    ids = json.loads((out / 'split_indices.json').read_text())
    groups = {name: set(frame.iloc[[int(i.rsplit(':', 1)[1]) for i in rows]]['capture'])
              for name, rows in ids.items()}
    assert not groups['train'] & groups['val']
    assert not groups['train'] & groups['test']
    assert not groups['val'] & groups['test']
    assert 'capture' not in json.loads((out / 'metadata.json').read_text())['features']


def test_chronological_split_respects_parsed_time(tmp_path):
    raw = tmp_path / 'chronology.csv'
    times = pd.date_range('2025-01-30', periods=60, freq='D')
    frame = pd.DataFrame({'clock': times.strftime('%d/%m/%Y'), 'x': np.arange(60), 'class': ['a', 'b']*30})
    frame = frame.sample(frac=1, random_state=7).reset_index(drop=True)
    frame.to_csv(raw, index=False)
    out = tmp_path / 'processed'
    prepare({'name': 'gas_pipeline', 'files': [str(raw)], 'target': 'class', 'split': 'chronological',
             'time_column': 'clock', 'time_format': '%d/%m/%Y'}, out)
    ids = json.loads((out / 'split_indices.json').read_text())
    values = {name: frame.iloc[[int(i.rsplit(':', 1)[1]) for i in rows]]['x'] for name, rows in ids.items()}
    assert values['train'].max() < values['val'].min()
    assert values['val'].max() < values['test'].min()


def test_missing_values_and_unseen_target_fail(tmp_path):
    raw = tmp_path / 'missing.csv'
    test = tmp_path / 'test.csv'
    pd.DataFrame({'x': [1., np.inf, np.nan, 4.]*10, 'protocol': ['a', None, 'b', '?']*10,
                  'class': ['a', 'b']*20}).to_csv(raw, index=False)
    cfg = {'name': 'gas_pipeline', 'files': [str(raw)], 'target': 'class'}
    out = tmp_path / 'processed'
    prepare(cfg, out)
    for part in ['train', 'val', 'test']:
        x, _ = load_partition(out, part)
        assert np.isfinite(x).all()
    pd.DataFrame({'x': [1], 'protocol': ['a'], 'class': ['unseen_attack']}).to_csv(test, index=False)
    cfg['test_files'] = [str(test)]
    with pytest.raises(ValueError):
        prepare(cfg, tmp_path / 'unseen')
