"""Artificial numeric fixtures only validate software, not IDS performance."""
import numpy as np
import pytest
from src.optimization.curve_predictor import CurvePredictor, curve_features
from src.utils.core import digest


def rows():
    return [{'epoch': i, 'macro_f1': i / 100, 'loss': 1., 'training_seconds': .1, 'validation_seconds': .01, 'parameters': 10} for i in range(1, 31)]


def records():
    rng = np.random.default_rng(2)
    return [{'architecture_id': str(i), 'partition': 'fit' if i < 8 else 'calibration' if i < 14 else 'test',
             'budget': b, 'features': rng.normal(size=5).tolist(), 'target': i / 20} for i in range(20) for b in [3, 10]]


def test_future_rows_never_enter_features(architecture):
    from src.optimization.encoding import BinaryCodec
    original = rows()
    before = curve_features(architecture, original, 3, BinaryCodec('debug'))
    original[-1]['macro_f1'] = 1e8
    assert np.array_equal(before, curve_features(architecture, original, 3, BinaryCodec('debug')))


def test_group_disjoint_enforced_and_test_targets_unused():
    data = records()
    first = CurvePredictor().fit(data, 'context')
    for row in data:
        if row['partition'] == 'test':
            row['target'] = -1000
    second = CurvePredictor().fit(data, 'context')
    a = first.predict_features([data[0]['features']], 3)
    b = second.predict_features([data[0]['features']], 3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    data.append(dict(data[0], partition='calibration'))
    with pytest.raises(ValueError, match='crosses'):
        CurvePredictor().fit(data, 'context')


def test_bundle_roundtrip_and_context_guard(tmp_path, architecture):
    model = CurvePredictor().fit(records(), 'context')
    model.save(tmp_path / 'model.joblib')
    model = CurvePredictor.load(tmp_path / 'model.joblib')
    with pytest.raises(ValueError, match='context'):
        model.predict(architecture, rows(), 3, 'other')
    model.archive_architectures.add(digest(architecture))
    with pytest.raises(ValueError, match='overlaps'):
        model.predict(architecture, rows(), 3, 'context')
    _, radius, low, high = model.predict_features([records()[0]['features']], 3)
    assert 0 <= low[0] <= high[0] <= 1 and radius[0] >= 0


def test_acquisition_units_and_ablation():
    from src.optimization.predictive import acquisition
    assert acquisition(.5, .1, 60, beta=1, cost_weight=.05) == pytest.approx(.55)
    assert acquisition(.5, .1, 60, beta=0, cost_weight=0) == .5
    with pytest.raises(ValueError):
        acquisition(.5, .1, -1)


@pytest.mark.parametrize('policy', ['learning_curve', 'dynamic'])
def test_predictive_scheduler_integration(fixture_data, tmp_path, policy):
    import os
    from src.optimization.encoding import BinaryCodec
    from src.optimization.search import run_search
    from test_training_search import config
    data, _ = fixture_data
    (data / 'test.npz').unlink()
    cfg = config(data)
    codec = BinaryCodec('debug')
    feature_count = codec.length + 10
    data_records = records()
    for r in data_records:
        r['features'] = np.zeros(feature_count).tolist()
        r['budget'] = 1
    import json
    from src.utils.core import digest
    meta = json.loads((data / 'metadata.json').read_text())
    context = digest({'data_fingerprint': meta['fingerprint'], 'training': cfg['training'], 'space': cfg['space']})
    predictor = CurvePredictor().fit(data_records, context)
    path = tmp_path / 'bundle.joblib'
    predictor.save(path)
    cfg['search'].update(method='ua_amf_bpso', promotion_policy=policy, predictor_bundle=str(path), budgets=[1, 2], generations=1)
    cfg['compute']['max_epochs'] = 4
    result = run_search(cfg, tmp_path / 'search')
    assert result['charged_epochs'] == 4
    assert result['selection_complete']
    assert result['best']['completed_budget'] == 2
    again = run_search(cfg, tmp_path / 'search')
    assert again['best']['fitness'] == result['best']['fitness']
    assert again['charged_epochs'] == result['charged_epochs']


def test_changed_bundle_rejected_before_training(fixture_data, tmp_path):
    from src.optimization.search import run_search
    from test_training_search import config
    data, _ = fixture_data
    cfg = config(data)
    path = tmp_path / 'changed.joblib'
    path.write_bytes(b'changed after the protocol was frozen')
    cfg['search'].update(method='ua_amf_bpso', promotion_policy='learning_curve', predictor_bundle=str(path),
                         predictor_sha256='0' * 64, budgets=[1, 2])
    with pytest.raises(ValueError, match='changed after protocol freeze'):
        run_search(cfg, tmp_path / 'rejected')
    import json
    assert json.loads((tmp_path / 'rejected/evaluations.json').read_text()) == []
