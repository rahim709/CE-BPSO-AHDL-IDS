"""Software-only fixtures; they are not IDS performance evidence."""
import json
import numpy as np
import pytest
from src.optimization.online import OnlinePredictor
from src.optimization.encoding import BinaryCodec
from src.optimization.search import run_search
from src.utils.core import digest
from test_training_search import config
from test_curve_predictor import rows


def test_online_capture_never_uses_future(architecture):
    model = OnlinePredictor(BinaryCodec('debug'))
    trajectory = rows()
    key = model.capture(architecture, trajectory, 3)
    snapshot = list(model.records[key]['snapshots'][3]['features'])
    trajectory[-1]['macro_f1'] = 1e8
    independent = OnlinePredictor(BinaryCodec('debug'))
    independent.capture(architecture, trajectory, 3)
    assert snapshot == independent.records[key]['snapshots'][3]['features']
    model.capture(architecture, trajectory, 3)
    assert snapshot == model.records[key]['snapshots'][3]['features']
    assert model.records[key]['target'] is None
    model.label(key, .7)
    assert snapshot == model.records[key]['snapshots'][3]['features']


def toy_record(partition, x, target=None):
    return {'partition': partition, 'snapshots': {3: {'features': [float(x)], 'observed_f1': float(x)*.1}},
            'forecasts': {}, 'target': target}


def test_prequential_gate_and_query_exclusion():
    model = OnlinePredictor(BinaryCodec(), {'max_width': 1.})
    for i, x in enumerate(np.linspace(0, 1, 8)):
        model.records[f'fit_{i}'] = toy_record('fit', x, .1+.6*x)
    for i, x in enumerate([.1,.3,.5,.7,.9]):
        model.records[f'cal_{i}'] = toy_record('calibration', x, .1+.6*x)
    fitted, gate = model.prepare(3, set())
    assert fitted is not None and not gate['allowed']
    for i, x in enumerate([.05,.25,.65,.95]):
        key = f'check_{i}'
        model.records[key] = toy_record('check', x)
        prediction = model.predict(key, 3, fitted)
        assert model.records[key]['target'] is None
        assert model.records[key]['forecasts'][3]['made_before_target']
        model.label(key, prediction['mean'])
    _, gate = model.prepare(3, set())
    assert gate['allowed'] and gate['mae'] == 0 and gate['coverage'] == 1
    # Previously useful shadow predictions cannot authorize promotion after
    # held-out outcomes demonstrate large error/drift.
    for i in range(4):
        model.records[f'check_{i}']['target'] = 1.
    _, failed = model.prepare(3, set())
    assert not failed['allowed']
    assert failed['reason'] == 'inadequate_prequential_accuracy_or_calibration'
    excluded = {'fit_0','cal_0'}
    other, _ = model.prepare(3, excluded)
    assert other is not None
    assert not (excluded & set(other['fit_ids']+other['calibration_ids']))
    # A check target observed before forecasting is not a held-out online check.
    model.records['late'] = toy_record('check', .4, .5)
    model.predict('late', 3, fitted)
    assert not model.records['late']['forecasts']


def test_own_architecture_cannot_enter_prediction():
    model = OnlinePredictor(BinaryCodec())
    model.records['a'] = toy_record('fit', .5, .5)
    with pytest.raises(ValueError, match='own predictor'):
        model.predict('a',3,{'fit_ids':['a'],'calibration_ids':[]})


def test_archive_free_is_exact_sh_until_gate(fixture_data, tmp_path, monkeypatch):
    from src.optimization.curve_predictor import CurvePredictor
    data, _ = fixture_data
    (data/'test.npz').unlink()
    cfg = config(data)
    cfg['search'].update(method='bpso_sh',budgets=[1,2])
    ordinary = run_search(cfg,tmp_path/'sh')
    def forbidden(*args, **kwargs):
        raise AssertionError('An offline predictor archive was read')
    monkeypatch.setattr(CurvePredictor,'load',forbidden)
    cfg['search'].update(method='ua_amf_bpso',promotion_policy='online')
    online = run_search(cfg,tmp_path/'online')
    left=json.loads((tmp_path/'sh/evaluations.json').read_text())
    right=json.loads((tmp_path/'online/evaluations.json').read_text())
    assert [(e['architecture'],e['requested_budget'],e['fitness']) for e in left] == [(e['architecture'],e['requested_budget'],e['fitness']) for e in right]
    assert ordinary['charged_epochs']==online['charged_epochs']
    assert ordinary['best']['fitness']==online['best']['fitness']
    summary=json.loads((tmp_path/'online/online_summary.json').read_text())
    assert summary['prediction_applied_decisions']==0 and summary['cpu_fit_count']==0
    repeated=run_search(cfg,tmp_path/'online')
    assert repeated['best']['fitness']==online['best']['fitness']
    assert repeated['charged_epochs']==online['charged_epochs']


def test_bundle_rejected_by_archive_free(fixture_data,tmp_path):
    data,_=fixture_data
    cfg=config(data)
    cfg['search'].update(method='ua_amf_bpso',promotion_policy='online',predictor_bundle='must-not-be-read.joblib')
    with pytest.raises(ValueError,match='must not receive'):
        run_search(cfg,tmp_path/'invalid')
