import os
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.optimization.encoding import BinaryCodec
from src.optimization.evaluator import Evaluator, BudgetExhausted
from src.optimization.search import run_search
from src.training.engine import train_to, final_evaluate
from src.evaluation.diagnostics import run_diagnostics
from src.evaluation.reporting import generate_reports


@pytest.mark.parametrize('family', ['mlp', 'cnn', 'gru'])
def test_component_controls_resume_without_test(fixture_data, architecture, tmp_path, family):
    data, _ = fixture_data
    (data / 'test.npz').unlink()
    arch = dict(architecture, model_family=family)
    cfg = {'device': os.environ.get('BPSO_TEST_DEVICE', 'cpu'), 'class_weighting': 'balanced'}
    train_to(arch, data, tmp_path / 'full', 2, 41, cfg)
    train_to(arch, data, tmp_path / 'resumed', 1, 41, cfg)
    row = train_to(arch, data, tmp_path / 'resumed', 2, 41, cfg)
    a = torch.load(tmp_path / 'full/last.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/last.pt', weights_only=False)
    assert all(torch.equal(a['model'][key], b['model'][key]) for key in a['model'])
    assert 0 <= row['macro_f1'] <= 1
    assert row['device'] == cfg['device']


def config(data):
    return {'seed': 9, 'space': 'debug', 'data': {'processed': str(data)},
            'training': {'device': os.environ.get('BPSO_TEST_DEVICE', 'cpu'), 'threads': 1, 'prediction_epochs': [1, 2]},
            'search': {'method': 'bpso', 'population': 3, 'generations': 2, 'epochs': 2, 'min_epochs': 1,
                       'eta': 2, 'candidates': 5, 'brackets': 1}, 'compute': {'max_epochs': 12}}


@pytest.mark.parametrize('weighting', ['none', 'balanced'])
def test_exact_training_resume(fixture_data, architecture, tmp_path, weighting):
    data, _ = fixture_data
    cfg = {'device': os.environ.get('BPSO_TEST_DEVICE', 'cpu'), 'threads': 1, 'class_weighting': weighting}
    train_to(architecture, data, tmp_path / 'continuous', 3, 17, cfg)
    train_to(architecture, data, tmp_path / 'resumed', 1, 17, cfg)
    train_to(architecture, data, tmp_path / 'resumed', 3, 17, cfg)
    a = torch.load(tmp_path / 'continuous/last.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/last.pt', weights_only=False)
    for name in a['model']:
        assert torch.equal(a['model'][name], b['model'][name]), name
    assert [r['macro_f1'] for r in a['rows']] == [r['macro_f1'] for r in b['rows']]
    with pytest.raises(ValueError):
        train_to(architecture, data, tmp_path / 'resumed', 4, 18, cfg)


@pytest.mark.parametrize('weighting', ['none', 'balanced'])
def test_search_never_reads_test_and_cache_budget(fixture_data, architecture, tmp_path, weighting):
    data, _ = fixture_data
    (data / 'test.npz').unlink()  # evaluator must work without it
    cfg = config(data)
    cfg['compute']['max_epochs'] = 2
    cfg['training']['class_weighting'] = weighting
    codec = BinaryCodec('debug')
    evaluator = Evaluator(cfg, tmp_path / 'evaluation', codec)
    bits = codec.encode(architecture)
    evaluator.evaluate(bits, 1)
    evaluator.evaluate(bits, 1)
    assert evaluator.used_epochs == 1
    assert isinstance(evaluator.events[0]['collapsed_to_single_class'], bool)
    assert evaluator.events[0]['predicted_class_count'] >= 1
    assert np.isclose(evaluator.events[0]['majority_baseline_macro_f1'], 1/6)
    evaluator.evaluate(bits, 2)
    assert evaluator.used_epochs == 2
    with pytest.raises(BudgetExhausted):
        evaluator.evaluate(bits, 3)


@pytest.mark.parametrize('method', ['bpso', 'bpso_reduced', 'bpso_sh', 'hyperband', 'random', 'surrogate', 'dehb'])
def test_all_search_methods(fixture_data, tmp_path, method):
    data, _ = fixture_data
    cfg = config(data)
    cfg['search']['method'] = method
    cfg['search']['warmup'] = 2
    result = run_search(cfg, tmp_path / method)
    assert result['best'] is not None
    assert result['charged_epochs'] <= 12
    assert 0 <= result['best']['fitness'] <= 1
    if method != 'dehb':
        repeat = run_search(cfg, tmp_path / method)
        assert repeat['charged_epochs'] == result['charged_epochs']
        assert repeat['best']['fitness'] == result['best']['fitness']


def test_final_evaluation_and_plots(fixture_data, architecture, tmp_path):
    data, _ = fixture_data
    out = tmp_path / 'fixed'
    result = final_evaluate(architecture, data, out, 2, 19, {'device': os.environ.get('BPSO_TEST_DEVICE', 'cpu')}, 'normal')
    assert result['epochs'] == 2
    assert len(result['per_class']) == 3
    assert (out / 'per_class.csv').exists()
    report = generate_reports(out, tmp_path / 'report')
    assert report['final_runs'] == 1
    assert list((tmp_path / 'report').glob('*.pdf'))


def test_small_diagnostics(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    cfg['compute']['max_epochs'] = 30
    cfg['diagnostics'] = {'candidates': 4, 'seeds': [9], 'epochs': [1, 2], 'long_epochs': 3,
                          'shortlist': 1, 'audit_rejected': 1, 'top_k': 1}
    result = run_diagnostics(cfg, tmp_path / 'diagnostic')
    assert result['rankings'][0]['n'] == 4
    assert result['rankings'][0]['scope'] == 'complete cohort'


def test_seed_scoped_cache_and_monotone_resources(fixture_data, architecture, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    codec = BinaryCodec('debug')
    evaluator = Evaluator(cfg, tmp_path / 'seed_cache', codec)
    bits = codec.encode(architecture)
    evaluator.evaluate(bits, 2, seed=7)
    evaluator.evaluate(bits, 1, seed=7)
    assert evaluator.used_epochs == 2
    evaluator.evaluate(bits, 1, seed=8)
    assert evaluator.used_epochs == 3
    assert len(list((tmp_path / 'seed_cache/candidates').iterdir())) == 2


def test_swarm_replay_after_interrupt(fixture_data, tmp_path, monkeypatch):
    import src.optimization.evaluator as module
    data, _ = fixture_data
    cfg = config(data)
    uninterrupted = run_search(cfg, tmp_path / 'full')
    original = module.train_to
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt('simulated interruption between candidates')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'train_to', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_search(cfg, tmp_path / 'interrupted')
    monkeypatch.setattr(module, 'train_to', original)
    resumed = run_search(cfg, tmp_path / 'interrupted')
    assert resumed['best']['architecture'] == uninterrupted['best']['architecture']
    assert resumed['best']['fitness'] == uninterrupted['best']['fitness']
    assert resumed['charged_epochs'] == uninterrupted['charged_epochs']


def test_time_cap_stops_before_new_training(fixture_data, architecture, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    cfg['compute']['max_seconds'] = 0
    codec = BinaryCodec('debug')
    evaluator = Evaluator(cfg, tmp_path / 'time_cap', codec)
    with pytest.raises(BudgetExhausted):
        evaluator.evaluate(codec.encode(architecture), 1)
    assert evaluator.used_epochs == 0


def test_partial_fidelity_is_not_a_complete_selection(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    cfg['search'].update(method='bpso_sh', population=3, epochs=3, min_epochs=1)
    cfg['compute']['max_epochs'] = 3
    result = run_search(cfg, tmp_path / 'partial')
    assert result['best']['completed_budget'] == 1
    assert result['selection_status'] == 'incomplete_fidelity'
    assert not result['selection_complete']
    assert result['required_selection_fidelity'] == 3
    cfg['search']['method'] = 'bpso_reduced'
    reduced = run_search(cfg, tmp_path / 'reduced')
    assert reduced['selection_complete']
    assert reduced['required_selection_fidelity'] == 1


def test_diagnostics_follow_configured_promotion_factor(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    cfg['search']['eta'] = 2
    cfg['diagnostics'] = {'candidates': 4, 'seeds': [9], 'epochs': [1, 2],
                          'long_epochs': 2, 'shortlist': 1, 'audit_rejected': 1, 'top_k': 1}
    result = run_diagnostics(cfg, tmp_path / 'eta_two')
    assert result['rankings'][0]['kept'] == 2
    assert result['rankings'][0]['promotion_eta'] == 2


def test_comparison_skips_test_access_for_incomplete_selection(fixture_data, tmp_path, monkeypatch):
    import sys
    from src.cli import main
    data, _ = fixture_data
    (data / 'test.npz').unlink()
    cfg = config(data)
    cfg['search'].update(method='bpso_sh', population=3, epochs=3, min_epochs=1)
    cfg['compute']['max_epochs'] = 3
    cfg['comparison'] = {'seeds': [9], 'methods': ['bpso_sh'], 'final_train': True}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    output = tmp_path / 'comparison'
    monkeypatch.setattr(sys, 'argv', ['cli', 'compare', '--config', str(path), '--output', str(output)])
    main()
    status = json.loads((output / 'comparison_status.json').read_text())
    assert status[0]['final_status'] == 'skipped_incomplete_selection'
    assert not (output / 'bpso_sh/seed_9/final').exists()


def test_multiseed_diagnostic_reliability(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    cfg['compute']['max_epochs'] = 30
    cfg['diagnostics'] = {'candidates': 3, 'seeds': [9, 10], 'epochs': [1, 2], 'long_epochs': 2,
                          'shortlist': 1, 'audit_rejected': 1, 'top_k': 1}
    run_diagnostics(cfg, tmp_path / 'reliability')
    report = json.loads((tmp_path / 'reliability/fitness_seed_reliability.json').read_text())
    assert len(report['candidate_fitness_std']) == 3
    assert len(report['seed_pairs']) == 1
    assert (tmp_path / 'reliability/class_learning_changes.csv').exists()


def test_search_config_rejection_without_events(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = config(data)
    codec = BinaryCodec('debug')
    Evaluator(cfg, tmp_path / 'identity', codec)
    cfg['search']['population'] = 4
    with pytest.raises(ValueError):
        Evaluator(cfg, tmp_path / 'identity', codec)


def test_complete_long_cohort_and_tied_seed_winners(fixture_data, tmp_path, monkeypatch):
    data, _ = fixture_data
    cfg = config(data)
    cfg['compute']['max_epochs'] = 30
    cfg['diagnostics'] = {'candidates': 3, 'seeds': [9, 10], 'epochs': [1, 2], 'long_epochs': 3,
                          'shortlist': 3, 'audit_rejected': 0, 'top_k': 1}
    original = Evaluator.evaluate
    def tied_evaluate(self, *args, **kwargs):
        original(self, *args, **kwargs)
        return .25
    monkeypatch.setattr(Evaluator, 'evaluate', tied_evaluate)
    output = tmp_path / 'complete'
    result = run_diagnostics(cfg, output)
    assert all(row['scope'] == 'complete cohort' for row in result['rankings'])
    long = json.loads((output / 'long_cohort_seed_9.json').read_text())
    assert long['censoring'].startswith('none;')
    pair = json.loads((output / 'fitness_seed_reliability.json').read_text())['seed_pairs'][0]
    assert pair['winner_agreement'] is None
    assert pair['winner_sets_overlap'] is True
    assert pair['winner_indices_a'] == pair['winner_indices_b'] == [0, 1, 2]
