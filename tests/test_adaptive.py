"""Software correctness tests; fixtures are not IDS performance evidence."""
import json
import numpy as np
import pytest
from src.optimization.adaptive import promotion_scores
from src.optimization.search import run_search
from test_training_search import config


def test_sem_and_cost_ablation():
    values = [[.2, .4, .6], [.39, .4, .41]]
    scores, means, sem = promotion_scores(values, [1, 2], 1, 0)
    assert means == pytest.approx([.4, .4])
    assert sem[0] == pytest.approx(.2 / np.sqrt(3))
    assert scores[0] > scores[1]
    scores, _, _ = promotion_scores(values, [1, 2], 0, .1)
    assert scores[0] > scores[1]
    with pytest.raises(ValueError):
        promotion_scores([[.4]], [1])


def adaptive_config(data):
    cfg = config(data)
    cfg['search'].update(method='ua_amf_bpso', budgets=[1, 2], population=3,
                         generations=2, replicate_offsets=[0, 100, 200])
    cfg['compute']['max_epochs'] = 24
    return cfg


def test_adaptive_validation_only_replay(fixture_data, tmp_path):
    data, _ = fixture_data
    (data / 'test.npz').unlink()
    cfg = adaptive_config(data)
    out = tmp_path / 'adaptive'
    result = run_search(cfg, out)
    repeated = run_search(cfg, out)
    assert result['best'] == repeated['best']
    assert result['charged_epochs'] == repeated['charged_epochs'] <= 24
    assert result['best']['completed_budget'] == 2
    assert result['best']['fitness'] == pytest.approx(np.mean(result['best']['observed_seed_fitness']))
    records = json.loads((out / 'promotions.json').read_text())
    assert records and all(len(r['selected_indices']) == 1 for r in records)


def test_partial_replicated_job_cannot_win(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = adaptive_config(data)
    cfg['compute']['max_epochs'] = 10
    result = run_search(cfg, tmp_path / 'partial')
    assert not result['selection_complete']
    assert result['best'] is None
    assert result['charged_epochs'] == 9


def test_adaptive_interruption_replay(fixture_data, tmp_path, monkeypatch):
    import src.optimization.evaluator as module
    data, _ = fixture_data
    cfg = adaptive_config(data)
    full = run_search(cfg, tmp_path / 'full')
    original = module.train_to
    count = 0
    def interrupt(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 11:
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'train_to', interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_search(cfg, tmp_path / 'resume')
    monkeypatch.setattr(module, 'train_to', original)
    resumed = run_search(cfg, tmp_path / 'resume')
    assert resumed['best'] == full['best']
    assert resumed['charged_epochs'] == full['charged_epochs']


def test_unlogged_epoch_recovery(fixture_data, architecture, tmp_path):
    from src.optimization.encoding import BinaryCodec
    from src.optimization.evaluator import Evaluator
    from src.training.engine import train_to
    data, _ = fixture_data
    cfg = adaptive_config(data)
    codec = BinaryCodec('debug')
    out = tmp_path / 'recovery'
    evaluator = Evaluator(cfg, out, codec)
    path = evaluator.candidate_path(architecture, cfg['seed'])
    train_to(architecture, data, path, 1, cfg['seed'], cfg['training'])
    # Simulate process exit between checkpoint commit and trajectory/event commit.
    (path / 'trajectory.json').unlink()
    recovered = Evaluator(cfg, out, codec)
    assert recovered.used_epochs == 1
    assert recovered.best()['fitness'] is not None
    recovered.evaluate(codec.encode(architecture), 2)
    assert recovered.used_epochs == 2
    again = Evaluator(cfg, out, codec)
    assert again.used_epochs == 2


def test_gpu_cap_checked_before_training(fixture_data, architecture, tmp_path):
    from src.optimization.encoding import BinaryCodec
    from src.optimization.evaluator import Evaluator, BudgetExhausted
    data, _ = fixture_data
    cfg = adaptive_config(data)
    cfg['compute']['max_gpu_seconds'] = 0
    evaluator = Evaluator(cfg, tmp_path / 'gpu_cap', BinaryCodec('debug'))
    evaluator.on_cuda = True  # Hardware-independent enforcement test.
    with pytest.raises(BudgetExhausted):
        evaluator.evaluate(evaluator.codec.encode(architecture), 1)
    assert evaluator.used_epochs == 0


def test_oom_is_logged_and_not_a_winner(fixture_data, architecture, tmp_path, monkeypatch):
    import torch
    import src.optimization.evaluator as module
    from src.optimization.encoding import BinaryCodec
    data, _ = fixture_data
    cfg = adaptive_config(data)
    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError('test allocation failure')
    monkeypatch.setattr(module, 'train_to', oom)
    evaluator = module.Evaluator(cfg, tmp_path / 'oom', BinaryCodec('debug'))
    assert evaluator.evaluate(evaluator.codec.encode(architecture), 1) == -np.inf
    assert evaluator.events[0]['failure'] == 'test allocation failure'
    assert evaluator.events[0]['wall_seconds'] > 0
    assert evaluator.best() is None


def test_invalid_ladder_rejected(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = adaptive_config(data)
    cfg['search']['budgets'] = [2, 1]
    with pytest.raises(ValueError):
        run_search(cfg, tmp_path / 'bad')


def test_three_rung_continuation(fixture_data, tmp_path):
    data, _ = fixture_data
    cfg = adaptive_config(data)
    cfg['search'].update(budgets=[1, 2, 3], epochs=3, generations=1)
    cfg['compute']['max_epochs'] = 15
    result = run_search(cfg, tmp_path / 'three_rungs')
    assert result['charged_epochs'] == 15
    assert result['best']['completed_budget'] == 3
    assert result['selection_complete']


def test_decoded_aliases_share_replicated_training(fixture_data, architecture, tmp_path, monkeypatch):
    import src.optimization.adaptive as module
    from src.optimization.encoding import BinaryCodec
    from src.optimization.bpso import BPSO
    data, _ = fixture_data
    codec = BinaryCodec('debug')
    bits = codec.encode(architecture)
    alias = None
    for i in range(len(bits)):
        trial = bits.copy()
        trial[i] ^= 1
        if codec.decode(trial) == architecture:
            alias = trial
            break
    assert alias is not None
    class AliasedSwarm(BPSO):
        def __init__(self, *args):
            super().__init__(*args)
            self.position[:] = bits
            self.position[1] = alias
    monkeypatch.setattr(module, 'BPSO', AliasedSwarm)
    cfg = adaptive_config(data)
    cfg['search']['generations'] = 1
    cfg['compute']['max_epochs'] = 6
    result = run_search(cfg, tmp_path / 'aliases')
    assert result['charged_epochs'] == 6
    assert result['unique_architectures'] == 1
    assert result['selection_complete']
