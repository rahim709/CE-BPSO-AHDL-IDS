import numpy as np
from src.evaluation.metrics import classification_metrics
from src.evaluation.ranking import ranking_metrics


def test_known_confusion():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 0, 2, 0])
    m = classification_metrics(y, pred, ['normal', 'a', 'b'], 'normal')
    assert m['accuracy'] == .5
    assert m['attack_detection']['false_positive'] == 1
    assert m['attack_detection']['false_negative'] == 2
    assert m['per_class']['b']['recall'] == .5


def test_rank_reversal():
    early, late = np.array([1., 2., 3.]), np.array([3., 2., 1.])
    r = ranking_metrics(early, late, k=1)
    assert r['spearman'] == -1
    assert r['kendall'] == -1
    assert r['selection_regret'] == 2
    assert r['premature_elimination_rate'] == 1
    assert r['top_k_overlap'] == 0


def test_seed_statistics_require_independent_seeds():
    import pytest
    from src.evaluation.reporting import seed_summary
    rows = [{'method': 'random', 'data_fingerprint': 'data1', 'protocol_fingerprint': 'protocol1',
             'seed': i, 'accuracy': value} for i, value in [(42, .7), (43, .9)]]
    summary = seed_summary(rows, ['accuracy'])
    assert summary[0]['n'] == 2
    assert np.isclose(summary[0]['mean'], .8)
    assert summary[0]['ci95_half_width'] > 0
    with pytest.raises(ValueError):
        seed_summary([rows[0], rows[0]], ['accuracy'])
