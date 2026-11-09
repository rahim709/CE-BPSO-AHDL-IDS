"""Quality diagnostics and train-only weighting, beyond successful execution."""
import numpy as np
import pytest
import torch
from src.evaluation.metrics import classification_metrics, majority_baseline
from src.evaluation.ranking import ranking_metrics
from src.training.losses import class_weights


def test_collapsed_predictions_are_visible_despite_high_accuracy():
    y = np.array([0] * 95 + [1] * 5)
    metrics = classification_metrics(y, np.zeros_like(y), ['normal', 'attack'])
    assert metrics['accuracy'] == .95
    assert metrics['collapsed_to_single_class'] is True
    assert metrics['predicted_class_count'] == 1
    assert metrics['zero_recall_classes'] == ['attack']
    assert metrics['prediction_counts'] == {'normal': 100, 'attack': 0}


def test_majority_baseline_uses_training_vocabulary_not_evaluation_majority():
    baseline = majority_baseline(np.array([0, 0, 1]), np.array([1, 1, 1, 0]), ['a', 'b'])
    assert baseline['training_majority_class'] == 'a'
    assert baseline['accuracy'] == .25


def test_constant_rankings_do_not_report_perfect_overlap_as_evidence():
    result = ranking_metrics(np.array([.1, .1, .1]), np.array([.2, .2, .2]), k=1)
    assert result['ranking_informative'] is False
    assert result['top_k_overlap'] is None
    assert result['premature_elimination_rate'] is None
    assert result['spearman'] is None
    assert result['deterministic_tie_break']['top_k_overlap'] == 1


def test_tied_early_winners_have_a_regret_interval():
    result = ranking_metrics(np.array([.5, .5, .2]), np.array([.1, .9, .3]), k=1)
    assert result['selection_regret'] is None
    assert result['selection_regret_range'] == [0, .8]
    assert result['top_k_overlap'] is None


def test_ties_outside_selection_boundaries_preserve_identifiable_metrics():
    result = ranking_metrics(np.array([.9, .8, .1, .1]), np.array([.9, .7, .2, .2]), k=1, keep_fraction=.5)
    assert result['ranking_informative'] is True
    assert result['top_k_overlap'] == 1
    assert result['premature_elimination_rate'] == 0


def test_balanced_loss_equalizes_aggregate_class_contributions():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    weights = class_weights(y, 3, 'balanced').numpy()
    assert np.allclose(np.bincount(y) * weights, np.repeat(80 * weights[0], 3))
    assert class_weights(y, 3, 'none') is None
    sqrt_weights = class_weights(y, 3, 'sqrt_balanced').numpy()
    assert np.isclose(sqrt_weights[2] / sqrt_weights[0], 4)


def test_balanced_loss_increases_minority_gradient_contribution():
    y = torch.tensor([0, 0, 0, 1])
    logits = torch.zeros(4, 2, requires_grad=True)
    loss = torch.nn.functional.cross_entropy(logits, y, weight=class_weights(y.numpy(), 2, 'balanced'))
    loss.backward()
    norm = logits.grad.norm(dim=1)
    assert torch.allclose(norm[3], 3 * norm[0])


def test_invalid_weighting_and_missing_training_classes_fail():
    with pytest.raises(ValueError):
        class_weights(np.array([0, 1]), 2, 'typo')
    with pytest.raises(ValueError):
        class_weights(np.array([0, 0]), 2, 'balanced')
