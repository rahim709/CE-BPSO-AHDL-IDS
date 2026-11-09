import os
import numpy as np
import pytest
import torch
from src.models.hybrid import HybridIDS, make_optimizer
from src.optimization.encoding import BinaryCodec


def test_paper_fields_and_examples():
    codec = BinaryCodec()
    assert codec.length == 73
    bits = np.zeros(73, dtype=np.int8)
    bits[:8] = list(map(int, '11001010'))
    bits[10:20] = list(map(int, '1101010000'))
    arch = codec.decode(bits)
    assert (arch['batch_size'], arch['optimizer'], arch['learning_rate']) == (2048, 'momentum', .0001)
    assert arch['cnn'][0] == {'padding': 'valid', 'filters': 256, 'kernel': 3, 'pool': 1, 'activation': 'sigmoid'}
    with pytest.raises(ValueError):
        codec.decode(np.zeros(10))
    with pytest.raises(ValueError):
        codec.decode(np.full(73, 2))


@pytest.mark.parametrize('space', ['paper', 'debug'])
def test_roundtrip(space):
    codec = BinaryCodec(space)
    rng = np.random.default_rng(9)
    for _ in range(100):
        arch = codec.decode(codec.sample(rng))
        assert codec.decode(codec.encode(arch)) == arch


@pytest.mark.parametrize('kind', ['simrnn', 'lstm', 'gru', 'linear'])
@pytest.mark.parametrize('padding', ['valid', 'same'])
def test_forward_shape_and_gradients(architecture, kind, padding):
    architecture['rnn'][0]['kind'] = kind
    architecture['cnn'][0].update(padding=padding, kernel=5, pool=4)
    model = HybridIDS(architecture, 1, 3).to(os.environ.get('BPSO_TEST_DEVICE', 'cpu'))
    out = model(torch.randn(2, 1, device=os.environ.get('BPSO_TEST_DEVICE', 'cpu')))
    assert out.shape == (2, 3)
    out.sum().backward()
    assert model.classifier.weight.grad is not None


def test_random_architectures():
    codec = BinaryCodec('debug')
    rng = np.random.default_rng(7)
    for _ in range(20):
        model = HybridIDS(codec.decode(codec.sample(rng)), 3, 4).to(os.environ.get('BPSO_TEST_DEVICE', 'cpu'))
        assert model(torch.randn(2, 3, device=os.environ.get('BPSO_TEST_DEVICE', 'cpu'))).shape == (2, 4)


def test_invalid_model(architecture):
    architecture['cnn'][0]['pool'] = 0
    with pytest.raises(ValueError):
        HybridIDS(architecture, 2, 3)


@pytest.mark.parametrize('optimizer', ['sgd', 'momentum', 'nesterov', 'adam', 'adamax', 'rmsprop', 'adadelta', 'adagrad'])
def test_optimizer_step(architecture, optimizer):
    architecture['optimizer'] = optimizer
    model = HybridIDS(architecture, 2, 3).to(os.environ.get('BPSO_TEST_DEVICE', 'cpu'))
    opt = make_optimizer(model, architecture)
    model(torch.randn(2, 2, device=os.environ.get('BPSO_TEST_DEVICE', 'cpu'))).sum().backward()
    opt.step()


def test_encoding_rejects_missing_active_fields(architecture):
    architecture['cnn'][0].pop('kernel')
    with pytest.raises(ValueError):
        BinaryCodec('debug').encode(architecture)
