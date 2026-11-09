"""Feature-axis Conv1d followed by sequence-preserving recurrent blocks."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def activation(name: str) -> nn.Module:
    options = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'none': nn.Identity}
    if name not in options:
        raise ValueError(f'Unknown activation {name}')
    return options[name]()


class ConvBlock(nn.Module):
    def __init__(self, channels: int, cfg: dict) -> None:
        super().__init__()
        if cfg['padding'] not in {'same', 'valid'} or min(cfg['filters'], cfg['kernel'], cfg['pool']) < 1:
            raise ValueError('Invalid convolution dimensions or padding')
        self.kernel, self.pool = cfg['kernel'], cfg['pool']
        self.same = cfg['padding'] == 'same'
        self.conv = nn.Conv1d(channels, cfg['filters'], self.kernel)
        self.act = activation(cfg['activation'])
        self.repairs = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.same:
            total = self.kernel - 1
            x = F.pad(x, (total // 2, total - total // 2))
        elif x.shape[-1] < self.kernel:
            x = F.pad(x, (0, self.kernel - x.shape[-1]))
            self.repairs += 1
        x = self.act(self.conv(x))
        if x.shape[-1] < self.pool:
            x = F.pad(x, (0, self.pool - x.shape[-1]))
            self.repairs += 1
        return F.max_pool1d(x, self.pool)


class RecurrentBlock(nn.Module):
    def __init__(self, channels: int, cfg: dict) -> None:
        super().__init__()
        kind = cfg['kind']
        if cfg['units'] < 1 or not 0 <= cfg['dropout'] < 1 or cfg['regularization'] not in {'none', 'l1', 'l2', 'l1_l2'}:
            raise ValueError('Invalid recurrent configuration')
        options = {'simrnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU, 'linear': nn.Linear}
        if kind not in options:
            raise ValueError(kind)
        self.linear = kind == 'linear'
        self.layer = options[kind](channels, cfg['units'], **({} if self.linear else {'batch_first': True}))
        self.act, self.dropout = activation(cfg['activation']), nn.Dropout(cfg['dropout'])
        self.regularization = cfg['regularization']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        return self.dropout(self.act(x if self.linear else x[0]))


class HybridIDS(nn.Module):
    def __init__(self, architecture: dict, input_features: int, classes: int) -> None:
        super().__init__()
        if input_features < 1 or classes < 2 or not architecture['cnn'] or not architecture['rnn']:
            raise ValueError('Require features, at least two classes, CNN and recurrent blocks')
        self.input_features = input_features
        self.cnn = nn.ModuleList()
        channels = 1
        for cfg in architecture['cnn']:
            self.cnn.append(ConvBlock(channels, cfg))
            channels = cfg['filters']
        self.rnn = nn.ModuleList()
        for cfg in architecture['rnn']:
            self.rnn.append(RecurrentBlock(channels, cfg))
            channels = cfg['units']
        self.classifier = nn.Linear(channels, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_features:
            raise ValueError('Input must be (batch, input_features)')
        x = x.unsqueeze(1)
        for block in self.cnn:
            x = block(x)
        x = x.transpose(1, 2)
        for block in self.rnn:
            x = block(x)
        return self.classifier(x[:, -1, :])

    def penalty(self, l1: float, l2: float) -> torch.Tensor:
        loss = self.classifier.weight.new_zeros(())
        for block in self.rnn:
            for name, p in block.layer.named_parameters():
                if 'weight' in name:
                    if block.regularization in {'l1', 'l1_l2'}:
                        loss = loss + l1 * p.abs().sum()
                    if block.regularization in {'l2', 'l1_l2'}:
                        loss = loss + l2 * p.square().sum()
        return loss


def make_optimizer(model: nn.Module, architecture: dict) -> torch.optim.Optimizer:
    name, lr = architecture['optimizer'], architecture['learning_rate']
    if lr <= 0:
        raise ValueError('Learning rate must be positive')
    if name in {'sgd', 'momentum', 'nesterov'}:
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=.9 if name != 'sgd' else 0,
                               nesterov=name == 'nesterov')
    options = {'adam': torch.optim.Adam, 'adamax': torch.optim.Adamax, 'adagrad': torch.optim.Adagrad,
               'adadelta': torch.optim.Adadelta, 'rmsprop': torch.optim.RMSprop}
    if name not in options:
        raise ValueError(name)
    return options[name](model.parameters(), lr=lr)
