"""Conventional neural controls, kept outside the binary NAS search space."""
from __future__ import annotations
from torch import nn
from src.models.hybrid import HybridIDS


class ComponentControl(nn.Module):
    def __init__(self, family: str, input_features: int, classes: int):
        super().__init__()
        self.family = family
        if family == 'mlp':
            self.network = nn.Sequential(nn.Linear(input_features, 64), nn.ReLU(), nn.Dropout(.1),
                                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, classes))
        elif family == 'cnn':
            self.network = nn.Sequential(nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                nn.Flatten(), nn.Linear(16 * (input_features // 2), 32), nn.ReLU(), nn.Dropout(.1), nn.Linear(32, classes))
        elif family == 'gru':
            self.recurrent = nn.GRU(1, 32, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(32, classes))
        else:
            raise ValueError(f'Unknown control family {family}')

    def forward(self, x):
        if self.family == 'gru':
            return self.head(self.recurrent(x.unsqueeze(-1))[0][:, -1])
        return self.network(x.unsqueeze(1) if self.family == 'cnn' else x)

    def penalty(self, l1, l2):
        return next(self.parameters()).new_zeros(())

    @property
    def cnn(self):
        # The shared trainer records short-sequence repairs from hybrid CNN blocks.
        return []


def make_model(architecture: dict, input_features: int, classes: int) -> nn.Module:
    family = architecture.get('model_family', 'hybrid')
    return (HybridIDS(architecture, input_features, classes) if family == 'hybrid'
            else ComponentControl(family, input_features, classes))
