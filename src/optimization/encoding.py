"""Fixed-slot binary interpretation of the paper's Fig.5."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Field:
    name: str
    values: tuple[Any, ...]
    @property
    def width(self) -> int:
        return (len(self.values) - 1).bit_length()


class BinaryCodec:
    def __init__(self, space: str = 'paper') -> None:
        if space not in {'paper', 'debug'}:
            raise ValueError(space)
        units = (8, 16, 32, 64, 128, 256, 384, 512) if space == 'paper' else (4, 8)
        activation = ('sigmoid', 'tanh', 'relu', 'none')
        self.fields = [Field('batch_size', (256, 512, 1024, 2048) if space == 'paper' else (8, 16)),
                       Field('optimizer', ('sgd', 'momentum', 'nesterov', 'adagrad', 'adamax', 'adam', 'rmsprop', 'adadelta')),
                       Field('learning_rate', (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2)),
                       Field('cnn_count', (1, 2, 3, 4) if space == 'paper' else (1, 2))]
        self.max_cnn = 4 if space == 'paper' else 2
        self.max_rnn = 2
        for i in range(self.max_cnn):
            for name, values in [('padding', ('same', 'valid')), ('filters', units), ('kernel', (2, 3, 4, 5)),
                                  ('pool', (1, 2, 3, 4)), ('activation', activation)]:
                self.fields.append(Field(f'cnn.{i}.{name}', values))
        self.fields.append(Field('rnn_count', (1, 2)))
        for i in range(self.max_rnn):
            for name, values in [('kind', ('simrnn', 'lstm', 'gru', 'linear')), ('units', units),
                                  ('activation', activation), ('regularization', ('none', 'l1', 'l2', 'l1_l2')),
                                  ('dropout', (0., .1, .2, .3))]:
                self.fields.append(Field(f'rnn.{i}.{name}', values))
        self.length = sum(f.width for f in self.fields)

    def decode(self, bits: np.ndarray) -> dict:
        bits = np.asarray(bits)
        if bits.shape != (self.length,) or not np.isin(bits, [0, 1]).all():
            raise ValueError(f'Expected {self.length} binary bits')
        values, offset = {}, 0
        for f in self.fields:
            index = int(''.join(str(int(b)) for b in bits[offset:offset + f.width]), 2)
            values[f.name] = f.values[index]
            offset += f.width
        architecture = {k: values[k] for k in ['batch_size', 'optimizer', 'learning_rate']}
        for section in ['cnn', 'rnn']:
            architecture[section] = [{f.name.split('.')[-1]: values[f.name] for f in self.fields
                                      if f.name.startswith(f'{section}.{i}.')} for i in range(values[f'{section}_count'])]
        return architecture

    def encode(self, architecture: dict) -> np.ndarray:
        if set(architecture) != {'batch_size', 'optimizer', 'learning_rate', 'cnn', 'rnn'}:
            raise ValueError('Architecture requires exactly batch_size, optimizer, learning_rate, cnn and rnn')
        values = {k: architecture[k] for k in ['batch_size', 'optimizer', 'learning_rate']}
        for section, maximum in [('cnn', self.max_cnn), ('rnn', self.max_rnn)]:
            if not 1 <= len(architecture[section]) <= maximum:
                raise ValueError(f'Invalid {section} layer count')
            values[f'{section}_count'] = len(architecture[section])
            for i, layer in enumerate(architecture[section]):
                required = {f.name.split('.')[-1] for f in self.fields if f.name.startswith(f'{section}.{i}.')}
                if set(layer) != required:
                    raise ValueError(f'Active {section} layer {i} requires fields {sorted(required)}')
                values.update({f'{section}.{i}.{k}': v for k, v in layer.items()})
        output = []
        for f in self.fields:
            val = values.get(f.name, f.values[0])
            index = f.values.index(val)
            output.extend(int(b) for b in format(index, f'0{f.width}b'))
        return np.asarray(output, dtype=np.int8)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=self.length, dtype=np.int8)
