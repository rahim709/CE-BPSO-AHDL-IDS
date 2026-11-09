from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import prepare


@pytest.fixture
def fixture_data(tmp_path):
    """Artificial software fixture; never represents any IDS dataset."""
    rng = np.random.default_rng(5)
    n = 90
    frame = pd.DataFrame({'sensor': rng.normal(size=n), 'pressure': rng.normal(size=n),
                          'protocol': np.tile(['a', 'b', 'c'], 30), 'class': np.tile(['normal', 'attack', 'rare'], 30)})
    path = tmp_path / 'software_fixture.csv'
    frame.to_csv(path, index=False)
    cfg = {'name': 'gas_pipeline', 'files': [str(path)], 'target': 'class', 'normal_label': 'normal',
           'normalization': 'minmax', 'split_seed': 17}
    out = tmp_path / 'processed'
    prepare(cfg, out)
    return out, cfg


@pytest.fixture
def architecture():
    return {'batch_size': 8, 'optimizer': 'adam', 'learning_rate': .001,
            'cnn': [{'padding': 'same', 'filters': 4, 'kernel': 2, 'pool': 1, 'activation': 'relu'}],
            'rnn': [{'kind': 'gru', 'units': 4, 'activation': 'tanh', 'regularization': 'none', 'dropout': .2}]}
