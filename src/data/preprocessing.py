"""Local official-data adapters and train-only preprocessing.

CSV, ARFF and Excel are supported. File revisions and schema are configured,
never guessed from an unofficial replacement. IDs refer to original file rows.
"""
from __future__ import annotations
import glob
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, Normalizer, OneHotEncoder
from src.utils.core import digest, save_json

PROFILES = {
    'gas_pipeline': {'target': 'class', 'normalization': 'column_l2', 'drop': ['binary result', 'categorized result', 'specific result', 'label', 'attack', 'result', 'timestamp']},
    'ton_iot': {'target': 'type', 'normalization': 'minmax', 'drop': ['label', 'ts', 'src_ip', 'dst_ip']},
    'bot_iot': {'target': 'category', 'normalization': 'minmax', 'drop': ['attack', 'subcategory', 'pkSeqID', 'seq', 'stime', 'ltime', 'saddr', 'daddr']},
    'swat': {'target': 'Normal/Attack', 'normalization': 'minmax', 'drop': ['Timestamp']},
}


def load_files(patterns: list[str], read_options: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    paths = set()
    for pattern in patterns:
        matches = [Path(p).resolve() for p in glob.glob(pattern) if Path(p).is_file()]
        if not matches:
            raise FileNotFoundError(f'No local dataset files match {pattern!r}. See docs/datasets.md.')
        paths.update(matches)
    paths = sorted(paths)
    if not paths:
        raise FileNotFoundError(f'No local dataset files match {patterns}. See docs/datasets.md.')
    frames, sources = [], []
    for filename in paths:
        path = Path(filename)
        options = read_options or {}
        if path.suffix.lower() == '.arff':
            frame = pd.DataFrame(arff.loadarff(path)[0])
            for col in frame:
                frame[col] = frame[col].map(lambda x: x.decode() if isinstance(x, bytes) else x)
        elif path.suffix.lower() in {'.xlsx', '.xls'}:
            frame = pd.read_excel(path, **options)
        else:
            frame = pd.read_csv(path, **options)
        frame.columns = frame.columns.str.strip()
        if frame.columns.duplicated().any():
            raise ValueError('Duplicate column names after whitespace removal')
        frame.index = [f'{path.resolve()}:{i}' for i in range(len(frame))]
        frames.append(frame)
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        sources.append({'path': str(path.resolve()), 'sha256': h.hexdigest(), 'rows': len(frame)})
    if any(set(f.columns) != set(frames[0].columns) for f in frames):
        raise ValueError('Files have incompatible schemas; configure datasets separately')
    return pd.concat(frames), sources


def partition(frame: pd.DataFrame, target: str, fraction: float, seed: int, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    mode = cfg.get('split', 'stratified')
    if not 0 < fraction < 1:
        raise ValueError('Split fraction must lie in (0,1)')
    if mode == 'chronological':
        column = cfg.get('time_column')
        if not column:
            raise ValueError('Chronological split requires time_column')
        frame = frame.sort_values(column, kind='stable')
        cut = int(len(frame) * (1 - fraction))
        return frame.iloc[:cut], frame.iloc[cut:]
    if mode == 'group':
        groups = frame[cfg['group_column']]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed).split(frame, groups=groups))
        return frame.iloc[a], frame.iloc[b]
    if mode != 'stratified':
        raise ValueError(f'Unknown split mode: {mode}')
    a, b = train_test_split(frame, test_size=fraction, random_state=seed, stratify=frame[target])
    return a, b


def overlap_diagnostics(parts: list[pd.DataFrame], features: list[str], target: str) -> dict:
    """Count shared unique content hashes, independently of target agreement.

    Label-inclusive overlap alone misses predictors duplicated with contradictory
    labels. These checks report content overlap; they cannot exclude dependence.
    """
    names = ['train', 'val', 'test']
    predictors = [set(pd.util.hash_pandas_object(p[features], index=False)) for p in parts]
    labelled = [set(pd.util.hash_pandas_object(p[features + [target]], index=False)) for p in parts]
    result = {'policy': 'report; use group/chronological splits for correlated records',
              'count_unit': 'unique pandas content hashes; not affected row counts'}
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        suffix = f'{names[i]}_{names[j]}'
        result[f'exact_row_overlap_{suffix}'] = len(labelled[i] & labelled[j])
        result[f'predictor_overlap_{suffix}'] = len(predictors[i] & predictors[j])
    combined = pd.concat(parts)
    hashes = pd.util.hash_pandas_object(combined[features], index=False).to_numpy()
    result['predictor_hashes_with_conflicting_labels'] = int(
        pd.DataFrame({'hash': hashes, 'target': combined[target].to_numpy()})
        .groupby('hash')['target'].nunique().gt(1).sum())
    return result


def prepare(cfg: dict, output: str | Path) -> dict:
    profile = PROFILES[cfg['name']]
    target = cfg.get('target', profile['target'])
    frame, sources = load_files(cfg['files'], cfg.get('read_options'))
    if target not in frame:
        raise ValueError(f'Missing target {target!r}. Available columns: {list(frame)}')
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        if frame[target].isna().any():
            raise ValueError('Missing labels: resolve explicitly rather than silently dropping rows')
        frame[target] = frame[target].astype(str).str.strip().replace(cfg.get('label_map', {}))
        return frame
    frame = clean(frame)
    if cfg.get('time_column') and cfg.get('time_format'):
        frame[cfg['time_column']] = pd.to_datetime(frame[cfg['time_column']], format=cfg['time_format'], errors='raise')
    seed = cfg.get('split_seed', 42)
    if cfg.get('test_files'):
        test, extra = load_files(cfg['test_files'], cfg.get('read_options'))
        test = clean(test)
        if cfg.get('time_column') and cfg.get('time_format'):
            test[cfg['time_column']] = pd.to_datetime(test[cfg['time_column']], format=cfg['time_format'], errors='raise')
        sources.extend(extra)
        if set(frame.index) & set(test.index):
            raise ValueError('Official train and test sources overlap')
        train, val = partition(frame, target, cfg.get('validation_fraction', .2), seed, cfg)
    else:
        trainval, test = partition(frame, target, cfg.get('test_fraction', .2), seed, cfg)
        train, val = partition(trainval, target, cfg.get('validation_fraction', .2) / (1 - cfg.get('test_fraction', .2)), seed + 1, cfg)
    encoder = LabelEncoder().fit(train[target])
    ys = [encoder.transform(part[target]) for part in (train, val, test)]  # unseen test classes fail explicitly
    drop = set(profile['drop'] + cfg.get('drop_columns', []) + [target])
    drop.update([cfg.get('time_column'), cfg.get('group_column')])
    features = cfg.get('features') or [c for c in train if c not in drop]
    if set(features) & drop:
        raise ValueError('Feature allowlist includes target, alternate target, ID or split column')
    if not features:
        raise ValueError('No predictor columns')
    def predictors(part: pd.DataFrame) -> pd.DataFrame:
        values = part[features].replace([np.inf, -np.inf, '?', 'Infinity', '-Infinity'], np.nan).copy()
        for col in cfg.get('numeric_columns', []):
            values[col] = pd.to_numeric(values[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
        return values
    raw = [predictors(p) for p in (train, val, test)]
    numeric = raw[0].select_dtypes(include='number').columns.tolist()
    categorical = [c for c in features if c not in numeric]
    for part in raw:
        for col in categorical:
            part[col] = part[col].map(lambda x: str(x) if pd.notna(x) else np.nan)
    transform = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
        ('categorical', Pipeline([('missing', SimpleImputer(strategy='constant', fill_value='__MISSING__')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)
    ], sparse_threshold=0)
    xs = [transform.fit_transform(raw[0])]
    xs.extend(transform.transform(p) for p in raw[1:])
    normalization = cfg.get('normalization', profile['normalization'])
    norm = None
    scales = None
    if normalization == 'minmax':
        norm = MinMaxScaler().fit(xs[0])
        xs = [norm.transform(x) for x in xs]
    elif normalization == 'column_l2':
        scales = np.linalg.norm(xs[0], axis=0)
        scales[scales == 0] = 1
        xs = [x / scales for x in xs]
    elif normalization == 'row_l2':
        norm = Normalizer().fit(xs[0])
        xs = [norm.transform(x) for x in xs]
    else:
        raise ValueError('normalization must be minmax, column_l2 or row_l2')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'config': cfg, 'sources': sources, 'classes': encoder.classes_.tolist(), 'features': features,
                'encoded_features': transform.get_feature_names_out().tolist(), 'normalization': normalization,
                'splits': {name: {'rows': len(part), 'classes': part[target].value_counts().to_dict(),
                                  'ids_hash': digest(part.index.tolist())} for name, part in zip(['train', 'val', 'test'], [train, val, test])}}
    metadata['fingerprint'] = digest(metadata)
    save_json(output / 'metadata.json', metadata)
    counts = pd.DataFrame({n: p[target].value_counts() for n, p in zip(['train', 'val', 'test'], [train, val, test])}).fillna(0).astype(int)
    counts.to_csv(output / 'class_distribution.csv')
    save_json(output / 'imbalance.json', {'train_majority_minority_ratio': float(counts['train'].max() / counts['train'].min()),
                                          'note': 'No class resampling or balancing applied'})
    save_json(output / 'split_indices.json', {n: p.index.tolist() for n, p in zip(['train', 'val', 'test'], [train, val, test])})
    # Avoid accidental train-test duplicates, but do not silently alter official data.
    save_json(output / 'overlap_diagnostics.json', overlap_diagnostics([train, val, test], features, target))
    joblib.dump({'transform': transform, 'normalizer': norm, 'column_scales': scales, 'labels': encoder,
                 'features': features, 'numeric_columns': cfg.get('numeric_columns', [])}, output / 'preprocessor.joblib')
    for name, x, y in zip(['train', 'val', 'test'], xs, ys):
        x = np.asarray(x, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite processed features')
        np.savez_compressed(output / f'{name}.npz', x=x, y=y.astype(np.int64))
    return metadata


def transform_new(artifact: str | Path, frame: pd.DataFrame) -> np.ndarray:
    """Apply the saved training transform to unlabelled samples without refitting."""
    saved = joblib.load(artifact)
    frame = frame.copy()
    frame.columns = frame.columns.str.strip()
    values = frame[saved['features']].replace([np.inf, -np.inf, '?', 'Infinity', '-Infinity'], np.nan)
    for col in saved['numeric_columns']:
        values[col] = pd.to_numeric(values[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
    categorical = saved['transform'].transformers_[1][2]
    for col in categorical:
        values[col] = values[col].map(lambda x: str(x) if pd.notna(x) else np.nan)
    x = saved['transform'].transform(values)
    if saved['normalizer'] is not None:
        x = saved['normalizer'].transform(x)
    elif saved['column_scales'] is not None:
        x = x / saved['column_scales']
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite transformed values')
    return np.asarray(x, dtype=np.float32)
