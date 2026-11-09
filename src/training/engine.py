"""Deterministic cumulative-epoch training; checkpoints contain RNG and optimizer."""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluation.metrics import classification_metrics, majority_baseline
from src.models.hybrid import make_optimizer
from src.models.controls import make_model
from src.training.losses import class_weights
from src.utils.core import digest, save_json, seed_all


def resolve_device(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(name)


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load_partition(directory: str | Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(directory) / f'{split}.npz') as data:
        return data['x'], data['y']


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(x), batch_size):
        chunks.append(model(torch.as_tensor(x[start:start + batch_size], device=device)).softmax(1).cpu().numpy())
    return np.concatenate(chunks)


def train_to(architecture: dict, data: str | Path, run: str | Path, budget: int, seed: int,
             cfg: dict, deadline: float | None = None, runtime: dict | None = None, execution_budget: float | None = None) -> dict:
    """Train to cumulative epoch budget, resume exactly, never open test.npz."""
    runtime = runtime or {}
    if budget < 1:
        raise ValueError('Budget must be positive')
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((Path(data) / 'metadata.json').read_text())
    identity_fields = {'architecture': architecture, 'data': metadata['fingerprint'], 'seed': seed, 'training': cfg}
    if runtime.get('amp'):
        identity_fields['precision'] = 'amp_fp16'
    identity = digest(identity_fields)
    device = resolve_device(cfg.get('device', 'auto'))
    seed_all(seed, cfg.get('threads', 1))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    from src.training import runtime as execution
    load = execution.load_arrays if runtime.get('cache_data') else load_partition
    x, y = load(data, 'train')
    vx, vy = load(data, 'val')
    weights = class_weights(y, len(metadata['classes']), cfg.get('class_weighting', 'none'))
    loss_weights = weights.to(device) if weights is not None else None
    baseline = majority_baseline(y, vy, metadata['classes'])
    checkpoint = run / 'last.pt'
    session_key = (str(run.resolve()), identity)
    session = execution.SESSIONS.get(session_key) if runtime.get('cache_model') else None
    if session and (not checkpoint.exists() or session['checkpoint_mtime'] != checkpoint.stat().st_mtime_ns):
        session = None
    if session is None:
        execution.SESSIONS.clear()
        model = make_model(architecture, x.shape[1], len(metadata['classes'])).to(device)
        optimizer = make_optimizer(model, architecture)
    else:
        model, optimizer = session['model'], session['optimizer']
    scaler = torch.amp.GradScaler('cuda', enabled=bool(runtime.get('amp')) and device.type == 'cuda')
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=architecture['batch_size'], shuffle=True, generator=generator, num_workers=0,
                        pin_memory=bool(runtime.get('pin_memory')) and device.type == 'cuda')
    if runtime.get('vectorized'):
        tx, ty = torch.from_numpy(x), torch.from_numpy(y)
        if runtime.get('resident'):
            tx, ty = execution.tensors(x, y, (str(Path(data).resolve()), metadata['fingerprint'], 'train'), device)
            vx, _ = execution.tensors(vx, vy, (str(Path(data).resolve()), metadata['fingerprint'], 'val'), device)
        loader = execution.vector_loader(tx, ty, architecture['batch_size'], generator,
                                         bool(runtime.get('pin_memory')) and not runtime.get('resident') and device.type == 'cuda')
    rows, start_epoch, elapsed = [], 0, 0.
    checkpoint = run / 'last.pt'
    if checkpoint.exists():
        saved = session['saved'] if session else torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError(f'Checkpoint configuration/data mismatch: {run}')
        if session is None:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
        if 'scaler' in saved:
            scaler.load_state_dict(saved['scaler'])
        start_epoch, rows, elapsed = saved['epoch'], saved['rows'], saved['elapsed']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        generator.set_state(saved['loader_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    save_json(run / 'architecture.json', architecture)
    invocation_elapsed = elapsed
    for epoch in range(start_epoch + 1, budget + 1):
        if execution_budget is not None and device.type == 'cuda' and elapsed - invocation_elapsed >= execution_budget:
            raise TimeoutError('GPU execution time budget reached at an epoch boundary')
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('Search time budget reached at an epoch boundary')
        sync(device)
        began = time.perf_counter()
        model.train()
        repairs_before = sum(block.repairs for block in model.cnn)
        loss_sum = 0.
        deferred_loss = torch.zeros((), dtype=torch.float64, device=device) if runtime.get('defer_scalar_sync') else None
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=bool(runtime.get('pin_memory'))), yb.to(device, non_blocking=bool(runtime.get('pin_memory')))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=bool(runtime.get('amp')) and device.type == 'cuda'):
                loss = nn.functional.cross_entropy(model(xb), yb, weight=loss_weights) + model.penalty(cfg.get('l1', 1e-5), cfg.get('l2', 1e-5))
            if deferred_loss is None and not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward()
            if cfg.get('gradient_clip'):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            if deferred_loss is None:
                loss_sum += loss.item() * len(yb)
            else:
                deferred_loss += loss.detach().to(torch.float64) * len(yb)
        if deferred_loss is not None:
            loss_sum = deferred_loss.item()
            if not np.isfinite(loss_sum):
                raise FloatingPointError('Nonfinite training loss')
        sync(device)
        training_seconds = time.perf_counter() - began
        vbegan = time.perf_counter()
        validate = not runtime.get('validation_epochs') or epoch in runtime['validation_epochs'] or epoch == budget
        probabilities = predict(model, vx, device) if validate else None
        sync(device)
        metrics = classification_metrics(vy, probabilities.argmax(1), metadata['classes']) if validate else None
        validation_seconds = time.perf_counter() - vbegan if validate else 0.
        elapsed += training_seconds + validation_seconds
        row = {'epoch': epoch, 'loss': loss_sum / len(y), 'macro_f1': metrics['macro_f1'] if metrics else None,
               'per_class_recall': {c: m['recall'] for c, m in metrics['per_class'].items()} if metrics else {},
               'training_seconds': training_seconds, 'validation_seconds': validation_seconds,
               'elapsed_seconds': elapsed, 'gpu_seconds': elapsed if device.type == 'cuda' else 0.,
               'device': str(device), 'metrics': metrics,
               'parameters': sum(p.numel() for p in model.parameters()),
               'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
               'class_weighting': cfg.get('class_weighting', 'none'),
               'class_weights': weights.tolist() if weights is not None else None,
               'majority_baseline': baseline,
               'beats_majority_macro_f1': bool(metrics['macro_f1'] > baseline['macro_f1']) if metrics else None,
               'shape_repairs_this_epoch': sum(block.repairs for block in model.cnn) - repairs_before,
               'quality_diagnostics': {'collapsed_to_single_class': metrics['collapsed_to_single_class'] if metrics else None,
                                       'predicted_class_count': metrics['predicted_class_count'] if metrics else None,
                                       'zero_recall_classes': metrics['zero_recall_classes'] if metrics else None}}
        rows.append(row)
        logging.getLogger(__name__).info(json.dumps({'event': 'epoch', 'run': str(run), 'epoch': epoch,
                                                   'macro_f1': row['macro_f1'], 'elapsed_seconds': elapsed}))
        if validate and (epoch in cfg.get('prediction_epochs', [3, 10, 30, 200]) or epoch == budget):
            np.savez_compressed(run / f'validation_epoch_{epoch}.npz', y=vy, probabilities=probabilities)
        saved = {'identity': identity, 'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'rows': rows, 'elapsed': elapsed, 'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                 'torch_rng': torch.get_rng_state(), 'loader_rng': generator.get_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                 'architecture': architecture, 'metadata_fingerprint': metadata['fingerprint'], 'seed': seed, 'scaler': scaler.state_dict()}
        torch.save(saved, run / 'last.pt.tmp')
        os.replace(run / 'last.pt.tmp', checkpoint)
        save_json(run / 'trajectory.json', rows)
        if runtime.get('cache_model'):
            execution.SESSIONS[session_key] = {'model': model, 'optimizer': optimizer, 'saved': saved,
                                               'checkpoint_mtime': checkpoint.stat().st_mtime_ns}
    eligible = [row for row in rows if row['epoch'] == budget]
    if not eligible:
        raise RuntimeError('Requested epoch not available')
    return eligible[0]


def final_evaluate(architecture: dict, data: str | Path, run: str | Path, budget: int,
                   seed: int, cfg: dict, normal_label: str | None = None) -> dict:
    """Dedicated restart directory; test evaluation only after fixed final budget."""
    row = train_to(architecture, data, run, budget, seed, cfg)
    run = Path(run)
    meta = json.loads((Path(data) / 'metadata.json').read_text())
    x, y = load_partition(data, 'test')
    device = resolve_device(cfg.get('device', 'auto'))
    model = make_model(architecture, x.shape[1], len(meta['classes'])).to(device)
    checkpoint = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != budget:
        raise ValueError('Final checkpoint is past requested budget; use a dedicated final run directory')
    model.load_state_dict(checkpoint['model'])
    # Warm up separately, then report complete batched held-out inference.
    predict(model, x[:min(32, len(x))], device)
    sync(device)
    began = time.perf_counter()
    probabilities = predict(model, x, device)
    sync(device)
    inference = time.perf_counter() - began
    result = classification_metrics(y, probabilities.argmax(1), meta['classes'], normal_label)
    training_targets = load_partition(data, 'train')[1]
    result['majority_baseline'] = majority_baseline(training_targets, y, meta['classes'])
    result['beats_majority_macro_f1'] = bool(result['macro_f1'] > result['majority_baseline']['macro_f1'])
    result['majority_baseline_macro_f1'] = result['majority_baseline']['macro_f1']
    result['majority_baseline_accuracy'] = result['majority_baseline']['accuracy']
    result['zero_recall_class_count'] = len(result['zero_recall_classes'])
    result['class_weighting'] = cfg.get('class_weighting', 'none')
    result.update({'inference_seconds': inference, 'inference_seconds_per_sample': inference / len(y),
                   'training_and_validation_seconds': row['elapsed_seconds'], 'epochs': budget,
                   'seed': seed, 'data_fingerprint': meta['fingerprint'],
                   'synthetic_data': bool(meta.get('config', {}).get('synthetic', False)),
                   'parameters': sum(p.numel() for p in model.parameters())})
    np.savez_compressed(run / 'test_predictions.npz', y=y, probabilities=probabilities)
    save_json(run / 'test_metrics.json', result)
    import pandas as pd
    pd.DataFrame(result['per_class']).T.to_csv(run / 'per_class.csv')
    pd.DataFrame(result['confusion_matrix'], index=meta['classes'], columns=meta['classes']).to_csv(run / 'confusion_matrix.csv')
    return result
