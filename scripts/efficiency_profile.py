"""Bounded engine profiling and one-factor GPU experiments; no test data."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from src.training.engine import train_to
from src.utils.core import save_json, environment

ROOT = Path('results/ua_amf_efficiency_20261108')
SOURCE = Path('results/ua_amf_prediction_20261108')


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT))
    parser.add_argument('--trace-only', action='store_true')
    parser.add_argument('--complete-missing', action='store_true')
    args = parser.parse_args()
    ROOT = Path(args.output)
    if (ROOT / 'profile_results_completed.json').exists() and not args.trace_only:
        print('Using saved verified profiles; choose a fresh --output to rerun timings.')
        return
    if args.trace_only and (ROOT / 'original_profiler.json').exists():
        print('Using saved original trace; choose a fresh --output to repeat profiling.')
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((SOURCE / 'resolved_config.json').read_text())
    cohort = json.loads((SOURCE / 'cohort_definitions.json').read_text())
    if args.trace_only:
        arch = cohort[10]['architecture']
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            train_to(arch, cfg['data']['processed'], ROOT / 'original_trace', 3, 981, cfg['training'])
        records = [{'name': e.key, 'count': e.count, 'cpu_time_total_us': e.cpu_time_total,
                    'self_device_time_total_us': getattr(e, 'self_device_time_total', 0.)} for e in prof.key_averages()]
        save_json(ROOT / 'original_profiler.json', records)
        return
    profiles = {
        'legacy': None,
        'vectorized': {'vectorized': True},
        'resident': {'vectorized': True, 'resident': True},
        'sparse_validation': {'validation_epochs': [3, 10, 30]},
        'deferred_scalars': {'defer_scalar_sync': True},
        'pinned': {'pin_memory': True},
        'amp': {'amp': True},
        'combined': {'vectorized': True, 'resident': True, 'validation_epochs': [3, 10, 30], 'defer_scalar_sync': True, 'cache_data': True, 'cache_model': True},
    }
    save_json(ROOT / 'profile_protocol.json', {'development_indices': [0, 7, 10], 'seed': 982, 'epochs': 30,
              'profiles': profiles, 'wall_cap': 180, 'selection_rule': 'Choose fastest finite FP32 combined profile only if final model tensors exactly equal legacy for all three development architectures; AMP rejected if drift >.01 or failure.'})
    save_json(ROOT / 'environment.json', environment())
    records = json.loads((ROOT / 'profile_results.json').read_text()) if args.complete_missing else []
    deadline = time.perf_counter() + (30 if args.complete_missing else 180)
    try:
        for index in [0, 7, 10]:
            for name in profiles:
                if any(r['index']==index and r['profile']==name for r in records):
                    continue
                path = ROOT / 'profiles' / name / f'arch_{index}'
                begin = time.perf_counter()
                from src.training.runtime import clear_caches
                clear_caches()
                runtime = profiles[name]
                try:
                    row = train_to(cohort[index]['architecture'], cfg['data']['processed'], path, 30, 982, cfg['training'], deadline=deadline, runtime=runtime)
                    weights = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)['model']
                    legacy = torch.load(ROOT / 'profiles/legacy' / f'arch_{index}/last.pt', map_location='cpu', weights_only=False)['model']
                    exact = all(torch.equal(weights[k], legacy[k]) for k in weights)
                    records.append({'index': index, 'profile': name, 'wall_seconds': time.perf_counter()-begin,
                                    'gpu_seconds': row['gpu_seconds'], 'macro_f1': row['macro_f1'], 'exact_weights': exact,
                                    'peak_gpu_memory_bytes': row['peak_gpu_memory_bytes'], 'failure': None})
                except TimeoutError:
                    raise
                except (RuntimeError, FloatingPointError) as exc:
                    records.append({'index': index, 'profile': name, 'failure': str(exc)})
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                save_json(ROOT / ('profile_results_completed.json' if args.complete_missing else 'profile_results.json'), records)
    except TimeoutError:
        save_json(ROOT / 'profile_stop.json', {'partial': True, 'reason': '180-second profile cap'})
    save_json(ROOT / ('profile_results_completed.json' if args.complete_missing else 'profile_results.json'), records)
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    main()
