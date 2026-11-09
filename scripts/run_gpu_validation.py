"""Measure the complete test suite and bounded CUDA CLI checks on local data.

Run from the repository root with GPU access. All artifacts use a fresh directory.
These small integration budgets do not reproduce the paper's experiments.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import torch

from src.utils.core import environment, read_config, save_json


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required; no CPU fallback is permitted.')
    # Exercise a CUDA kernel before recording the device as available.
    assert (torch.ones(8, device='cuda') * 2).sum().item() == 16
    torch.cuda.synchronize()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path('results/gpu_validation') / stamp
    root.mkdir(parents=True)
    env = os.environ.copy()
    env['MPLCONFIGDIR'] = '/tmp/bpso-gpu-mpl'
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    save_json(root / 'environment.json', {
        **environment(), 'device_properties': str(torch.cuda.get_device_properties(0)),
        'nvidia_smi': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        'scope': 'Full CPU/CUDA test suites plus bounded GPU integration; not paper reproduction',
    })
    records = []
    began_all = time.perf_counter()

    def execute(label: str, args: list[str], overrides: dict | None = None) -> None:
        log = root / f'{label}.log'
        began = time.perf_counter()
        with log.open('w') as stream:
            try:
                result = subprocess.run([sys.executable, *args], stdout=stream,
                                        stderr=subprocess.STDOUT, env={**env, **(overrides or {})},
                                        timeout=1800)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
        record = {'case': label, 'exit_code': code, 'wall_seconds': time.perf_counter() - began,
                  'command': [sys.executable, *args], 'environment_overrides': overrides or {},
                  'log': str(log)}
        records.append(record)
        save_json(root / 'commands.json', records)
        print(f'{label}: {"PASS" if code == 0 else "FAIL"} ({record["wall_seconds"]:.2f}s)', flush=True)
        if code:
            print(log.read_text()[-3000:], flush=True)

    tests = []
    for device in ['cpu', 'cuda']:
        xml = root / f'pytest_{device}.xml'
        execute(f'pytest_{device}', ['-m', 'pytest', '-q', '--durations=0', f'--junitxml={xml}'],
                {'BPSO_TEST_DEVICE': device})
        if xml.exists():
            for case in ET.parse(xml).iter('testcase'):
                tests.append({'device': device, 'class': case.get('classname'), 'name': case.get('name'),
                              'seconds': float(case.get('time', 0)),
                              'status': 'failed' if case.find('failure') is not None else
                                        'error' if case.find('error') is not None else
                                        'skipped' if case.find('skipped') is not None else 'passed'})
    save_json(root / 'test_cases.json', tests)
    if tests:
        with (root / 'test_cases.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tests[0]))
            writer.writeheader()
            writer.writerows(tests)

    profiles = [('gas_pipeline', 'configs/gas_baselines_smoke.yaml'),
                *[(name, f'configs/synthetic/{name}.yaml') for name in ['ton_iot', 'bot_iot', 'swat']]]
    metrics = []
    for name, source in profiles:
        cfg = read_config(source)
        cfg['training']['device'] = 'cuda'
        cfg['data']['processed'] = str(root / name / 'processed')
        path = root / name / 'config.json'
        save_json(path, cfg)
        diagnostic = read_config(path)
        diagnostic['compute'] = {'max_epochs': 12}
        diagnostic_path = root / name / 'diagnostic_config.json'
        save_json(diagnostic_path, diagnostic)
        for stage in ['prepare', 'train', 'search', 'diagnostics', 'compare', 'report']:
            args = ['-m', 'src.cli', stage]
            if stage == 'report':
                args += ['--input', str(root / name / 'train'), '--output', str(root / name / 'reports')]
            else:
                args += ['--config', str(diagnostic_path if stage == 'diagnostics' else path)]
                if stage != 'prepare':
                    args += ['--output', str(root / name / stage)]
            execute(f'{name}_{stage}', args)
        execute(f'{name}_search_resume', ['-m', 'src.cli', 'search', '--config', str(path),
                                        '--output', str(root / name / 'search')])
        execute(f'{name}_selected_train', ['-m', 'src.cli', 'train', '--config', str(path),
                                         '--architecture', str(root / name / 'search/best_architecture.json'),
                                         '--output', str(root / name / 'selected_train')])
        execute(f'{name}_train_resume', ['-m', 'src.cli', 'train', '--config', str(path),
                                        '--output', str(root / name / 'train')])
        metric_path = root / name / 'train/test_metrics.json'
        if metric_path.exists():
            metrics.append({'dataset': name, **json.loads(metric_path.read_text())})

    trajectories = []
    for path in root.rglob('trajectory.json'):
        rows = json.loads(path.read_text())
        if isinstance(rows, list) and rows and 'device' in rows[0]:
            trajectories.append({'path': str(path), 'epochs': len(rows),
                                 'devices': sorted({r['device'] for r in rows}),
                                 'training_seconds': sum(r['training_seconds'] for r in rows),
                                 'validation_seconds': sum(r['validation_seconds'] for r in rows),
                                 'gpu_seconds': rows[-1]['gpu_seconds']})
    save_json(root / 'gpu_training_measurements.json', trajectories)
    save_json(root / 'held_out_metrics.json', metrics)
    summary = {'directory': str(root), 'wall_seconds': time.perf_counter() - began_all,
               'commands_passed': sum(r['exit_code'] == 0 for r in records),
               'commands_failed': sum(r['exit_code'] != 0 for r in records),
               'tests': {device: {status: sum(t['device'] == device and t['status'] == status for t in tests)
                                  for status in ['passed', 'failed', 'error', 'skipped']}
                         for device in ['cpu', 'cuda']},
               'gpu_training_runs': len(trajectories),
               'all_integration_training_on_cuda': bool(trajectories) and all(
                   all(d.startswith('cuda') for d in run['devices']) for run in trajectories),
               'scope': 'Bounded integration checks; real Gas Pipeline and three synthetic profiles; not paper reproduction'}
    save_json(root / 'summary.json', summary)
    lines = ['# GPU validation measurements', '', summary['scope'], '',
             f'GPU: {torch.cuda.get_device_name(0)}', '',
             '| Test suite | Passed | Failed/errors | Skipped |', '|---|---:|---:|---:|']
    for device, counts in summary['tests'].items():
        lines.append(f'| {device} | {counts["passed"]} | {counts["failed"] + counts["error"]} | {counts["skipped"]} |')
    lines += ['', '| CLI/test case | Status | Wall seconds |', '|---|---|---:|']
    lines += [f'| {r["case"]} | {"PASS" if r["exit_code"] == 0 else "FAIL"} | {r["wall_seconds"]:.3f} |' for r in records]
    lines += ['', '| Dataset | Synthetic | Test accuracy | Macro F1 | Train + validation seconds | Inference ms/sample |',
              '|---|---|---:|---:|---:|---:|']
    lines += [f'| {m["dataset"]} | {m["synthetic_data"]} | {m["accuracy"]:.6f} | {m["macro_f1"]:.6f} | '
              f'{m["training_and_validation_seconds"]:.3f} | {m["inference_seconds_per_sample"] * 1000:.6f} |' for m in metrics]
    lines += ['', 'Per-test timings: `test_cases.csv`. Per-class metrics and confusion matrices: `<dataset>/train/`.',
              'Search ledgers, checkpoints and synchronized CUDA timings are retained under each dataset.',
              'Small, two-epoch fixed models are software checks; these scores do not establish detector quality.']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    save_json('results/gpu_validation/latest.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if summary['commands_failed'] or not summary['all_integration_training_on_cuda']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
