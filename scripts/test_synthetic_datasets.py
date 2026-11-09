"""Exercise all three synthetic profiles through actual CLI commands.

Raw generation is separate so users can inspect the manifests before running.
No original-dataset files or official configurations are modified.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from src.utils.core import read_config, save_json


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path('results/synthetic_datasets') / stamp
    root.mkdir(parents=True)
    save_json(root / 'DISCLAIMER.json', {'synthetic_data': True, 'research_results': False,
              'purpose': 'Schema-inspired software validation, not original datasets or paper reproduction'})
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR', '/tmp/bpso-mpl')
    commands, datasets = [], []
    for name in ['ton_iot', 'bot_iot', 'swat']:
        cfg = read_config(f'configs/synthetic/{name}.yaml')
        cfg['data']['processed'] = str(root / name / 'processed')
        cfg_path = root / name / 'config.json'
        save_json(cfg_path, cfg)
        diagnostic = dict(cfg)
        diagnostic['compute'] = {'max_epochs': 12}
        diag_path = root / name / 'diagnostic_config.json'
        save_json(diag_path, diagnostic)
        for stage in ['prepare', 'train', 'search', 'diagnostics', 'compare', 'report']:
            args = [sys.executable, '-m', 'src.cli', stage]
            if stage == 'report':
                # Comparison already exports its own search tables/curves. Report the
                # fixed model separately to avoid treating the repeated BPSO seed as independent.
                args += ['--input', str(root / name / 'train'), '--output', str(root / name / 'reports')]
            else:
                args += ['--config', str(diag_path if stage == 'diagnostics' else cfg_path)]
                if stage != 'prepare':
                    args += ['--output', str(root / name / stage)]
            log = root / name / f'{stage}.log'
            began = time.perf_counter()
            with log.open('w') as stream:
                result = subprocess.run(args, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=300)
            commands.append({'dataset': name, 'stage': stage, 'command': args, 'exit_code': result.returncode,
                             'wall_seconds': time.perf_counter()-began, 'log': str(log)})
            save_json(root / 'commands.json', commands)
            print(f'{name}/{stage}: exit{result.returncode}', flush=True)
            if result.returncode:
                print(log.read_text()[-4000:], flush=True)
                raise SystemExit(result.returncode)
        metrics = json.loads((root / name / 'train/test_metrics.json').read_text())
        datasets.append({'name': name, 'synthetic_data': True, 'fixed_model_metrics': metrics,
                         'comparison': json.loads((root / name / 'compare/reports/summary.json').read_text())})
    summary = {'synthetic_data': True, 'research_results': False, 'commands_passed': len(commands),
               'datasets': datasets, 'directory': str(root)}
    save_json(root / 'synthetic_validation_summary.json', summary)
    save_json('results/synthetic_datasets/latest.json', {'directory': str(root), 'synthetic_data': True,
              'commands_passed': len(commands), 'summary': str(root / 'synthetic_validation_summary.json')})
    print(f'Completed{len(commands)} synthetic-only CLI checks. Artifacts:{root}', flush=True)


if __name__ == '__main__':
    main()
