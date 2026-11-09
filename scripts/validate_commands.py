"""Exercise CLI commands using explicitly artificial software data.

Official dataset preparation commands are checked separately and recorded as
blocked when raw files are absent. Fixture metrics are never research results.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from src.utils.core import environment, read_config, save_json


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root = Path('results/command_validation') / stamp
    root.mkdir(parents=True)
    save_json(root / 'DISCLAIMER.json', {'official_data': False, 'research_results': False,
              'purpose': 'CLI software verification with artificial records; no IDS conclusions'})
    save_json(root / 'environment.json', environment())
    results: list[dict] = []
    env = os.environ.copy()
    env.setdefault('MPLCONFIGDIR', '/tmp/bpso-mpl')

    def execute(label: str, arguments: list[str], *, official: bool = False) -> None:
        command = [sys.executable, '-m', 'src.cli', *arguments]
        log = root / 'logs' / f'{len(results):02d}_{label}.log'
        log.parent.mkdir(exist_ok=True)
        began = time.perf_counter()
        with log.open('w') as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=180)
        contents = log.read_text()
        if run.returncode == 0:
            status = 'passed'
        elif official and 'FileNotFoundError' in contents:
            status = 'blocked_missing_official_data'
        else:
            status = 'failed'
        results.append({'label': label, 'command': command, 'exit_code': run.returncode, 'status': status,
                        'wall_seconds': time.perf_counter()-began, 'log': str(log)})
        save_json(root / 'commands.json', results)
        print(f'{label}: {status}', flush=True)
        if status == 'failed':
            print(contents[-4000:], flush=True)
            raise RuntimeError(f'Command failed: {label}. See {log}')

    for name in ['gas_pipeline', 'ton_iot', 'bot_iot', 'swat']:
        execute(f'official_prepare_{name}', ['prepare', '--config', f'configs/{name}.yaml',
                                           '--output', str(root / 'official' / name)], official=True)
    for command in ['train', 'search', 'diagnostics', 'compare']:
        arguments = [command, '--config', 'configs/gas_pipeline.yaml',
                     '--output', str(root / 'official' / f'gas_{command}')]
        if (Path(read_config('configs/gas_pipeline.yaml')['data']['processed']) / 'metadata.json').exists():
            results.append({'label': f'official_gas_{command}', 'status': 'deferred_full_experiment',
                            'command': [sys.executable, '-m', 'src.cli', *arguments],
                            'reason': 'Full paper budgets are outside software verification; use configs/gas_smoke.yaml for a bounded real-data check.'})
            save_json(root / 'commands.json', results)
            print(f'official_gas_{command}: deferred_full_experiment', flush=True)
        else:
            execute(f'official_gas_{command}', arguments, official=True)

    rng = np.random.default_rng(381)
    fixture = root / 'ARTIFICIAL_SOFTWARE_FIXTURE.csv'
    pd.DataFrame({'sensor_a': rng.normal(size=90), 'sensor_b': rng.normal(size=90),
                  'protocol': ['x', 'y', 'z']*30, 'class': ['class_a', 'class_b', 'class_c']*30}).to_csv(fixture, index=False)
    cfg = read_config('configs/debug.yaml')
    cfg['data'] = {'name': 'gas_pipeline', 'files': [str(fixture)], 'target': 'class', 'normal_label': 'class_a',
                   'synthetic': True, 'normalization': 'minmax', 'processed': str(root / 'processed'), 'split_seed': 381}
    cfg['comparison']['methods'] = ['bpso', 'bpso_reduced', 'bpso_sh', 'hyperband', 'dehb', 'random', 'surrogate']
    cfg['comparison']['seeds'] = [42, 43, 44]
    path = root / 'fixture_config.json'
    save_json(path, cfg)
    common = ['--config', str(path)]
    execute('fixture_prepare', ['prepare', *common])
    execute('fixture_fixed_train', ['train', *common, '--output', str(root / 'fixed_model')])
    for method in cfg['comparison']['methods']:
        execute(f'fixture_search_{method}', ['search', *common, '--method', method,
                                           '--output', str(root / 'searches' / method)])
    execute('fixture_search_resume', ['search', *common, '--output', str(root / 'searches' / 'bpso')])
    execute('fixture_selected_train', ['train', *common, '--architecture', str(root / 'searches/bpso/best_architecture.json'),
                                      '--output', str(root / 'selected_model')])
    diagnostic = copy.deepcopy(cfg)
    diagnostic['compute']['max_epochs'] = 90
    diagnostic['diagnostics']['seeds'] = [42, 43, 44]
    diagnostic_path = root / 'diagnostic_config.json'
    save_json(diagnostic_path, diagnostic)
    execute('fixture_diagnostics', ['diagnostics', '--config', str(diagnostic_path), '--output', str(root / 'diagnostics')])
    execute('fixture_comparison', ['compare', *common, '--output', str(root / 'comparisons')])
    execute('fixture_report', ['report', '--input', str(root / 'comparisons'), '--output', str(root / 'reports')])
    execute('fixture_ranking_report', ['report', '--input', str(root / 'diagnostics'), '--output', str(root / 'ranking_reports')])
    timed = copy.deepcopy(cfg)
    timed['compute'] = {'max_epochs': 1000000, 'max_seconds': 2}
    timed['search']['candidates'] = 100000
    timed_path = root / 'time_config.json'
    save_json(timed_path, timed)
    execute('fixture_time_budget', ['search', '--config', str(timed_path), '--method', 'random',
                                    '--output', str(root / 'time_search')])
    counts = {status: sum(r['status'] == status for r in results)
              for status in ['passed', 'blocked_missing_official_data', 'deferred_full_experiment', 'failed']}
    report = json.loads((root / 'reports/summary.json').read_text())
    summary = {'fixture_only': True, 'commands': counts, 'comparison_searches': len(report['searches']),
               'comparison_final_models': len(report['finals']), 'artifact_directory': str(root),
               'tests_xml': 'results/command_validation/tests.xml'}
    save_json(root / 'validation_summary.json', summary)
    save_json(Path('results/command_validation/latest.json'), summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
