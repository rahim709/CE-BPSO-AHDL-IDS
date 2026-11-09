"""Persist a small integration run using an explicitly artificial software fixture.

This command produces NO real-data research evidence or IDS performance claim.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.preprocessing import prepare
from src.evaluation.diagnostics import run_diagnostics
from src.evaluation.reporting import generate_reports
from src.optimization.search import run_search
from src.training.engine import final_evaluate
from src.utils.core import environment, read_config, save_json


def main() -> None:
    output = Path('results/software_validation') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(381)
    raw = output / 'ARTIFICIAL_SOFTWARE_FIXTURE.csv'
    pd.DataFrame({'sensor_a': rng.normal(size=90), 'sensor_b': rng.normal(size=90),
                  'protocol': ['x', 'y', 'z']*30, 'class': ['class_a', 'class_b', 'class_c']*30}).to_csv(raw, index=False)
    cfg = read_config('configs/debug.yaml')
    cfg['data'] = {'name': 'gas_pipeline', 'files': [str(raw)], 'target': 'class',
                   'synthetic': True, 'normalization': 'minmax', 'processed': str(output / 'processed'), 'split_seed': 381}
    cfg['comparison']['methods'].append('surrogate')
    prepare(cfg['data'], cfg['data']['processed'])
    save_json(output / 'environment.json', environment())
    save_json(output / 'resolved_config.json', cfg)
    save_json(output / 'DISCLAIMER.json', {'purpose': 'software verification only', 'official_data': False,
                                           'research_results': False, 'warning': 'No IDS accuracy or hypothesis inference is valid.'})
    results = []
    for method in cfg['comparison']['methods']:
        experiment = copy.deepcopy(cfg)
        experiment['search']['method'] = method
        summary = run_search(experiment, output / 'searches' / method)
        results.append({'method': method, 'status': summary['status'], 'charged_epochs': summary['charged_epochs'],
                        'selected_candidate': summary['best'] is not None})
    diagnostic = copy.deepcopy(cfg)
    diagnostic['compute']['max_epochs'] = 30
    run_diagnostics(diagnostic, output / 'diagnostics')
    final_evaluate(cfg['architecture'], cfg['data']['processed'], output / 'fixed_model', 2, 10042,
                   cfg['training'], normal_label='class_a')
    report = generate_reports(output, output / 'figures')
    save_json(output / 'verification_summary.json', {'fixture_only': True, 'methods': results, 'reports': report})
    save_json(output.parent / 'latest.json', {'directory': str(output)})
    print(f'Software fixture validation complete. See {output}/verification_summary.json')


if __name__ == '__main__':
    main()
