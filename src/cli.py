"""Experiment command line. Run from repository root: python -m src.cli --help."""
from __future__ import annotations
import argparse
import copy
import json
import logging
from pathlib import Path
from src.data.preprocessing import prepare
from src.evaluation.diagnostics import run_diagnostics
from src.evaluation.reporting import generate_reports
from src.optimization.search import run_search
from src.training.engine import final_evaluate
from src.utils.core import environment, read_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'search', 'diagnostics', 'compare', 'report'])
    parser.add_argument('--config', default='configs/gas_pipeline.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--method', choices=['bpso', 'bpso_reduced', 'bpso_sh', 'hyperband', 'dehb', 'random', 'surrogate', 'ua_amf_bpso'])
    parser.add_argument('--architecture', help='Architecture JSON from search; overrides fixed architecture')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--input', help='Report input directory')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.command == 'report':
        result = generate_reports(args.input or 'results', args.output or 'results/figures')
    else:
        cfg = read_config(args.config)
        if args.seed is not None:
            cfg['seed'] = args.seed
        if args.method:
            cfg['search']['method'] = args.method
        output = Path(args.output or f"results/{args.command}/{cfg['data']['name']}/seed_{cfg.get('seed', 42)}")
        if args.command == 'prepare':
            result = prepare(cfg['data'], args.output or cfg['data']['processed'])
        else:
            output.mkdir(parents=True, exist_ok=True)
            # Preserve the prior record if a user accidentally reuses an output directory.
            existing_config = output / 'resolved_config.json'
            if existing_config.exists() and json.loads(existing_config.read_text()) != cfg:
                raise ValueError('Output directory has a different resolved config; choose a fresh output')
            save_json(output / 'environment.json', environment())
            save_json(output / 'resolved_config.json', cfg)
            if args.command == 'train':
                architecture = json.loads(Path(args.architecture).read_text()) if args.architecture else cfg['architecture']
                result = final_evaluate(architecture, cfg['data']['processed'], output,
                                        cfg.get('final_epochs', 200), cfg.get('final_seed', cfg.get('seed', 42) + 10000),
                                        cfg.get('training', {}), cfg['data'].get('normal_label'))
            elif args.command == 'search':
                result = run_search(cfg, output)
            elif args.command == 'diagnostics':
                result = run_diagnostics(cfg, output)
            else:
                result = []
                for seed in cfg.get('comparison', {}).get('seeds', [42, 43, 44]):
                    for method in cfg.get('comparison', {}).get('methods', ['bpso', 'bpso_reduced', 'bpso_sh', 'hyperband', 'dehb', 'random']):
                        run_cfg = copy.deepcopy(cfg)
                        run_cfg['seed'] = seed
                        run_cfg['search']['method'] = method
                        folder = output / method / f'seed_{seed}'
                        save_json(folder / 'resolved_config.json', run_cfg)
                        summary = run_search(run_cfg, folder)
                        final_status = ('disabled' if not cfg.get('comparison', {}).get('final_train', True)
                                        else 'skipped_incomplete_selection')
                        if summary['selection_complete'] and cfg.get('comparison', {}).get('final_train', True):
                            final_evaluate(summary['best']['architecture'], cfg['data']['processed'], folder / 'final',
                                           cfg.get('final_epochs', 200), seed + 10000, cfg.get('training', {}), cfg['data'].get('normal_label'))
                            final_status = 'completed'
                        result.append({'method': method, 'seed': seed, 'status': summary['status'],
                                       'selection_status': summary['selection_status'], 'final_status': final_status})
                save_json(output / 'comparison_status.json', result)
                generate_reports(output, output / 'reports')
    logging.info('Completed: %s', json.dumps(result, default=str)[:3000])


if __name__ == '__main__':
    main()
