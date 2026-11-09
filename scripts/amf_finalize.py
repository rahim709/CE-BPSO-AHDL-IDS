"""Final audit/manifest for the measured predictive UA-AMF investigation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import torch
from src.models.controls import make_model
from src.utils.core import save_json

STUDY = Path('results/ua_amf_prediction_20261108')
BENCH = Path('results/ua_amf_prediction_benchmark_20261108')
FINAL = Path('results/ua_amf_prediction_final_20261108')


def main():
    def tests(path):
        suites = ET.parse(path).getroot().iter('testsuite')
        suites = list(suites)
        return {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    source = json.loads((BENCH / 'source_hashes.json').read_text())
    mismatched = [p for p, h in source.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
    assert not mismatched, f'Runtime source differs from benchmark freeze: {mismatched}'
    meta = json.loads((FINAL / 'data/metadata.json').read_text())
    with np.load(FINAL / 'data/train.npz') as data:
        features = data['x'].shape[1]
    model_audits = []
    for path in sorted((FINAL / 'models').glob('*/last.pt')):
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == 30 and checkpoint['metadata_fingerprint'] == meta['fingerprint']
        assert all(k in checkpoint for k in ['optimizer', 'python_rng', 'numpy_rng', 'torch_rng', 'loader_rng', 'cuda_rng'])
        model = make_model(checkpoint['architecture'], features, len(meta['classes']))
        model.load_state_dict(checkpoint['model'])
        parameters = sum(p.numel() for p in model.parameters())
        metrics = json.loads((path.parent / 'test_metrics.json').read_text())
        assert parameters == metrics['parameters']
        assert metrics['seed'] == checkpoint['seed']
        model_audits.append({'checkpoint': str(path), 'parameters': parameters, 'state_and_architecture_match': True})
        del checkpoint, model
    save_json(FINAL / 'checkpoint_audit.json', model_audits)
    table = pd.read_csv(BENCH / 'comparison.csv').set_index('variant')
    tex = (BENCH / 'comparison.tex').read_text()
    for name, row in table.iterrows():
        expected = f"{name.replace('_', chr(92) + '_')} & {row.f1:.4f} & {row.gpu:.2f} & {row.gpu_saving_percent:.1f}"
        assert expected in tex
    section = Path('docs/latex/amf_efficiency.tex').read_text()
    assert 'Computational Efficiency of Adaptive Multi-Fidelity BPSO' in section
    for name in ['bpso', 'ua_amf_bpso']:
        assert f'{table.loc[name, "f1"]:.4f}' in section
    budget = json.loads((BENCH / 'budget_audit.json').read_text())
    assert len(budget) == 24 and all(r['all_job_starts_within_cap'] for r in budget)
    final_stats = pd.read_csv(FINAL / 'seed_metrics.csv').groupby(['variant', 'search_seed']).mean(numeric_only=True).groupby('variant').agg(['mean', 'std'])
    final_metrics = {name: {metric: {'mean': row[(metric, 'mean')], 'sd_across_search_seed_means': row[(metric, 'std')]}
                              for metric in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'normal_fpr']} for name, row in final_stats.iterrows()}
    files = ['src/optimization/curve_predictor.py', 'src/optimization/predictive.py', 'src/optimization/search.py',
             'configs/ua_amf_prediction.yaml', 'tests/test_curve_predictor.py',
             'scripts/amf_reliability.py', 'scripts/amf_fit_predictor.py', 'scripts/amf_benchmark.py', 'scripts/amf_analyze.py',
             'scripts/amf_final_evaluation.py', 'scripts/amf_write_report.py', 'scripts/amf_finalize.py',
             'docs/ua_amf_prediction_study.md', 'docs/latex/amf_efficiency.tex', 'docs/latex/manuscript_body.tex',
             'docs/latex/references.bib', 'docs/latex/manuscript.pdf', 'README.md', 'docs/latex/README.md']
    full = table.loc['ua_amf_bpso']
    summary = {'cpu_tests': tests(STUDY / 'tests_cpu_final.xml'), 'cuda_targeted_tests': tests(STUDY / 'tests_cuda.xml'),
               'reliability': json.loads((STUDY / 'reliability_summary.json').read_text()),
               'predictor_heldout_calibration': json.loads((STUDY / 'predictor_report.json').read_text())['calibration'],
               'completed_benchmark_searches': 24, 'search_seeds': [501, 502, 503],
               'full_method_online_gpu_saving_percent': full.gpu_saving_percent,
               'full_method_online_wall_saving_percent': full.wall_saving_percent,
               'full_validation_mean': full.f1, 'standard_validation_mean': table.loc['bpso', 'f1'],
               'ordinary_halving_validation_mean': table.loc['bpso_sh', 'f1'],
               'predictive_uncertainty_component_f1_difference': table.loc['bpso_prediction_uncertainty', 'f1'] - table.loc['bpso_prediction', 'f1'],
               'cost_component_f1_difference': full.f1 - table.loc['bpso_prediction_uncertainty', 'f1'],
               'unamortized_full_gpu_seconds': full.total_gpu,
               'final_test': final_metrics, 'final_test_status': 'historical, previously examined; not blind independent data',
               'final_model_references_completed': 18, 'unique_final_models': len(model_audits),
               'verified_predictions': {'cohort': 183, 'benchmark': 485, 'final_unique': len(model_audits)},
               'all_runtime_sources_match_freeze': True, 'manuscript_table_matches_seed_results': True,
               'max_search_wall_overrun_seconds': max(r['epoch_boundary_overrun_seconds'] for r in budget),
               'remaining_incomplete': ['14 third-seed reliability terminal trajectories', 'full-training rather than reduced-row comparisons',
                                        'larger architecture calibration/holdout', 'independent capture/test', 'more search seeds and formal noninferiority study'],
               'research_conclusion': 'No demonstrated full-method GPU saving or benefit beyond successive halving; uncertainty hurt one seed; cost term gave no terminal-score benefit. No net end-to-end savings including archive cost, no established novelty/superiority, and final false-alarm rates are unacceptable.',
               'files_created_or_modified_by_task': files,
               'prior_manuscript_backup': str(STUDY / 'manuscript_before')}
    # Check counts against the saved audit records.
    for key, path in [('cohort', STUDY / 'cohort_prediction_audit.json'), ('benchmark', BENCH / 'prediction_audit.json')]:
        assert summary['verified_predictions'][key] == json.loads(path.read_text())['verified_archives']
    save_json(STUDY / 'FINAL_SUMMARY.json', summary)
    manifest = {}
    for root in [STUDY, BENCH, FINAL]:
        for path in root.rglob('*'):
            if path.is_file() and path.name != 'artifact_manifest.json' and path.suffix in {'.json', '.csv', '.tex', '.pdf', '.png', '.xml', '.joblib', '.npz', '.pt', '.log'}:
                manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in files:
        manifest[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    save_json(STUDY / 'artifact_manifest.json', manifest)
    print(json.dumps({'cpu_tests': summary['cpu_tests'], 'cuda_tests': summary['cuda_targeted_tests'],
                      'final_models_checked': len(model_audits), 'hashed_artifacts': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
