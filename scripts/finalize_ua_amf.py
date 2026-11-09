"""Collect measured final status and source hashes without altering experiment logs."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from src.utils.core import save_json


def main():
    root = Path('results/ua_amf_validation')
    def tests(path):
        suites = list(ET.parse(path).getroot().iter('testsuite'))
        return {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ['tests', 'failures', 'errors', 'skipped']}
    def method(path):
        frame = pd.read_csv(path).set_index('variant').astype(object)
        return frame.where(pd.notnull(frame), None).loc['ua_amf_bpso'].to_dict()
    files = ['src/optimization/adaptive.py', 'src/optimization/evaluator.py', 'src/optimization/search.py', 'src/training/engine.py', 'src/cli.py',
             'tests/test_adaptive.py', 'configs/ua_amf.yaml', 'configs/ua_amf_cohort.yaml', 'configs/ua_amf_matched.yaml',
             'scripts/ua_amf_study.py', 'scripts/audit_ua_amf.py', 'scripts/build_ua_amf_report.py', 'scripts/build_ua_amf_cohort.py', 'scripts/finalize_ua_amf.py',
             'docs/ua_amf_implementation.md', 'docs/latex/ua_amf_manuscript.tex', 'docs/latex/ua_amf_manuscript.pdf', 'README.md', 'docs/latex/README.md']
    summary = {'implemented': True, 'cpu_tests': tests(root / 'tests_cpu_final.xml'), 'cuda_tests': tests(root / 'tests_cuda.xml'),
               'completed_searches': {'initial_cpu_not_pooled': 9, 'cpu_three_seed': 27, 'cuda_one_seed': 9},
               'search_epochs_per_run': 24,
               'cpu_full_method': method('results/ua_amf_multiseed_20261108/comparison_table.csv'),
               'cuda_full_method': method('results/ua_amf_cuda_pilot_20261108/comparison_table.csv'),
               'cohort': json.loads(Path('results/ua_amf_cohort_20261108/cohort_audit.json').read_text()),
               'new_final_test_evaluation': False, 'independent_evaluation_data': False, 'full_data_matched_search_executed': False,
               'conclusion': 'No demonstrated useful-performance speedup or uncertainty benefit. Full-method assessment recall is zero for all attacks. Not ready for IEEE optimizer-superiority claims.',
               'limitations': ['Reduced-row pilots and fixed validation reuse', 'Three search seeds; no significance claim',
                               'CPU timing briefly overlapped CUDA jobs', 'No full-search-universe reference regret',
                               'No independent capture/test', 'SEM bonus is not a future-fidelity confidence bound',
                               'Full-space 4 GB fit not guaranteed', 'Epoch-boundary overruns; hard-crash wall accounting is a lower bound'],
               'files_created_or_modified_by_task': files,
               'historical_artifacts': 'Preserved; pre-existing dirty changes retained. Additive manuscript and separate new result directories.',
               'source_versions': 'Launch-time source hashes saved per study. Final recovered-score eligibility fix applies to hard-crash recovery, which successful performance runs did not exercise.'}
    save_json(root / 'FINAL_SUMMARY.json', summary)
    save_json(root / 'final_source_hashes.json', {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files if Path(p).suffix in {'.py', '.yaml'}})
    manifest = {}
    for directory in ['results/ua_amf_multiseed_20261108', 'results/ua_amf_cuda_pilot_20261108', 'results/ua_amf_cohort_20261108']:
        for path in Path(directory).rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.csv', '.png', '.tex'}:
                manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(root / 'artifact_hashes.json', manifest)
    print(json.dumps({'cpu_tests': summary['cpu_tests'], 'cuda_tests': summary['cuda_tests'], 'hashed_artifacts': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
