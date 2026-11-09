"""Recalculate budgets, physical continuation cost, metrics and artifact hashes."""
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

ROOT=Path('results/ua_amf_efficiency_20261108')


def tests(path):
    suites=list(ET.parse(path).getroot().iter('testsuite'))
    return {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ['tests','failures','errors','skipped']}


def main():
    before=json.loads((ROOT/'benchmark_sources.json').read_text())
    after=json.loads((ROOT/'followup20/source_hashes.json').read_text())
    assert before==after
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
    initial=pd.read_csv(ROOT/'seed_results.csv')
    follow=pd.read_csv(ROOT/'followup20/seed_results.csv')
    assert len(initial)==108 and len(follow)==54
    old_gpu=0.
    old_epochs=0
    for path in (ROOT/'followup20').glob('*/*/seed_*/import_provenance.json'):
        info=json.loads(path.read_text())
        source=Path(info['source'])/'evaluations.json'
        assert hashlib.sha256(source.read_bytes()).hexdigest()==info['source_ledger_sha256']
        old_gpu+=info['original_gpu_seconds']
        old_epochs+=info['original_epochs']
    additional_gpu=follow.gpu_seconds.sum()-old_gpu
    additional_epochs=int(follow.epochs.sum()-old_epochs)
    assert additional_gpu>=-1e-8 and additional_epochs>=0
    profile=json.loads((ROOT/'profile_results_completed.json').read_text())
    cache=json.loads((ROOT/'continuation_cache_check.json').read_text())
    profile_gpu=sum(r.get('gpu_seconds',0.) for r in profile)
    cache_gpu=sum(r['gpu_seconds'] for r in cache['records'])
    trace=json.loads((ROOT/'original_trace/trajectory.json').read_text())[-1]['gpu_seconds']
    final_gpu=0.
    final_epochs=0
    model_audits=[]
    meta=json.loads(Path('results/ua_amf_prediction_final_20261108/data/metadata.json').read_text())
    for path in (ROOT/'final/models').glob('*/last.pt'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==30 and checkpoint['seed']==1601
        assert checkpoint['metadata_fingerprint']==meta['fingerprint']
        assert all(k in checkpoint for k in ['optimizer','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng'])
        model=make_model(checkpoint['architecture'],26,8)
        model.load_state_dict(checkpoint['model'])
        count=sum(p.numel() for p in model.parameters())
        metrics=json.loads((path.parent/'test_metrics.json').read_text())
        assert count==metrics['parameters']
        final_gpu+=checkpoint['rows'][-1]['gpu_seconds']
        final_epochs+=30
        model_audits.append({'path':str(path),'parameters':count,'weights_architecture_seed_match':True})
        del checkpoint,model
    save_json(ROOT/'final/checkpoint_audit.json',model_audits)
    resources={'pilot_gpu_seconds':float(initial.gpu_seconds.sum()),'followup_incremental_gpu_seconds':float(additional_gpu),
               'followup_imported_gpu_seconds_not_double_counted':float(old_gpu),'profile_gpu_seconds':profile_gpu,
               'trace_gpu_seconds':trace,'continuation_mechanism_check_gpu_seconds':cache_gpu,
               'new_final_training_gpu_seconds':final_gpu,
               'new_measured_experiment_gpu_elapsed_total':float(initial.gpu_seconds.sum()+additional_gpu+profile_gpu+trace+cache_gpu+final_gpu),
               'pilot_epochs':int(initial.epochs.sum()),'followup_incremental_epochs':additional_epochs,'final_unique_epochs':final_epochs,
               'historical_archive_gpu_seconds':403.8190207847874,'new_archive_gpu_seconds':0.,
               'scope':'Sum of measured training/validation elapsed, same definition as benchmarks. GPU warmups/setup/inference, tests and publication-generation work are not kernel-profiled here; stage wall and inference latency are recorded separately.'}
    save_json(ROOT/'resource_accounting.json',resources)
    conclusion=json.loads((ROOT/'research_conclusion.json').read_text())
    audits=[json.loads((ROOT/p).read_text()) for p in ['audit.json','followup20/audit.json','final/audit.json']]
    summary={'cpu_tests':tests(ROOT/'tests_cpu.xml'),'cuda_targeted_tests':tests(ROOT/'tests_cuda.xml'),
             'initial_searches':108,'charged_continuations':54,'continuations_are_not_new_independent_replications':True,
             'seeds':[601,602,603],'initial_quality_targets':[.4,.5,.6],'followup_primary_target':.6,
             'budgets':[8.,20.],'final_model_references':27,'unique_final_models':len(model_audits),
             'prediction_audits':audits,'conclusion':conclusion,'resources':resources,
             'all_runtime_sources_match_frozen_benchmark':True,'new_archive_neural_training_gpu_seconds':0.,
             'numerical_or_memory_search_failures':int(initial.numerical_failures.sum()+follow.numerical_failures.sum()),
             'unfinished_experiments':['More than three search seeds','More than one independent final-training seed per selected architecture',
                                       'Full-training-row rather than8192-row NAS comparison','Independent capture/release evaluation'],
             'profile_interruption':'Last combined trial resumed under explicit30-second allowance; prior partial results preserved; resumed wall is invocation-only.',
             'manuscript_backup':str(ROOT/'manuscript_before')}
    save_json(ROOT/'FINAL_SUMMARY.json',summary)
    snapshot=ROOT/'source_snapshot'
    if not snapshot.exists():shutil.copytree('src',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files=['src/training/engine.py','src/training/runtime.py','src/optimization/evaluator.py','src/optimization/search.py',
           'src/optimization/curve_predictor.py','src/optimization/dynamic.py','configs/ua_amf_efficient.yaml',
           'tests/test_runtime_efficiency.py','tests/test_efficiency_search.py','tests/test_curve_predictor.py',
           'scripts/efficiency_profile.py','scripts/efficiency_development.py','scripts/efficiency_benchmark.py',
           'scripts/efficiency_extend.py','scripts/efficiency_analyze.py','scripts/efficiency_archive_ablation.py',
           'scripts/efficiency_cache_check.py','scripts/efficiency_final.py','scripts/efficiency_report.py','scripts/efficiency_finalize.py',
           'docs/efficiency_optimization.md','docs/latex/amf_efficiency.tex','docs/latex/manuscript_body.tex','docs/latex/manuscript.pdf',
           'README.md','docs/latex/README.md']
    save_json(ROOT/'files_created_or_modified.json',files)
    manifest={}
    for path in ROOT.rglob('*'):
        if path.is_file() and path.name!='artifact_manifest.json' and '__pycache__' not in path.parts:
            manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for p in files:manifest[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    save_json(ROOT/'artifact_manifest.json',manifest)
    print(json.dumps({'cpu':summary['cpu_tests'],'cuda':summary['cuda_targeted_tests'],
                      'new_measured_training_gpu_elapsed':resources['new_measured_experiment_gpu_elapsed_total'],
                      'unique_final_models':len(model_audits),'hashed_artifacts':len(manifest)},indent=2))


if __name__=='__main__':main()
