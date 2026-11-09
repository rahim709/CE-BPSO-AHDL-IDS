"""Freeze the archive-free study and verify persisted continuation artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import pandas as pd
import torch
from src.utils.core import digest, save_json


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='results/ua_amf_online_20261108')
    root=Path(parser.parse_args().input)
    start=json.loads((root/'source_hashes.json').read_text())
    assert all(sha(Path(p))==expected for p,expected in start.items()),'Algorithm source changed after benchmark launch'
    source=root/'source_snapshot'
    files=[Path(p) for p in start]+list(Path('scripts').glob('archive_free_*.py'))+[Path('configs/ua_amf_online.yaml'),Path('tests/test_online_search.py'),Path('docs/archive_free_ua_amf.md')]
    for path in files:
        target=source/path
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    logs=root/'execution_logs'
    logs.mkdir(exist_ok=True)
    if root.resolve()==Path('results/ua_amf_online_20261108').resolve():
        for name in ['online_benchmark.log','online_diagnostic.log','online_tests_cpu.log','online_tests_cuda.log','online_tests_cuda_final.log']:
            shutil.copy2(Path('/tmp')/name,logs/name)
    counts={}
    for name in ['tests_cpu.xml','tests_cuda.xml','tests_online_cuda_final.xml']:
        suites=list(ET.parse(root/name).getroot().iter('testsuite'))
        assert not sum(int(s.get('failures',0))+int(s.get('errors',0)) for s in suites)
        counts[name]=sum(int(s.get('tests',0)) for s in suites)
    checked=0
    promoted_pairs=0
    reference=None
    folders=sorted(root.glob('*/*/seed_*/search_summary.json'))
    for path in folders:
        folder=path.parent
        cfg=json.loads((folder/'resolved_config.json').read_text())
        shared={key:cfg[key] for key in ['data','space','training','runtime']}
        if reference is None:reference=shared
        assert shared==reference,'Unmatched data, search space, training or runtime'
        assert cfg['search']['epochs']==30 and cfg['search']['budgets']==[3,10,30]
        meta=json.loads((Path(cfg['data']['processed'])/'metadata.json').read_text())
        assert len(meta['classes'])==8
        assert not (Path(cfg['data']['processed'])/'test.npz').exists()
        events=json.loads((folder/'evaluations.json').read_text())
        for candidate in (folder/'candidates').iterdir():
            trajectory=candidate/'trajectory.json'
            if not trajectory.exists():continue
            rows=json.loads(trajectory.read_text())
            saved=torch.load(candidate/'last.pt',map_location='cpu',weights_only=False)
            required={'identity','epoch','model','optimizer','python_rng','numpy_rng','torch_rng','loader_rng','cuda_rng'}
            assert required<=set(saved)
            assert saved['epoch']==rows[-1]['epoch'] and saved['cuda_rng'] is not None
            architecture=json.loads((candidate/'architecture.json').read_text())
            identity=digest({'architecture':architecture,'data':meta['fingerprint'],'seed':cfg['seed'],'training':cfg['training']})
            assert saved['identity']==identity==candidate.name
            charged=[e for e in events if e['candidate']==candidate.name]
            row_gpu=sum(r['training_seconds']+r['validation_seconds'] for r in rows if r['device'].startswith('cuda'))
            assert abs(row_gpu-sum(e['gpu_seconds'] for e in charged))<1e-8
            assert len(rows)==sum(e['charged_epochs'] for e in charged)
            checked+=1
        # Charged continuation is incremental and does not restart old epochs.
        previous={}
        for event in events:
            key=event['candidate']
            assert event['charged_epochs']==event['completed_budget']-previous.get(key,0)
            if previous.get(key,0):promoted_pairs+=1
            previous[key]=event['completed_budget']
    raw=pd.read_csv(root/'seed_results.csv')
    assert len(raw)==43 and (raw.phase!='diagnostic').sum()==42
    assert not raw.numerical_failures.sum()
    summary=pd.read_csv(root/'comparison.csv')
    q=summary[summary.phase=='quality'].set_index('variant')
    for key,row in q.iterrows():
        line=next(x for x in (root/'quality_table.tex').read_text().splitlines() if x.startswith({'bpso':'BPSO &','bpso_sh':'BPSO + SH &','dehb':'DEHB &','offline_ua':'Offline UA &','online_prediction':'Online prediction &','online_uncertainty':'Online + uncertainty &','online_full':'Archive-free full &'}[key]))
        assert f'{row.gpu_mean:.3f}' in line and f'{row.total_gpu_mean:.3f}' in line and f'{row.f1_mean:.4f}' in line
    for name in ['manuscript.tex','manuscript_body.tex','manuscript.pdf','amf_efficiency.tex','references.bib']:
        target=root/'manuscript_final'/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(Path('docs/latex')/name,target)
    save_json(root/'commands.json',{
      'primary':'MPLCONFIGDIR=/tmp/online-mpl .venv/bin/python -m scripts.archive_free_benchmark --output results/ua_amf_online_20261108',
      'diagnostic':'MPLCONFIGDIR=/tmp/online-mpl .venv/bin/python -m scripts.archive_free_benchmark --output results/ua_amf_online_20261108 --diagnostic',
      'analyze':'MPLCONFIGDIR=/tmp/online-mpl .venv/bin/python -m scripts.archive_free_analyze --input results/ua_amf_online_20261108',
      'report':'.venv/bin/python -m scripts.archive_free_report --input results/ua_amf_online_20261108',
      'tests_cpu':'.venv/bin/python -m pytest -q --junitxml=results/ua_amf_online_20261108/tests_cpu.xml',
      'tests_cuda':'BPSO_TEST_DEVICE=cuda .venv/bin/python -m pytest -q tests/test_training_search.py tests/test_adaptive.py tests/test_curve_predictor.py tests/test_runtime_efficiency.py tests/test_efficiency_search.py tests/test_online_search.py',
      'manuscript':'cd docs/latex; pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex; bibtex manuscript; pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex; pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex',
      'finalize':'.venv/bin/python -m scripts.archive_free_finalize --input results/ua_amf_online_20261108'})
    save_json(root/'FINAL_VERIFICATION.json',{'complete_primary_runs':42,'separate_diagnostics':1,'new_search_gpu_seconds':float(raw.gpu_seconds.sum()),
      'checkpoint_files_verified':checked,'incremental_continuation_pairs_verified':promoted_pairs,'source_unchanged_during_experiments':True,
      'matched_data_training_space_runtime':True,'eight_labels':True,'no_search_test_array':True,'table_metrics_reconciled':True,
      'all_tests_passed':counts,'primary_predictive_decisions':int(raw.loc[raw.phase!='diagnostic','prediction_decisions'].sum()),
      'external_archive_gpu_new_method':0.,'recommendation':'Use DEHB for the strongest measured control; use SH for a simple BPSO alternative. No online predictive superiority demonstrated.',
      'incomplete':['Larger-horizon multi-seed activation and calibration comparison','Full-data/full-scale searches','New frozen final detectors','Independent evaluation release','Algorithm novelty/superiority evidence']})
    manifest={str(path.relative_to(root)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in root.rglob('*')
              if path.is_file() and path.name!='artifact_manifest.json' and path.suffix not in ['.pt','.npz','.pyc']}
    save_json(root/'artifact_manifest.json',manifest)
    print(f'Frozen {len(folders)} runs, verified {checked} checkpoints and {promoted_pairs} incremental continuations.',flush=True)


if __name__=='__main__':main()
