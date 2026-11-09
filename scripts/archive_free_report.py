"""Build manuscript additions from audited measurements, never from target goals."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

LABELS={'bpso':'BPSO','bpso_sh':'BPSO + SH','dehb':'DEHB','offline_ua':'Offline UA','online_prediction':'Online prediction','online_uncertainty':'Online + uncertainty','online_full':'Archive-free full'}
ORDER=list(LABELS)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default='results/ua_amf_online_20261108')
    args=parser.parse_args()
    root=Path(args.input)
    s=pd.read_csv(root/'comparison.csv')
    raw=pd.read_csv(root/'seed_results.csv')
    assert len(raw[raw.phase!='diagnostic'])==42
    assert (s.seeds==3).all()
    q=s[s.phase=='quality'].set_index('variant')
    c=s[s.phase=='compute'].set_index('variant')
    b,u,h,d,o=[q.loc[k] for k in ['bpso','online_full','bpso_sh','dehb','offline_ua']]
    audit=json.loads((root/'metric_budget_audit.json').read_text())
    assert audit['all_metrics_match'] and audit['all_jobs_started_within_gpu_cap']
    quality=['\\begin{tabular}{lrrrrr}\\toprule','Method & Online GPU & Archive & Total GPU & F1 & Hits\\\\\\midrule']
    compute=['\\begin{tabular}{lrrr}\\toprule','Method & Terminal F1 (SD) & Epochs & GPU s\\\\\\midrule']
    markdown=['| Method | Online GPU s | Wall s | Standalone total GPU s | Terminal F1 (SD) | Target reached |',
              '|---|---:|---:|---:|---:|---:|']
    for k in ORDER:
        r=q.loc[k]
        archive=403.8190207847874 if k=='offline_ua' else 0.
        quality.append(f'{LABELS[k]} & {r.gpu_mean:.3f} & {archive:.2f} & {r.total_gpu_mean:.3f} & {r.f1_mean:.4f} & {int(r.reached)}/3\\\\')
        markdown.append(f'| {LABELS[k]} | {r.gpu_mean:.3f} | {r.wall_mean:.3f} | {r.total_gpu_mean:.3f} | {r.f1_mean:.4f} ({r.f1_sd:.4f}) | {int(r.reached)}/3 |')
        t=c.loc[k]
        compute.append(f'{LABELS[k]} & {t.f1_mean:.4f} ({t.f1_sd:.4f}) & {t.epochs_mean:.1f} & {t.gpu_mean:.3f}\\\\')
    quality.append('\\bottomrule\\end{tabular}')
    compute.append('\\bottomrule\\end{tabular}')
    (root/'quality_table.tex').write_text('\n'.join(quality)+'\n')
    (root/'compute_table.tex').write_text('\n'.join(compute)+'\n')
    primary=raw[raw.phase!='diagnostic']
    online=primary[primary.variant.str.startswith('online_')]
    diagnostic=raw[raw.phase=='diagnostic']
    diagtext='No longer diagnostic completed.'
    if len(diagnostic):
        r=diagnostic.iloc[0]
        folder=root/'diagnostic/online_full'/f'seed_{int(r.seed)}'
        gates=json.loads((folder/'online_decisions.json').read_text())
        reasons={reason:sum(g['gate']['reason']==reason for g in gates) for reason in sorted({g['gate']['reason'] for g in gates})}
        reasons_text='; '.join(f"{key.replace('_', ' ')}: {value}" for key,value in reasons.items())
        diagtext=(f'The separate seed {int(r.seed)} diagnostic used {r.gpu_seconds:.3f} GPU seconds and {r.wall_seconds:.3f} wall seconds, '
                  f'matured {int(r.mature_architectures)} distinct terminal architectures, performed {int(r.cpu_fit_count)} CPU fits '
                  f'and applied prediction at {int(r.prediction_decisions)} promotion decisions. '
                  f'Preparation consumed {r.predictor_preparation_cpu_wall_seconds:.4f} CPU wall seconds. Gate counts: {reasons_text}. '
                  'This is a cold-start diagnostic, not a matched performance comparison.')
    cpu=root/'tests_cpu.xml'
    cuda=root/'tests_cuda.xml'
    import xml.etree.ElementTree as ET
    def test_count(path):
        if not path.exists():return 'not run'
        node=ET.parse(path).getroot()
        suites=list(node.iter('testsuite'))
        return f"{sum(int(x.get('tests','0')) for x in suites)} passed" if all(int(x.get('failures','0'))+int(x.get('errors','0'))==0 for x in suites) else 'FAIL'
    report=['# Archive-free computational-efficiency study','',
            '## Outcome','',
            f'Zero separately trained GPU predictor archive was required. Archive-free full used {u.gpu_mean:.3f} actual GPU seconds on average versus {b.gpu_mean:.3f} for BPSO: {u.total_saving_vs_bpso:.2f}% lower total GPU cost in the target-or-cap pilot. Targets were reached in {int(u.reached)}/3 versus {int(b.reached)}/3 seeds. All failures remain right-censored in the saved analysis.', '',
            f'However, all three online variants made {int(online.prediction_decisions.sum())} predictive promotion decisions across {len(online)} matched runs. They used SH fallback. They produced identical winner scores to SH, whose mean target-or-cap GPU was {h.gpu_mean:.3f}. The {u.total_saving_vs_sh:.2f}% timing difference versus SH is small execution variation, not evidence that online learning improves SH. DEHB reached {int(d.reached)}/3 targets at {d.gpu_mean:.3f} GPU seconds and remains the recommended measured optimizer for this pilot. SH is the recommended simple BPSO alternative.','',
            '## Measured target-or-cap comparison','',*markdown,'',
            'Means use all three seeds601/602/603; failed target0.6 runs are retained. Scores are best genuinely evaluated30-epoch validation macro-F1 before the strict GPU/wall horizon, not extrapolated prediction scores. Large sample SD indicates instability. Actual GPU time includes final epoch overruns; target decisions beyond the cap are censored. Restricted time-to-target is recorded separately and is not substituted for measured GPU consumption. No superiority test is claimed.','',
            'Standalone total GPU assigns the entire403.819-second historical archive to one offline-UA search. `campaign_costs.csv` also charges it once across three seeds, avoiding triple-charging a reused archive. Offline archive wall elapsed was420.722 seconds. The old offline CPU development/refitting preparation is not fully isolated in these new search logs, so archive-inclusive wall is a documented lower bound; no zero-cost assumption is made. CPU predictor fitting for the new variant is measured inside each search wall time.','',
            f'In the primary online trials CPU preparation ranged {online.predictor_preparation_cpu_wall_seconds.min():.5f}--{online.predictor_preparation_cpu_wall_seconds.max():.5f} seconds; CPU fits totalled {int(online.cpu_fit_count.sum())}. Candidate maturation is paid search work; it is not an uncharged calibration phase.','',
            f'Total newly executed search/diagnostic work across all methods was {raw.gpu_seconds.sum():.3f} GPU seconds; the separate diagnostic consumed {diagnostic.gpu_seconds.sum():.3f} of these seconds. Its model was not transferred into any matched run. Peak allocated GPU memory across this study was {raw.peak_gpu_memory_bytes.max()/2**20:.1f} MiB. Numerical/non-budget failures: {int(raw.numerical_failures.sum())}. Largest final-epoch cap overrun: {raw.gpu_epoch_overrun_seconds.max():.3f} seconds. Tests and analysis are separate verification work, not search savings.','',
            '## Fixed-compute and ablations','',
            '| Method | 20-second terminal macro-F1 mean (SD) | Epochs mean |',
            '|---|---:|---:|',
            *[f'| {LABELS[k]} | {c.loc[k].f1_mean:.4f} ({c.loc[k].f1_sd:.4f}) | {c.loc[k].epochs_mean:.1f} |' for k in ORDER], '',
            'Equal GPU caps measure quality at fixed compute; near-equal consumption does not prove savings. Five/ten/twenty-second anytime scores and raw seed traces are saved. Prediction-only, uncertainty-only and full cost-aware policies have no causal allocation difference when the gate never activates. Any terminal score difference at a hard time boundary would be a timing effect, not evidence for a component. The older offline control uses its frozen development-tuned coefficients and does not transfer labels or tuned coefficients into the new modes.','',
            f'At fixed20-second compute, archive-free full preserved SH scores exactly but trailed offline UA by {c.loc["offline_ua"].f1_mean-c.loc["online_full"].f1_mean:.4f} and DEHB by {c.loc["dehb"].f1_mean-c.loc["online_full"].f1_mean:.4f} absolute macro-F1. Both gaps exceed the proposed0.02 engineering margin. Competitive quality against the strongest control was not preserved.','',
            '## Online predictor diagnostic','',diagtext,'',
            'The conservative predictor requires enough disjoint fit/calibration architectures and four genuinely prequential checks per rung. Its held-out coverage is descriptive under adaptive sampling, not a formal coverage guarantee. Insufficient observations or inaccurate/overwide predictions trigger SH. Removing gates retrospectively to obtain activation would invalidate this pilot. A next experiment should prespecify a larger matched horizon based on maturation counts and assess prediction diagnostics before optimizer superiority claims. If reliable activation remains uneconomic, retain SH or DEHB.','',
            '## Changes and verification','',
            '- `src/optimization/online.py`: CPU forests, frozen prefix features, fixed architecture partitions, prequential checks, SH fallback, measured cost ranking and independent audit logs.',
            '- `src/optimization/search.py`: additive online policy dispatch; standard BPSO remains available.',
            '- `configs/ua_amf_online.yaml`: archive-free deterministic settings.',
            '- `scripts/archive_free_benchmark.py`, `archive_free_analyze.py`, `archive_free_report.py`: matched runs, ledger/prediction/leakage audits, tables, plots and manuscript updates.',
            '- `tests/test_online_search.py`: future-feature exclusion, own-architecture guards, degraded-prediction fallback, exact SH fallback/continuation, no offline loader access.', '',
            f'Final checks: CPU {test_count(cpu)}; targeted CUDA {test_count(cuda)}. Independently re-scored {audit["verified_prediction_archives"]} saved validation prediction archives; all checked metrics and GPU/epoch ledgers matched. Architecture fit/calibration/query identities and prefix features were audited. Logs and per-run checkpoints remain under the study directory.','',
            'No new test evaluation was run for this change. Existing MLP/CNN/GRU and selected-model test results remain contextual; the historical Gas test is not new blind independent data. The study uses a bounded8192/4096 real-data view and three seeds on one GPU; full-data/full-scale performance, calibrated uncertainty under proposal shift and IEEE-level optimizer novelty are not established.','',
            '## Reproduce','',
            'See `docs/archive_free_ua_amf.md` for formulation, pseudocode, frozen protocol and exact commands. Run benchmark in a fresh output for a fresh replication; existing completed runs are intentionally skipped. The data folder references shared training/validation arrays, not an offline predictor. The dependency audit distinguishes those artifacts.']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    # Preserve the immediately preceding manuscript before adding new evidence.
    before=root/'manuscript_before'
    before.mkdir(exist_ok=True)
    for name in ['manuscript.tex','manuscript_body.tex','manuscript.pdf','amf_efficiency.tex','references.bib']:
        src=Path('docs/latex')/name
        if not (before/name).exists():shutil.copy2(src,before/name)
    section=Path('docs/latex/amf_efficiency.tex')
    old=(before/'amf_efficiency.tex').read_text().replace('The current regression suite passed 105 CPU', 'At the preceding study stage, the regression suite passed 105 CPU')
    additions=r'''
\subsection{Archive-free online extension}
The archive-dependent result above motivated a fresh, separately bounded experiment. The new implementation starts with precisely ordinary BPSO--SH proposals, quotas and tie handling. It captures architecture, learning-curve prefix and measured incremental-cost features at epochs3 and10, attaching a terminal label only after actual epoch30 training. No offline bundle, historical target, cross-method checkpoint or separately trained GPU archive enters this variant. Candidate maturation is charged to ordinary search work.

A fixed architecture hash assigns fit/calibration/check roles with expected proportions50/20/30\%. CPU random forests use24 trees of maximum depth4. All current query architectures are excluded from fitting and calibration. Check forecasts are frozen before their terminal labels arrive. At least six fit, four calibration and four prequential check architectures per rung are required. The point gate requires MAE$\leq0.08$, at least10\% and0.005 absolute MAE improvement over current-F1 prediction, and Spearman$\geq0.3$. For uncertainty modes, empirical coverage must be at least0.75 for nominal80\% intervals, with mean width at most0.3; current intervals must also meet the width limit. These are conservative engineering checks, not a proof of calibration under adaptive sampling.

The interval uses a separate calibration quantile of absolute residual divided by forest dispersion (floor0.02). If useful, ranking uses the earlier priority equation with generic frozen $\beta=0.1$, $\lambda=0.05$ and median-normalized measured remaining cost. Prediction-only sets both coefficients to zero; uncertainty-only sets $\lambda=0$. Failure of a gate retains ordinary SH ranking. Only actual terminal observations update swarm bests. Fixed SH quotas are intentionally retained to isolate ranking; no new BPSO equation or algorithmic novelty is claimed.

\subsection{Fresh end-to-end comparison}
The42 new runs use seeds601--603, seven methods, target0.6 and fixed20-GPU-second phases. Dataset rows, eight labels, preprocessing, architecture space, FP32 execution and3/10/30 fidelities match the preceding20-second experiment. A50-second wall safety cap also applies. Method order is a seeded permutation; methods run serially. Earlier checkpoints are not imported. Misses are retained and censored; strict-horizon quality excludes terminal evaluations beyond the cap, while actual resource consumption includes epoch-boundary overruns. Times follow the existing synchronized elapsed convention.

\begin{table}[!t]\centering\caption{Fresh target0.6 runs, means across three search seeds. GPU seconds are measured; archive cost is charged to a standalone search. All misses are retained. Hits count actual terminal targets before the strict horizon.}\label{tab:online-quality}
\resizebox{\columnwidth}{!}{\input{../../RESULTROOT/quality_table.tex}}\end{table}

Archive-free full consumed UGPU GPU seconds versus BGPU for BPSO, a SAVING\% total reduction, with UHITS/3 versus BHITS/3 target attainment. Terminal validation macro-F1 means were UFONE and BFONE respectively. SH consumed HGPU seconds and produced identical winner scores to the archive-free variants. The small SHDIFF\% timing difference does not establish an advantage over SH. DEHB used DGPU seconds and reached DHITS/3 targets. Removing the archive lowers standalone total cost relative to offline UA (OTOTAL seconds), but the online component itself must earn its cost.

Across the18 matched online trials, predictive promotion activated APPLIED times and CPU fitting ran FITS times. Thus the new modes effectively executed SH in this short-budget study. Their ablations do not establish any benefit from uncertainty or cost ranking. The zero external archive cost is real; an online-learning advantage is not demonstrated. CPU preparation ranged PREPMIN--PREPMAX seconds per run and is included in search wall time.

At fixed20-second compute, archive-free full preserved SH scores but trailed offline UA by OFFLINEGAP and DEHB by DEHBGAP absolute macro-F1. Both exceed the0.02 engineering preservation margin; competitive quality against the strongest control was not maintained.

\begin{table}[!t]\centering\caption{Fresh fixed20-GPU-second comparison. F1 is best genuinely evaluated30-epoch validation macro-F1, mean (sample SD). Epoch counts and actual GPU elapsed include charged incremental work.}\label{tab:online-compute}
\resizebox{\columnwidth}{!}{\input{../../RESULTROOT/compute_table.tex}}\end{table}
\begin{figure}[!t]\centering\includegraphics[width=\columnwidth]{../../RESULTROOT/anytime.pdf}
\caption{Fresh fixed-compute anytime quality. Error bars show search-seed SD; points require all three terminal scores. Equal budget consumption is not evidence of cost saving.}\end{figure}

The full archive is charged once for a reusable multi-search campaign in the accompanying CSV, rather than once per seed. Offline CPU development preparation is not fully isolated; recorded archive wall time420.722 seconds therefore supplies only a lower bound for full offline wall preparation. Online CPU fitting/inference and scheduler costs are recorded separately. All saved validation predictions were independently re-scored, and epoch/GPU ledgers and architecture-disjoint feature/predictor identities were verified. The expanded suite passed CPUTESTS CPU and CUDATESTS targeted CUDA-configured tests.

DIAGNOSTIC

No new final detector test was executed in this extension. The preceding historical test results are retained, and the test remains development-known. The evidence supports removing an uneconomic archive and using SH fallback; it does not support superior online uncertainty-aware allocation. DEHB remains the strongest measured control, with SH the simpler BPSO alternative. More seeds, a prespecified larger matched horizon, useful online calibration and independent data remain necessary before an IEEE optimizer contribution can be defended.
'''
    replacement={'RESULTROOT':str(root),'UGPU':f'{u.gpu_mean:.3f}','BGPU':f'{b.gpu_mean:.3f}','SAVING':f'{u.total_saving_vs_bpso:.2f}',
                 'UHITS':str(int(u.reached)),'BHITS':str(int(b.reached)),'UFONE':f'{u.f1_mean:.4f}','BFONE':f'{b.f1_mean:.4f}',
                 'HGPU':f'{h.gpu_mean:.3f}','SHDIFF':f'{abs(u.total_saving_vs_sh):.2f}','DGPU':f'{d.gpu_mean:.3f}',
                 'DHITS':str(int(d.reached)),'OTOTAL':f'{o.total_gpu_mean:.3f}','APPLIED':str(int(online.prediction_decisions.sum())),
                 'FITS':str(int(online.cpu_fit_count.sum())),'PREPMIN':f'{online.predictor_preparation_cpu_wall_seconds.min():.4f}',
                 'PREPMAX':f'{online.predictor_preparation_cpu_wall_seconds.max():.4f}', 'DIAGNOSTIC':diagtext.replace('_',r'\_').replace('%',r'\%')}
    replacement.update(CPUTESTS=test_count(cpu).split()[0],CUDATESTS=test_count(cuda).split()[0])
    replacement.update(OFFLINEGAP=f'{c.loc["offline_ua"].f1_mean-c.loc["online_full"].f1_mean:.4f}',DEHBGAP=f'{c.loc["dehb"].f1_mean-c.loc["online_full"].f1_mean:.4f}')
    for key,value in replacement.items():additions=additions.replace(key,value)
    # Insert human-readable numeric spacing without touching cite IDs or paths.
    for a,z in [('epochs3','epochs 3'),('and10','and 10'),('epoch30','epoch 30'),('proportions50','proportions 50'),('use24','use 24'),('depth4','depth 4'),('least10','least 10'),('nominal80','nominal 80'),('floor0','floor 0'),('The42','The 42'),('seeds601','seeds 601'),('target0','target 0'),('fixed20','fixed 20'),('and3/','and 3/'),('preceding20','preceding 20'),('A50','A 50'),('the18','the 18'),('time420','time 420'),('evaluated30','evaluated 30'),('and0.005','and 0.005'),('least0.75','least 0.75'),('most0.3','most 0.3')]:
        additions=additions.replace(a,z)
    additions=additions.replace('the0.02', 'the 0.02')
    section.write_text(old+'\n'+additions)
    body=Path('docs/latex/manuscript_body.tex')
    original=(before/'manuscript_body.tex').read_text()
    start=original.index('\\begin{abstract}')
    stop=original.index('\\end{abstract}')+len('\\end{abstract}')
    abstract=r'''\begin{abstract}
Neural architecture search for intrusion detection requires useful evaluation under limited compute. This project retains a73-bit BPSO CNN--RNN framework and investigates multi-fidelity allocation on real Gas Pipeline data. An earlier archive-based uncertainty-aware method reduced online GPU use but required403.82 GPU seconds of predictor observations, eliminating end-to-end savings. We implement an archive-free candidate that begins as BPSO with successive halving and learns only from search-generated observations using architecture-disjoint, prequentially checked CPU prediction. In42 fresh bounded runs across three seeds, archive-free full used UGPU GPU seconds on average to reach validation macro-F1 target0.6 or exhaust the cap, versus BGPU for standard BPSO: SAVING\% less total GPU computation. Targets were attained in UHITS/3 versus BHITS/3 seeds. However, predictive promotion never activated in the short searches; results matched ordinary halving, and the measured saving is attributable to multi-fidelity fallback, not online uncertainty learning. DEHB reached DHITS/3 targets using DGPU GPU seconds. All failed target runs were retained as censored. Existing test results use a development-known historical split. The work supports reproducible engineering and removal of archive overhead; superiority or novelty of uncertainty-aware optimization remains unsupported.
\end{abstract}'''
    for key,value in replacement.items():abstract=abstract.replace(key,value)
    for a,z in [('a73','a 73'),('required403','required 403'),('In42','In 42'),('target0','target 0')]:abstract=abstract.replace(a,z)
    updated=original[:start]+abstract+original[stop:]
    updated=updated.replace('the main compute-efficiency contribution remains unestablished', 'an adaptive-prediction advantage remains unestablished')
    updated=updated.replace('The efficiency follow-up found a limited online saving versus BPSO, but full UA did not beat halving or retain quality relative to DEHB. Historical archive cost removes net single-query gains. The contribution is therefore a reproducible course-project implementation, measured FP32 execution optimization and a negative optimizer comparison; a new IEEE optimizer-superiority claim is not supported.',
        f'The archive-free follow-up removed separately trained GPU predictor observations and reduced total target-or-cap GPU cost by {u.total_saving_vs_bpso:.2f}\\% relative to BPSO. It remained in SH fallback during the short matched searches, so this saving does not establish a contribution from online prediction or uncertainty. DEHB was the stronger measured control. The contribution is reproducible engineering and a candid optimizer comparison; a new IEEE optimizer-superiority claim is not supported.')
    updated=updated.replace('establishing net search-efficiency gains and deployment performance requires larger archive-inclusive matched-resource comparisons and independent evaluation data.',
        'establishing an adaptive-prediction advantage and deployment performance requires larger matched-resource comparisons and independent evaluation data.')
    body.write_text(updated)
    print('Built audited report and manuscript addition.',flush=True)


if __name__=='__main__':main()
