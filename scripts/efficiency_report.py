"""Verified course-project efficiency report and consistent IEEE manuscript update."""
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json

ROOT=Path('results/ua_amf_efficiency_20261108')
FOLLOW=ROOT/'followup20'
LABELS={'bpso':'BPSO','bpso_sh':'BPSO-SH','dehb':'DEHB','hyperband':'Hyperband','fixed_prediction':'Fixed prediction',
        'prediction_only':'Dynamic prediction','ua_no_cost':'UA without cost','ua_no_uncertainty':'UA without bonus','ua_full':'Full UA'}


def readable_numbers(text):
    protected={}
    def shield(match):
        token='PROTECTEDTOKEN'+chr(65+len(protected))+'END'
        protected[token]=match.group(0)
        return token
    text=re.sub(r'\\(?:cite|ref|label|url|path)\{[^}]+\}',shield,text)
    text=text.replace('FP32','FLOATFULL').replace('F1','FONE')
    lines=[]
    for line in text.splitlines():
        if r'\input{' not in line and r'\includegraphics' not in line:
            line=re.sub(r'(?<=[A-Za-z])(?=[0-9])', ' ', line)
            line=re.sub(r'(?<=[0-9])(?=[A-Za-z])', ' ', line)
        lines.append(line)
    result='\n'.join(lines).replace('FLOATFULL','FP32').replace('FONE','F1')
    result=result.replace('saves**','saves **').replace('is**','is **').replace('with**','with **').replace('is-0.','is -0.')
    for token,value in protected.items():
        result=result.replace(token,value)
    return result


def final_metrics():
    root=ROOT/'final'
    rows=[]
    data=Path('results/ua_amf_prediction_final_20261108/data')
    meta=json.loads((data/'metadata.json').read_text())
    results=json.loads((root/'results.json').read_text())
    verified=0
    for path in (root/'models').glob('*/test_predictions.npz'):
        with np.load(path) as pred:
            metrics=classification_metrics(pred['y'],pred['probabilities'].argmax(1),meta['classes'],'0')
        saved=json.loads((path.parent/'test_metrics.json').read_text())
        for key in ['accuracy','macro_precision','macro_recall','macro_f1']:
            assert abs(metrics[key]-saved[key])<1e-12
        assert metrics['confusion_matrix']==saved['confusion_matrix']
        assert abs(metrics['attack_detection']['fpr']-saved['attack_detection']['fpr'])<1e-12
        for label in meta['classes']:
            assert metrics['per_class'][label]['recall']==saved['per_class'][label]['recall']
        verified+=1
    for r in results:
        if r['status']!='completed':continue
        rows.append({k:r[k] for k in ['variant','search_seed','accuracy','macro_precision','macro_recall','macro_f1','parameters','inference_seconds_per_sample']} |
                    {'normal_fpr':r['attack_detection']['fpr']})
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'seed_metrics.csv',index=False)
    table=frame.groupby('variant').agg(n=('search_seed','count'),f1=('macro_f1','mean'),f1_sd=('macro_f1','std'),
                  accuracy=('accuracy','mean'),precision=('macro_precision','mean'),recall=('macro_recall','mean'),
                  normal_fpr=('normal_fpr','mean'),parameters=('parameters','mean'),latency=('inference_seconds_per_sample','mean'))
    table.to_csv(root/'comparison.csv')
    save_json(root/'audit.json',{'unique_test_archives_verified':verified,'completed_model_references':len(frame),
                               'planned_model_references':27,'all_metrics_match':True,
                               'limitation':'One dedicated training seed1601 per selected architecture; three search seeds share historical test rows.'})
    return table


def main():
    table=pd.read_csv(FOLLOW/'comparison.csv')
    quality=table[table.experiment=='quality'].set_index('variant')
    compute=table[table.experiment=='compute'].set_index('variant')
    b,u,sh,d=[quality.loc[x] for x in ['bpso','ua_full','bpso_sh','dehb']]
    final=final_metrics()
    profiles=pd.read_csv(ROOT/'profile_comparison.csv')
    original=profiles[profiles.profile=='legacy'].gpu_seconds.sum()
    optimized=profiles[profiles.profile=='combined'].gpu_seconds.sum()
    engine_saving=100*(original-optimized)/original
    calibration=json.loads((ROOT/'sparse_calibration.json').read_text())
    cache=json.loads((ROOT/'continuation_cache_check.json').read_text())
    assert all(r['exact_weights'] for r in json.loads((ROOT/'profile_results_completed.json').read_text()) if r['profile']=='combined')
    full_vs_sh=100*(sh.gpu_seconds-u.gpu_seconds)/sh.gpu_seconds
    # Fixed-pool and full-run contrasts are reported without significance claims.
    verdict={'full_quality_gpu_saving_vs_bpso_percent':float(u.saving_percent),
             'full_quality_gpu_saving_vs_sh_percent':float(full_vs_sh),
             'full_target_hits':int(u.reached),'bpso_target_hits':int(b.reached),'sh_target_hits':int(sh.reached),'dehb_target_hits':int(d.reached),
             'full_fixed_compute_f1':float(compute.loc['ua_full','macro_f1']),
             'sh_fixed_compute_f1':float(compute.loc['bpso_sh','macro_f1']),
             'dehb_fixed_compute_f1':float(compute.loc['dehb','macro_f1']),
             'full_f1_gap_vs_strongest_control':float(compute.loc['ua_full','macro_f1']-compute.loc['dehb','macro_f1']),
             'engineering_aggregate_gpu_saving_percent':float(engine_saving),
             'historical_archive_gpu_seconds':403.8190207847874,
             'new_predictor_archive_gpu_seconds':0.,'net_single_search_gpu_saving_including_archive':False,
             'full_uncertainty_bonus_f1_difference':float(compute.loc['ua_full','macro_f1']-compute.loc['ua_no_uncertainty','macro_f1']),
             'cost_term_f1_difference':float(compute.loc['ua_full','macro_f1']-compute.loc['ua_no_cost','macro_f1']),
             'best_free_control_fixed_compute':'dehb','statistical_superiority_claimed':False,
             'accept_full_as_best_optimizer':False,
             'reason':'Full UA has a finite-horizon online saving versus BPSO but is slower than SH, well below DEHB quality, and archive cost removes net saving.'}
    save_json(ROOT/'research_conclusion.json',verdict)
    md=['# Verified GPU-efficiency investigation','',
        'The new experiments separate fixed quality from fixed compute. All nine controls/ablations use the same FP32 runtime, 73-bit space, eight-class data view, common30-epoch selection fidelity, and seeds601/602/603. No test scores tune coefficients or promotions.','',
        '## Diagnosis before changes','',
        'The prior30-second benchmark intentionally exhausted equal wall caps, so equal GPU totals did not measure computational saving. It had no target-stop condition. Continuation already preserved optimizer/RNG; deduplication already worked. Predictor calls averaged0.10s, while archive training cost403.82GPU seconds. Cost prediction median error was1.9% on terminal survivors only. The original profile contained319 stream synchronizations and1251 scalar reads in three epochs; these counts include framework/optimizer activity and are not all model arithmetic.','',
        '## Engineering changes and measured preservation','',
        f'Vectorized batches preserve the original RandomSampler/DataLoader generator sequence. CPU-array and one-model caches are bounded; the model cache key includes the trial path, preventing cross-experiment weight reuse. Resident tensors reduce transfers. Loss scalar synchronization is deferred to the epoch boundary. Validation runs at required3/10/30 fidelities. Checkpoints still commit every completed epoch: less frequent writes were not adopted because they weaken hard-interruption recovery and do not directly reduce the original training/validation timer.','',
        f'Across three development architectures, summed GPU elapsed fell from{original:.3f} to{optimized:.3f}s: **{engine_saving:.2f}%**. All final FP32 tensors were exactly equal. These are one-pass development profiles in fixed order, not a population speedup guarantee; first-use effects may remain. The interrupted last profile resumed under an explicit30-second extension; its GPU time includes both parts, while its recorded wall field is continuation-only and is not used as full-trial wall time.','',
        f'AMP was profiled and rejected: it changed weights and shifted one development macro-F1 beyond0.01. Pinned-memory-only timing was inconsistent, so resident data was used instead. The continuation mechanism check reduced model constructions3→1 and checkpoint loads2→0, with exact weights. Its combined-runtime GPU times were{cache["records"][0]["gpu_seconds"]:.3f}→{cache["records"][1]["gpu_seconds"]:.3f}s; that one case does not isolate caching from other runtime changes.','',
        '## Dynamic allocation','',
        'The optimized scheduler initially probes two candidates, promotes by predicted final F1 plus a smaller uncertainty bonus minus median-normalized remaining cost, and forces another probe after at most two promotions. There is no rigid halving quota. It stops a candidate when its full calibrated upper bound is below the incumbent terminal score plus0.01. It never prunes solely on low initial F1. Only actual30-epoch scores update swarm attractors. Remaining cost separates training epochs from the number of future validation passes.','',
        'Beta0.1 and cost weight0.2 were frozen using architecture-disjoint three-fold development replay on archive IDs0–11. Replay costs are historical proxies, not new executed timing. The sparse, timing-invariant predictor was refit from existing logs; no new GPU label archive was trained. The held-out calibration set was not used to select coefficients. Nominal80% interval observation coverage was13/14 at3 epochs and14/14 at10, with broad mean widths0.573 and0.481; universal calibration under adaptive proposals is not established.','',
        '## Experiment A: fixed quality','',
        'The initial8-GPU-second pilot executed all three targets0.4/0.5/0.6 independently. Low standard/SH attainment motivated a declared exploratory continuation to20GPU seconds for the primary0.6 target. Every previous epoch, GPU second and wall second is retained; checkpoint-copy/replay overhead is charged. This extension is not a new independent replication and no coefficients or seeds were changed. All seeds—including misses—remain in the table. GPU times below are mean actual consumption to the target or deadline; unsuccessful cases are right-censored, not discarded.','',
        '| Algorithm | GPU s ± SD | Wall s | Macro-F1 ± SD | Saving vs BPSO | Reached0.6 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,row in quality.iterrows():
        md.append(f'| {LABELS[name]} | {row.gpu_seconds:.3f} ± {row.gpu_sd:.3f} | {row.wall_seconds:.3f} | {row.macro_f1:.4f} ± {row.f1_sd:.4f} | {row.saving_percent:.2f}% | {int(row.reached)}/3 |')
    md+=['',f'Full UA saves**{u.saving_percent:.2f}%** versus BPSO, just below the25% engineering goal, but is**{-full_vs_sh:.2f}% slower than SH**. It attains2/3 targets versus1/3 for BPSO and2/3 for SH. DEHB attains3/3 with**{d.saving_percent:.2f}%** less GPU than BPSO. These finite-horizon, three-seed results do not establish statistical superiority or unconditional expected time-to-target.',
         '', 'Restricted completion GPU time uses min(T,20), retaining misses at20; the full method saves25.14% on this different censored estimand. It must not replace the actual-consumption result24.72% to claim the25% goal was met. The full tables include both measures and explicit success counts.',
         '', '## Experiment B: fixed compute and ablations','',
         'All methods consume approximately20GPU seconds by design. These totals do not demonstrate saving; compare terminal quality and the5/10/20-second anytime curves. Mean/SD points require all three seeds to have a terminal model; raw seed traces also preserve missing early outcomes.','',
         '| Algorithm | Macro-F1 ± SD | GPU s |', '|---|---:|---:|']
    for name,row in compute.iterrows():md.append(f'| {LABELS[name]} | {row.macro_f1:.4f} ± {row.f1_sd:.4f} | {row.gpu_seconds:.3f} |')
    md+=['', f'Full fixed-compute F1 is{compute.loc["ua_full","macro_f1"]:.4f}, versus BPSO{compute.loc["bpso","macro_f1"]:.4f}, SH{compute.loc["bpso_sh","macro_f1"]:.4f}, and DEHB{compute.loc["dehb","macro_f1"]:.4f}. Its gap to DEHB is{verdict["full_f1_gap_vs_strongest_control"]:.4f}, exceeding the0.02 preservation margin. Full UA is therefore rejected as the best quality-preserving optimizer in this study.',
         '', f'The uncertainty bonus changes no selected terminal F1 relative to the no-bonus ablation. The cost term improves mean F1 by{verdict["cost_term_f1_difference"]:.4f} over no-cost, but does not make full UA beat SH on GPU time. Fixed-quota prediction is faster than dynamic prediction here; dynamic allocation is not independently supported as an improvement. Bound-pruning counts and promotions are retained in decision logs; skipped work alone is not evidence of a better search.',
         '', '## Predictor and end-to-end accounting','',
         f'New archive GPU cost is0; CPU development/refitting cost is recorded separately. Historical full archive cost is403.819GPU seconds (308.386 for the fit/calibration subset alone). Charging the full archive to one full-UA quality search gives{u.unamortized_gpu_seconds:.3f}GPU seconds versus BPSO{b.gpu_seconds:.3f}; **no net single-search saving remains**. Reusing historical logs is legitimate, but their computation is not free.',
         '', 'Smaller archive CPU refits use actual historical labels:8 fit/4 calibration architectures cost142.656 historical GPU seconds for that subset and retained broad intervals;4/4 cost96.386 but worsened prediction errors/widths. They were not substituted into the frozen benchmark after seeing outcomes. Lightweight online calibration was not adopted: very few terminal candidates at these budgets and adaptive sampling do not support an independently validated calibration set. No single-curve variation is described as a confidence interval.',
         '', '## Final detector evaluation','',
         'Every method’s fixed20-second winners were frozen before final evaluation. One dedicated training seed1601 and30 epochs are common to all methods; identical architectures share one final fit. The test is the previously examined historical split, not blind independent data. Existing MLP/CNN/GRU/reference results remain contextual.','',
         '| Algorithm | Test macro-F1 ± search-seed SD | Accuracy | Normal FPR |', '|---|---:|---:|---:|']
    for name,row in final.iterrows():md.append(f'| {LABELS[name]} | {row.f1:.4f} ± {row.f1_sd:.4f} | {row.accuracy:.4f} | {row.normal_fpr:.4f} |')
    md+=['','Precision/recall, attack-class recalls, confusion matrices, parameter counts and measured inference latency are retained per model. One final training seed and repeated historical rows limit generalization conclusions. High false-alarm rates prevent deployment claims.',
         '', '## Submission assessment','',
         'The experiments are reproducible course-project evidence of engineering optimization and the need for strong controls. They do not support superior/novel UA-AMF-BPSO. DEHB is the strongest measured free control; BPSO-SH is the simpler BPSO alternative and beats full UA on fixed-quality GPU cost. Do not present the exploratory continuation as confirmatory, discard misses, or omit archive cost.',
         '', '## Reproduction','', 'See docs/efficiency_optimization.md for exact commands, source/configuration hashes, caps and dependencies. Old validated results and the previous manuscript are retained separately.']
    (ROOT/'REPORT.md').write_text(readable_numbers('\n'.join(md))+'\n')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    names=list(quality.index)
    xs=np.arange(len(names))
    ax.bar(xs,[quality.loc[n,'gpu_seconds'] for n in names],yerr=[quality.loc[n,'gpu_sd'] for n in names],capsize=3)
    for i,name in enumerate(names):ax.text(i,quality.loc[name,'gpu_seconds']+.4,f'{int(quality.loc[name,"reached"])}/3',ha='center',fontsize=8)
    ax.set_xticks(xs,[LABELS[n] for n in names],rotation=25,ha='right')
    ax.set(ylabel='GPU elapsed to target or cap (s)',title='F1 target0.6; every seed retained; labels show attainment')
    fig.tight_layout()
    fig.savefig(ROOT/'target_cost.png',dpi=170)
    fig.savefig(ROOT/'target_cost.pdf')
    plt.close(fig)
    latex=[r'\begin{tabular}{lrrrr}\toprule',r'Method & GPU s & F1 & Saving (\%) & Hits\\\midrule']
    for name,row in quality.iterrows():latex.append(f'{LABELS[name]} & {row.gpu_seconds:.2f} & {row.macro_f1:.3f} & {row.saving_percent:.1f} & {int(row.reached)}/3'+chr(92)*2)
    latex.append(r'\bottomrule\end{tabular}')
    (ROOT/'quality_table.tex').write_text('\n'.join(latex)+'\n')
    section=r'''\section{Computational Efficiency of Adaptive Multi-Fidelity BPSO}
\label{sec:amf-efficiency}
\subsection{Diagnosis and experimental correction}
The earlier matched-wall study gave each method30 seconds with high iteration ceilings and no target-stop rule. It therefore measured quality at a common budget, not saving to a common quality: BPSO and UA each consumed about28.76 GPU seconds. Predictor calls cost about0.10 seconds; its archive cost403.82 GPU seconds. Continuation and decoded deduplication already worked. Per-example batching, scalar synchronization, dense validation and repeated model/checkpoint loading were execution overheads.

We distinguish fixed-quality and fixed-compute experiments. For target $\tau$, define $T(\tau)$ as cumulative measured GPU elapsed when an actually evaluated30-epoch candidate first reaches validation macro-F1 $\tau$. Misses are right-censored at budget $B$; restricted completion time is $\min(T,B)$. Every seed remains in the comparison. Fixed-compute experiments compare best actual30-epoch scores at common GPU limits and cannot themselves prove cost saving.

GPU elapsed retains the original convention: synchronized training/validation elapsed time, including host stalls within those intervals; it is not pure kernel time or energy. Predictor, scheduling, checkpoint and import overhead also appears in search wall time. GPU caps are checked at completed epoch boundaries, so a last epoch can overrun. Strict-budget score comparisons exclude terminal observations after the stated limit.

\subsection{Engineering and candidate policy}
Vectorized batches preserve the original sampler and RNG order. Bounded data/resident-tensor caches reduce loading/transfers, and a one-model cache keyed by trial directory avoids redundant reconstruction and optimizer loading during immediate continuation. FP32 loss-scalar synchronization is deferred. Validation runs at3/10/30 epochs; checkpoints still commit each epoch. Across three development architectures, cumulative GPU elapsed fell from25.059 to20.182 seconds (19.47\%), with identical final parameter tensors. Fixed-order one-pass profiles have limited precision. AMP changed outcomes beyond the0.01 development tolerance and was rejected; batch sizes/search space were unchanged.

The candidate initially probes two particles, then chooses continuations dynamically rather than using a fixed halving quota. A new probe is forced after at most two promotions. Its priority is
\begin{equation}
\begin{split}
P(a)={}&\widehat f_{30}(a)+\beta u(a)\\
&-\lambda\frac{\widehat c(a)}{\operatorname{median}_{z\in\mathcal A}\widehat c(z)}.
\end{split}
\end{equation}
where $\widehat c$ separates remaining training epochs from remaining validation passes. A candidate is stopped if its full prediction upper bound is below the best observed30-epoch score plus0.01; low initial F1 alone does not stop it. Only actual terminal scores update BPSO attractors. All predictive variants retain the same20\% diversity restart.

The sparse, timing-invariant forest is refit using existing logs. Architecture-disjoint development replay on IDs0--11 fixed $\beta=0.1$, $\lambda=0.2$ before the new search seeds. Calibration IDs12--17 and held-out IDs18--24 retain the split-conformal residual construction~\cite{angelopoulos2021}. Nominal80\% intervals covered13/14 held-out observations at3 epochs and14/14 at10, with mean widths0.573 and0.481. These broad intervals do not guarantee useful calibration under adaptive proposal shift. No new GPU archive was collected; historical collection cost is still charged.

\subsection{Measured quality, cost and ablations}
Nine methods/ablations used seeds601--603, the same8192/4096 training/validation view, all eight classes, original training-fitted Min--Max preprocessing, balanced loss, FP32 runtime and maximum fidelity30. The initial8-GPU-second pilot executed independent targets0.4/0.5/0.6 and fixed-compute runs. Low control attainment motivated a declared exploratory continuation to20 GPU seconds for target0.6 and fixed compute. It retained all methods/seeds/coefficients, resumed saved optimizer/RNG states, and charged every previously spent epoch, GPU second and wall second, including checkpoint-import overhead. It is not a new independent replication.

\begin{table}[!t]\centering\caption{Target0.6: actual mean GPU elapsed to target or cap, with all three seeds retained. Misses are censored rather than excluded. Scores are actual terminal validation macro-F1. Archive cost is separate.}\label{tab:amf-compute}
\resizebox{\columnwidth}{!}{\input{../../results/ua_amf_efficiency_20261108/quality_table.tex}}
\end{table}
Full UA used12.803 GPU seconds versus17.008 for BPSO: a24.72\% online reduction, just below the25\% engineering goal. It attained2/3 targets versus1/3 for BPSO. SH used12.103 seconds with2/3 hits, so full UA was5.78\% slower. DEHB used6.776 seconds, attained3/3 targets and saved60.16\% versus BPSO. These finite-horizon, three-seed results do not establish statistical superiority. The restricted-time saving25.14\% is a different censored estimand and must not replace the actual-consumption result to claim the25\% goal was met.

\begin{figure}[!t]\centering\includegraphics[width=\columnwidth]{../../results/ua_amf_efficiency_20261108/target_cost.pdf}
\caption{Actual target-or-cap GPU consumption, including misses. Error bars show search-seed SD; annotations give target attainment.}\end{figure}
\begin{figure}[!t]\centering\includegraphics[width=\columnwidth]{../../results/ua_amf_efficiency_20261108/followup20/gpu_anytime.pdf}
\caption{Fixed-compute quality at5/10/20 GPU seconds. Means/SD require all three seeds to have a terminal model; missing points are not replaced by zero. Raw seed traces are also retained.}\end{figure}
At fixed20-second compute, mean validation macro-F1 was0.5391 for full UA,0.2696 for BPSO,0.5030 for SH and0.7356 for DEHB. The full-method gap to DEHB was0.1965, exceeding the0.02 preservation margin. Removing the uncertainty bonus changed no terminal winner scores. Removing cost reduced mean F1 by0.0320, but cost-aware UA still did not beat SH on target-time cost. Fixed-quota prediction used11.940 target-or-cap GPU seconds and F1~0.5291; dynamic prediction used12.978 and0.5071. The more complex allocation is not independently supported as an improvement.

\subsection{Total cost, final models and limitations}
Charging the historical archive gives416.622 GPU seconds for one full-UA quality search versus17.008 for BPSO: no net single-search saving remains. The fit/calibration subset alone cost308.386 historical GPU seconds. Smaller-log CPU refits reduced that attributable subset but increased held-out point-prediction error; they were not substituted into the frozen benchmark. Historical reuse is not free computation. Online calibration was not claimed from a few adaptively selected terminal observations.

All methods' fixed-compute architecture choices were frozen before final testing, using dedicated seed1601 and30 epochs. The previously examined historical test is not blind independent data. @@FINAL@@ Precision, attack recalls, confusion matrices, parameters and measured inference latency are retained; one final training seed and shared historical rows limit generalization. Existing conventional MLP/CNN/GRU/reference results remain contextual.

The current regression suite passed 105 CPU and 50 targeted CUDA tests. Saved validation/test predictions were re-scored, epoch ledgers were checked, and the old manuscript and results were preserved.

The strongest measured free control is DEHB; SH is the simpler BPSO alternative and beats full UA on target-time cost. Runtime engineering is verified, but novelty and superior uncertainty-aware allocation are not established. The initial pilot and exploratory extension are reported separately; more search/final seeds, full-training studies, independently validated calibration and independent captures remain necessary.
'''
    finaltext=[]
    for name in ['bpso','bpso_sh','ua_full','dehb']:
        r=final.loc[name]
        finaltext.append(f'{LABELS[name]} test macro-F1 was{r.f1:.4f}, with normal FPR{r.normal_fpr:.4f}.')
    section=section.replace('@@FINAL@@',' '.join(finaltext))
    Path('docs/latex/amf_efficiency.tex').write_text(readable_numbers(section))
    bodypath=Path('docs/latex/manuscript_body.tex')
    body=bodypath.read_text()
    abstract=r'''\begin{abstract}
Neural architecture search for intrusion detection requires useful candidate evaluation under limited compute. This project retains a73-bit BPSO CNN--RNN framework and investigates an uncertainty-aware adaptive multi-fidelity extension. An earlier equal-wall-budget study found essentially no GPU-time saving. We diagnosed that protocol, added genuine target stopping and measured GPU-budget comparisons, and implemented vectorized/resident data, bounded continuation caches and sparse validation. Three development architectures retained identical FP32 weights while aggregate GPU elapsed fell19.47\%. Nine methods/ablations then used three search seeds. An8-second pilot and exploratory20-second continuation preserved checkpoint states and charged prior work. For validation macro-F1 target0.6 at30-epoch fidelity, full UA consumed24.72\% less online GPU than BPSO, attaining2/3 targets versus1/3. However, ordinary halving was5.78\% faster than full UA, and DEHB attained3/3 targets with60.16\% less GPU than BPSO. At fixed compute, full UA also trailed DEHB substantially. The historical predictor archive cost403.82 GPU seconds and removes net single-search savings. The uncertainty bonus supplied no measured winner-score benefit. Final evaluation used the development-known historical test, not an independent blind set. The evidence supports reproducible engineering and strong-control benchmarking, not novel or superior uncertainty-aware optimization.
\end{abstract}'''
    body=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda _:readable_numbers(abstract),body,count=1,flags=re.S)
    marker='\\section{Reproducibility}'
    addition='The new efficiency study adds explicit target stopping, actual GPU-budget comparisons and an exploratory charged continuation. Its controls, engineering profiles, censoring and archive-inclusive costs are reported in Section~\\ref{sec:amf-efficiency}; none establish superior uncertainty-aware optimization.\n\n'
    if addition not in body:body=body.replace(marker,addition+marker)
    conclusion='The efficiency follow-up found a limited online saving versus BPSO, but full UA did not beat halving or retain quality relative to DEHB. Historical archive cost removes net single-query gains. The contribution is therefore a reproducible course-project implementation, measured FP32 execution optimization and a negative optimizer comparison; a new IEEE optimizer-superiority claim is not supported.\n\n'
    if conclusion not in body:body=body.replace(r'\input{false_alarm_conclusion}',conclusion+r'\input{false_alarm_conclusion}')
    body=body.replace('An alternative cap bounds search wall time.', 'Resource caps can bound search wall time or measured GPU training/validation elapsed time. The fixed-quality protocol stops only after an actual30-epoch target is reached; fixed-compute comparisons use common GPU horizons.')
    body=body.replace('These are empirical and reproducibility contributions.', 'The follow-up adds FP32 execution optimization, target-stopping and actual-GPU-budget experiments. The evidence supports empirical and reproducibility contributions.')
    body=body.replace('an actual30-epoch','an actual 30-epoch')
    bodypath.write_text(body)
    print('Verified efficiency report and consistent manuscript section generated.',flush=True)


if __name__=='__main__':main()
