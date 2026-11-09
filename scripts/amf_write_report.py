"""Write the implementation results and manuscript section from verified artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json

STUDY = Path('results/ua_amf_prediction_20261108')
BENCH = Path('results/ua_amf_prediction_benchmark_20261108')
FINAL = Path('results/ua_amf_prediction_final_20261108')


def final_report():
    if not (FINAL / 'results.json').exists():
        return 'Final dedicated-model evaluation has not been executed.', None
    items = [r for r in json.loads((FINAL / 'results.json').read_text()) if r['status'] == 'completed']
    meta = json.loads((FINAL / 'data/metadata.json').read_text())
    for path in sorted((FINAL / 'models').glob('*/test_predictions.npz')):
        with np.load(path) as p:
            metrics = classification_metrics(p['y'], p['probabilities'].argmax(1), meta['classes'], '0')
        recorded = json.loads((path.parent / 'test_metrics.json').read_text())
        for key in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']:
            assert abs(metrics[key] - recorded[key]) < 1e-12
        assert metrics['confusion_matrix'] == recorded['confusion_matrix']
        assert abs(metrics['attack_detection']['fpr'] - recorded['attack_detection']['fpr']) < 1e-12
        for label in meta['classes']:
            assert abs(metrics['per_class'][label]['recall'] - recorded['per_class'][label]['recall']) < 1e-12
    rows = [{k: r[k] for k in ['variant', 'search_seed', 'training_seed', 'accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'parameters', 'inference_seconds_per_sample']} |
            {'normal_fpr': r['attack_detection']['fpr']} for r in items]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return 'Final-stage compute cap prevented complete models; see preserved checkpoints.', None
    frame.to_csv(FINAL / 'seed_metrics.csv', index=False)
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'normal_fpr', 'parameters', 'inference_seconds_per_sample']
    per_search = frame.groupby(['variant', 'search_seed'])[metrics].mean()
    per_search.to_csv(FINAL / 'by_search_seed.csv')
    aggregate = per_search.groupby('variant').agg(['mean', 'std'])
    aggregate.to_csv(FINAL / 'comparison.csv')
    save_json(FINAL / 'prediction_audit.json', {'verified_unique_models': len(list((FINAL / 'models').glob('*/test_predictions.npz'))),
                                               'all_metrics_match': True, 'complete_model_references': len(items), 'requested_model_references': 18})
    text = 'Final models use 30 epochs and dedicated training seeds 1501/1502/1503, frozen after validation selection. The test is the already examined historical split, **not a blind independent test**. Training uses the same 8192-row pilot subset; test rows are the historical full partition.\n\n'
    for name, row in aggregate.iterrows():
        n_search = len(per_search.loc[name])
        text += f"- {name}: {n_search} selected search-seed architectures; mean test macro-F1 {row[('macro_f1', 'mean')]:.4f}, sample SD of search-seed means {row[('macro_f1', 'std')]:.4f}; accuracy {row[('accuracy', 'mean')]:.4f}; normal FPR {row[('normal_fpr', 'mean')]:.4f}.\n"
    text += '\nPer-class recalls, precision, confusion matrices, parameter counts and measured inference latency are retained in each model directory and `results.json`. Repeated training seeds and repeated architectures are not independent datasets. Existing MLP/CNN/GRU/CNN-GRU results remain contextual, with different protocols; no matched detector-baseline superiority is inferred.\n'
    return text, aggregate


def main():
    reliability = json.loads((STUDY / 'reliability_summary.json').read_text())
    calibration = json.loads((STUDY / 'predictor_report.json').read_text())
    audit = json.loads((BENCH / 'prediction_audit.json').read_text())
    assert audit['all_match']
    comparison = pd.read_csv(BENCH / 'comparison.csv').set_index('variant')
    seed_results = pd.read_csv(BENCH / 'seed_results.csv')
    assert len(seed_results) == 24, 'Do not write a completed-study report from partial benchmark output'
    b, p, sh = comparison.loc['bpso'], comparison.loc['ua_amf_bpso'], comparison.loc['bpso_sh']
    final_text, final_table = final_report()
    targets = pd.read_csv(BENCH / 'time_to_target_summary.csv')
    paired_path = BENCH / 'paired_time_to_target.csv'
    paired = pd.read_csv(paired_path) if paired_path.stat().st_size > 1 else pd.DataFrame()
    report = '# Computational efficiency of predictive UA-AMF-BPSO\n\n'
    report += f"Completed: 25 sampled architectures; {reliability['completed_30_epoch_trajectories']}/75 terminal reliability trajectories; 24 timed searches (eight methods/ablations, three seeds). Reliability charged {reliability['charged_epochs']} epochs, {reliability['wall_seconds']:.3f} wall seconds and {reliability['gpu_seconds']:.3f} synchronized GPU seconds. Fourteen third-seed reliability trajectories remain unexecuted.\n\n"
    report += '## Early-fidelity reliability\n\n| Seed | Paired architectures | Early epoch | Spearman | Kendall | Top-5 recall | Regret |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for r in reliability['rankings']:
        value = lambda x: f'{x:.4f}' if x is not None else 'not identifiable (ties)'
        report += f"| {r['seed']} | {r['n']}{'' if r['cohort_complete'] else ' (partial prefix)'} | {r['early_budget']} | {value(r['spearman'])} | {value(r['kendall'])} | {value(r['top_k_overlap'])} | {value(r['selection_regret'])} |\n"
    report += '\nLate-blooming early-rejected candidates and regret tie ranges are exported separately. The incomplete third-seed prefix is not pooled as a complete replicate.\n\n## Timed baseline comparison\n\n'
    report += (BENCH / 'comparison.md').read_text() + '\n'
    report += f"The strict-30-second validation macro-F1 difference (full minus standard BPSO) is {p.f1 - b.f1:+.4f}; full minus ordinary halving is {p.f1 - sh.f1:+.4f}. GPU execution saving versus standard BPSO is {p.gpu_saving_percent:.2f}%; search-wall saving is {p.wall_saving_percent:.2f}%. Equal wall caps constrain total-time differences by design. A lower GPU figure alone is not a speedup if detection performance is worse. SDs describe three search seeds, not independent capture variability or confidence intervals.\n\n"
    report += '## Time-to-target and anytime behavior\n\n'
    report += 'The complete 10/20/30-second results are in `anytime_results.csv`; missing terminal models remain missing. Target hit counts and conditional hit times are in `time_to_target_summary.csv`, with censoring at 30 seconds. A target missed by either method does not receive a fabricated saving.\n\n'
    for target in [.2, .4, .6]:
        for name in ['bpso', 'bpso_sh', 'ua_amf_bpso']:
            row = targets[(targets.variant == name) & (targets.target == target)].iloc[0]
            report += f"- Target {target:.1f}, {name}: {int(row.attained)}/{int(row.total)} seeds attained; conditional mean hit time {row.conditional_mean_seconds:.3f} s.\n"
        if not paired.empty:
            hits = paired[paired.target == target]
            if len(hits):
                report += f"  Paired full-versus-BPSO saving: mean {hits.saving_percent.mean():.2f}% across {len(hits)} jointly attaining seeds only.\n"
    report += '\n## Predictor, promotions and ablations\n\n'
    for r in calibration['calibration']:
        report += f"At epoch {r['budget']}, held-out observation coverage is {r['empirical_observation_coverage']:.3f}, all-seed architecture coverage {r['empirical_architecture_all_seeds_coverage']:.3f}, mean width {r['mean_interval_width']:.4f}, MAE {r['mae']:.4f} ({r['test_architectures']} held-out architectures). Nominal coverage was 0.8.\n\n"
    report += 'The intervals are broad and conservatively covering in this small holdout; universal calibration on adaptive BPSO proposals is not demonstrated. `heldout_promotion_replay.csv` compares raw early rank, prediction, uncertainty and cost scoring on disjoint held-out architectures; it is offline analysis, not simulated detection evidence or timed search. `promotion_survivor_analysis.csv` checks later scores and continuation-cost predictions for terminal survivors only, which is selection-biased.\n\n'
    for name in ['bpso_prediction', 'bpso_prediction_uncertainty', 'ua_amf_bpso']:
        r = comparison.loc[name]
        report += f"- {name}: validation macro-F1 {r.f1:.4f} ± {r.f1_sd:.4f}; mean GPU execution {r.gpu:.3f} s; mean scheduling/setup {r.scheduler:.3f} s, predictor calls {r.predictor:.3f} s (already included in scheduling and search wall).\n"
    report += f"\nUncertainty-minus-prediction-only macro-F1 difference: {comparison.loc['bpso_prediction_uncertainty', 'f1'] - comparison.loc['bpso_prediction', 'f1']:+.4f}; cost-term difference (full minus uncertainty-only): {p.f1 - comparison.loc['bpso_prediction_uncertainty', 'f1']:+.4f}. These component contrasts are descriptive and cannot establish a causal benefit with three seeds.\n\n"
    report += f"\nOne-time predictor build cost is {calibration['total_build_wall_seconds']:.3f} wall seconds, in addition to the archive cost. Charging the complete archive to one predictive search gives {p.total_gpu:.3f} GPU seconds versus {b.gpu:.3f} for ordinary BPSO. Thus any online advantage must not be presented as demonstrated net end-to-end savings. No archive amortization or cross-task transfer is assumed.\n\n"
    report += '## Final detector evaluation\n\n' + final_text
    report += '\n## Verification and research interpretation\n\n'
    report += f"{audit['verified_archives']} benchmark prediction archives reproduce logged validation macro-F1; reliability endpoints have a separate audit. Epoch ledgers match summaries. The main table is generated from seed-level results; deadline overshoots, scheduler/predictor overhead, memory, unique architectures, promotions, controlled budget stops, numerical failures and retries are exported. Best/winner scores are actual epoch-30 evaluations, with strict wall cutoffs for comparisons.\n\n"
    report += 'The candidate combines established learning-curve prediction, forest ensembles, conformal residual calibration, cost scoring, checkpoint continuation and BPSO exploration. It is distinct in implementation from Hyperband bracket allocation, BOHB proposal models and DEHB differential evolution, but algorithmic novelty and superiority are not established. BOHB was discussed, not executed as a benchmark. Three seeds, one release, a reduced training subset, incomplete third-seed reliability, seven predictor-holdout architectures, interval width, adaptive distribution shift and archive cost prevent an IEEE superiority claim. No statistical significance or formal noninferiority is claimed.\n\n'
    report += 'The next justified study is an independently frozen, larger architecture calibration cohort, full-training candidate evaluations, more search seeds and matched total resources including archive amortization assumptions. Use a prespecified practical noninferiority margin and independent captures before making a detector-generalization claim. Existing historical results and unsuccessful pilot findings remain preserved.\n'
    (BENCH / 'REPORT.md').write_text(report)
    section = r'''\section{Computational Efficiency of Adaptive Multi-Fidelity BPSO}
\label{sec:amf-efficiency}
\subsection{Candidate method and resource accounting}
We extended the existing implementation with a learning-curve variant of UA-AMF-BPSO. It is an investigational combination of established components, not an established novel optimizer. At cumulative budget $b\in\{3,10\}$, a 64-tree forest predicts terminal validation macro-F1 $\mu_b(a)$ from canonical architecture bits, parameter count, observed learning-curve summaries, current loss and measured epoch cost. Features are restricted to epochs at or below $b$. Architecture indices 0--11 fit the predictor, 12--17 calibrate it, and 18--24 assess it; every seed of an architecture stays in the same group. Development-archive architectures are excluded from predictive search proposals.

For calibration architecture $a$, let $R_a=\max_s |F_{a,s,30}-\mu_b(a,s)|/s_b(a,s)$, where $s_b$ is forest prediction SD floored at 0.02. With $n_c$ calibration architectures and $\alpha=0.2$, use the $\lceil(n_c+1)(1-\alpha)\rceil$th calibration residual to scale $s_b$, producing radius $u_b$. Intervals are clipped to $[0,1]$. This group split-conformal construction follows established residual calibration principles~\cite{angelopoulos2021}; coverage on adaptive BPSO proposals is not guaranteed by the measured small random-architecture holdout.

Promotion uses
\begin{equation}
S(a,b)=\mu_b(a)+\beta u_b(a)-\lambda\widehat c(a,b\!\to\!30)/60,
\end{equation}
where $\widehat c$ is remaining epochs times observed mean epoch training/validation seconds. Cost is normalized by 60 seconds; F1 and radius are dimensionless. The fixed protocol uses $\beta=1$, $\lambda=0.05$, without tuning on held-out architectures or IDS test outcomes. Halving quotas remain fixed. Only actual epoch-30 scores update swarm attractors; lower budgets and predictions cannot become reported winners. Checkpoints retain model, optimizer and RNG states, decoded duplicates share training, and all predictive ablations use the same 20\% restart mechanism. Standard BPSO mechanics remain unchanged.

Search wall includes predictor loading, prediction and scheduling. Archive training and predictor fitting are charged separately and also shown unamortized. GPU execution denotes synchronized training/validation elapsed time, not kernel-profiler time or energy. GPU-cap enforcement conservatively charges full CUDA evaluation wall; epoch-boundary deadline checks permit a final-epoch overrun. Strict anytime comparisons exclude terminal observations completed after their stated wall limit.

\subsection{Reliability and held-out calibration}
The prespecified pilot sampled 25 distinct architectures from the full 73-bit space and used 8192 training and 4096 validation rows, preserving all eight labels and the original split. Parent training-fitted Min--Max scaling and balanced loss were frozen from earlier validation evidence. No test rows entered selection, preprocessing fitting or promotion. The cap was 480 wall seconds and 420 conservatively charged GPU seconds. It completed 61 of 75 planned terminal trajectories: all 25 architectures for seeds 42/43 and 11 for seed 44. Fourteen third-seed trajectories remain unexecuted; the partial prefix is not a complete replicate. Actual costs were 1830 epochs, 420.72 wall seconds and 403.82 GPU execution seconds.

For the first complete seed, Spearman correlation with epoch 30 was 0.363 at epoch 3 and 0.913 at epoch 10; selection regret was 0.1371 and 0.0199 respectively. The second complete seed and tie-aware Kendall/top-$k$/regret results are retained in the reproducibility artifacts. These observations support investigating early promotion reliability without assuming an uncertainty benefit. Seven held-out architectures (14 completed seed observations) gave empirical coverage 1.00 at both rungs for nominal 0.80 intervals. Mean widths were 0.699 at epoch 3 and 0.394 at epoch 10, with MAE 0.126 and 0.082. Broad intervals and the tiny holdout limit the evidence for useful calibration.

\subsection{Matched-time search and ablations}
Eight methods/ablations ran serially on the same 4 GB RTX 3050 Laptop GPU, with search seeds 501--503, population six, a common 30-epoch terminal fidelity and 30-second wall/GPU-charge caps. Method order was a seeded permutation. Anytime limits 10/20/30 seconds and F1 targets 0.2/0.4/0.6 were prespecified. Missing terminal models and unattained targets remain missing/censored. Prediction-only, prediction-plus-uncertainty and full cost-aware variants differ only in acquisition coefficients. Hyperband and DEHB keep native brackets; ordinary BPSO-SH uses 3/10/30.

\begin{table}[!t]\centering\caption{Measured three-seed search comparison. F1 is best actual 30-epoch validation F1 completed within 30 wall seconds. GPU saving uses mean measured execution time relative to BPSO; archive cost is excluded here and reported separately.}\label{tab:amf-compute}
\resizebox{\columnwidth}{!}{\input{../../results/ua_amf_prediction_benchmark_20261108/comparison.tex}}
\end{table}
\begin{figure}[!t]\centering
\includegraphics[width=\columnwidth]{../../results/ua_amf_prediction_benchmark_20261108/anytime_f1.pdf}
\caption{Best terminal-fidelity validation macro-F1 at fixed measured wall limits. Means and SD are shown only when all three search seeds have terminal models at that limit; otherwise the point is omitted. Raw seed traces are also exported. Error bars show SD, not confidence intervals.}\label{fig:amf-anytime}
\end{figure}
The full method obtained mean validation macro-F1 @@PF1@@ $\pm$ @@PSD@@, compared with @@BF1@@ $\pm$ @@BSD@@ for standard BPSO and @@SHF1@@ for ordinary halving. Mean online GPU execution saving versus BPSO was @@GSAVE@@\%, while search-wall saving was @@WSAVE@@\%; equal wall caps constrain total-time differences by design. Prediction-only and uncertainty-only ablations are included in Table~\ref{tab:amf-compute}. These descriptive three-seed outcomes do not establish statistical superiority or formal noninferiority.

@@TARGET_TEXT@@

\subsection{Limitations and final-model evidence}
@@FINAL_TEXT@@
The one-time archive consumed 403.82 GPU seconds; adding it to a single full-method search gives @@TOTALGPU@@ GPU seconds versus @@BGPU@@ for BPSO. Thus an online saving must not be described as a demonstrated net end-to-end saving without an explicit reuse/amortization assumption. Small architecture partitions, wide intervals, adaptive proposal shift, reduced rows and a coarse continuation-cost model remain limitations. The contribution is an implemented and measured candidate, not demonstrated novelty. Hyperband~\cite{li2018}, BOHB~\cite{falkner2018}, DEHB~\cite{awad2021} and learning-curve allocation already motivate closely related mechanisms. A larger frozen calibration cohort, full-training study, more search seeds and independent captures are needed before optimizer-superiority or detector-generalization claims.
'''
    target_text = ''
    for target in [.2, .4, .6]:
        btarget = targets[(targets.variant == 'bpso') & (targets.target == target)].iloc[0]
        ptarget = targets[(targets.variant == 'ua_amf_bpso') & (targets.target == target)].iloc[0]
        target_text += f'For target macro-F1 {target:.1f}, BPSO attained it in {int(btarget.attained)}/3 seeds and the full method in {int(ptarget.attained)}/3 seeds. '
    t02b = targets[(targets.variant == 'bpso') & (targets.target == .2)].iloc[0]
    t02p = targets[(targets.variant == 'ua_amf_bpso') & (targets.target == .2)].iloc[0]
    target_text += f'At target 0.2, attained by all seeds, mean hit time was {t02b.conditional_mean_seconds:.2f} seconds for BPSO and {t02p.conditional_mean_seconds:.2f} for the full method. '
    target_text += 'Conditional hit times and paired savings are exported with their attained-seed counts; unattained targets are censored at 30 seconds.'
    final_tex = 'A new final-model test evaluation was not completed; the previously examined historical test is not a blind independent set.'
    if final_table is not None:
        final_tex = 'Selected standard/full architectures were frozen from validation, then trained for 30 epochs with dedicated seeds 1501--1503. The evaluation uses the previously examined historical test, not a new blind independent set. '
        for name in ['bpso', 'ua_amf_bpso']:
            if name in final_table.index:
                final_tex += f"{'Standard' if name == 'bpso' else 'Full'} mean test macro-F1 was {final_table.loc[name, ('macro_f1', 'mean')]:.4f}, with normal-class FPR {final_table.loc[name, ('normal_fpr', 'mean')]:.4f}. "
        final_tex += 'Model-level precision, recall, confusion matrices, attack recalls, parameters and inference latency are retained; conventional MLP/CNN/GRU results remain contextual under their original protocols.'
    replacements = {'PF1': f'{p.f1:.4f}', 'PSD': f'{p.f1_sd:.4f}', 'BF1': f'{b.f1:.4f}', 'BSD': f'{b.f1_sd:.4f}',
                    'SHF1': f'{sh.f1:.4f}', 'GSAVE': f'{p.gpu_saving_percent:.2f}', 'WSAVE': f'{p.wall_saving_percent:.2f}',
                    'TOTALGPU': f'{p.total_gpu:.2f}', 'BGPU': f'{b.gpu:.2f}', 'TARGET_TEXT': target_text, 'FINAL_TEXT': final_tex}
    for key, value in replacements.items():
        section = section.replace('@@' + key + '@@', value)
    assert '@@' not in section
    Path('docs/latex/amf_efficiency.tex').write_text(section)
    body_path = Path('docs/latex/manuscript_body.tex')
    body = body_path.read_text()
    marker = r'\input{amf_efficiency}'
    if marker not in body:
        body = body.replace(r'\section{Limitations and Remaining Research}', marker + '\n\n' + r'\section{Limitations and Remaining Research}')
    body = body.replace('The four fixed presets do not replace the planned 25-architecture, three-seed 3/10/30 study or a complete 200-epoch cohort.',
                        'The four fixed presets and the new bounded 25-architecture pilot do not establish full-training, complete three-seed search reliability or a complete 200-epoch cohort.')
    body = body.replace('No new BPSO update or confidence-aware allocation rule is proposed.', r'A separate candidate extension adds architecture-disjoint learning-curve prediction and uncertainty-aware resource allocation, with bounded matched-time comparisons and ablations (Section~\ref{sec:amf-efficiency}). No novel BPSO update or established algorithmic novelty is claimed.')
    body = body.replace('Matched-resource search comparisons and independent evaluation remain necessary before claiming search-efficiency gains or deployment performance.', 'A further reduced-row, 25-architecture reliability pilot and 24 matched-time searches investigate predictive uncertainty-aware promotion; archive-inclusive efficiency and deployment claims still require larger studies and independent evaluation.')
    body = body.replace('establishing search-efficiency gains and deployment performance requires matched-resource search comparisons and independent evaluation data.', 'establishing net search-efficiency gains and deployment performance requires larger archive-inclusive matched-resource comparisons and independent evaluation data.')
    body_path.write_text(body)
    bib = Path('docs/latex/references.bib')
    bibliography = bib.read_text()
    if '@misc{angelopoulos2021,' not in bibliography:
        bibliography += '\n@misc{angelopoulos2021,\n author={Angelopoulos, Anastasios N. and Bates, Stephen},\n title={A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},\n year={2021}, eprint={2107.07511}, archivePrefix={arXiv}, url={https://arxiv.org/abs/2107.07511}}\n'
        bib.write_text(bibliography)
    save_json(BENCH / 'research_conclusion.json', {'complete_searches': 24, 'gpu_saving_percent_online': float(p.gpu_saving_percent),
              'wall_saving_percent_online': float(p.wall_saving_percent), 'validation_f1_difference': float(p.f1 - b.f1),
              'difference_vs_successive_halving': float(p.f1 - sh.f1), 'net_savings_with_full_archive_demonstrated': False,
              'statistical_superiority_claimed': False, 'novelty_established': False, 'independent_test_available': False})
    print('Verified report and manuscript section generated.')


if __name__ == '__main__':
    main()
