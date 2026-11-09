"""Actual quality/computation costs with explicit censoring and archive charges."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import digest, save_json

ROOT=Path('results/ua_amf_efficiency_20261108')


def analyze(root,followup=False):
    paths=sorted(root.glob('*/*/*/search_summary.json')) if followup else sorted(root.glob('*/*/*/*/search_summary.json'))
    records,curves,traces,verified=[],[],[],0
    for path in paths:
        folder=path.parent
        cfg=json.loads((folder/'resolved_config.json').read_text())
        s=json.loads(path.read_text())
        events=json.loads((folder/'evaluations.json').read_text())
        variant=folder.parent.name
        experiment=folder.parent.parent.name if followup else folder.parent.parent.parent.name
        target=cfg['search'].get('stop_f1')
        cap=cfg['compute']['max_gpu_seconds']
        if experiment not in ['quality','compute']:
            continue
        gpu=0.
        terminals=[]
        last_epoch_time=0.
        for event in events:
            assert event['charged_epochs']>=0
            assert gpu<cap+.001,'Training started after execution cap'
            gpu+=event['gpu_seconds']
            if event['fitness'] is None or event['completed_budget']!=30:
                continue
            row=dict(event,cumulative_gpu=gpu)
            terminals.append(row)
        assert abs(gpu-s['gpu_seconds'])<1e-8
        assert sum(e['charged_epochs'] for e in events)==s['charged_epochs']
        valid=[e for e in terminals if e['cumulative_gpu']<=cap and e['search_wall_seconds']<=cfg['compute']['max_seconds']]
        winner=max(valid,key=lambda e:e['fitness']) if valid else None
        hit=next((e for e in valid if target is not None and e['fitness']>=target),None)
        if winner:
            save_json(folder/'strict_winner.json',winner)
        for candidate in (folder/'candidates').glob('*/trajectory.json'):
            rows=json.loads(candidate.read_text())
            for pred in candidate.parent.glob('validation_epoch_*.npz'):
                epoch=int(pred.stem.split('_')[-1])
                row=next(r for r in rows if r['epoch']==epoch)
                with np.load(pred) as data:
                    meta=json.loads((Path(cfg['data']['processed'])/'metadata.json').read_text())
                    metrics=classification_metrics(data['y'],data['probabilities'].argmax(1),meta['classes'],'0')
                assert abs(metrics['macro_f1']-row['macro_f1'])<1e-12
                verified+=1
        decisions_path=folder/'dynamic_decisions.json'
        decisions=json.loads(decisions_path.read_text()) if decisions_path.exists() else []
        statepath=folder/'adaptive_state.json'
        state=json.loads(statepath.read_text()) if statepath.exists() else {}
        records.append({'experiment':experiment,'target':target,'variant':variant,'seed':s['seed'],'gpu_cap':cap,
                        'gpu_seconds':gpu,'wall_seconds':s['wall_seconds'],'target_reached':hit is not None,
                        'target_gpu_seconds':hit['cumulative_gpu'] if hit else None,'target_wall_seconds':hit['search_wall_seconds'] if hit else None,
                        'restricted_gpu_time':hit['cumulative_gpu'] if hit else cap,
                        'censoring_gpu_seconds':cap if not hit else None,
                        'macro_f1':winner['fitness'] if winner else None,'epochs':s['charged_epochs'],
                        'unique_architectures':s['unique_architectures'],'parameters':winner.get('parameters') if winner else None,
                        'numerical_failures':sum(bool(e['failure']) and 'time budget' not in e['failure'] for e in events),
                        'peak_gpu_memory_bytes':s['peak_gpu_memory_bytes'],'scheduler_seconds':s['scheduler_and_setup_seconds'],
                        'prediction_seconds':state.get('prediction_seconds',0.),
                        'promotions':sum(d.get('action')=='promote' for d in decisions),
                        'pruned_candidates':sum(d.get('action')=='prune_upper_bound' for d in decisions),
                        'gpu_overrun_seconds':max(0.,gpu-cap),
                        'winner_architecture':digest(winner['architecture']) if winner else None})
        if experiment=='compute':
            best_so_far=-1.
            for event in valid:
                best_so_far=max(best_so_far,event['fitness'])
                traces.append({'variant':variant,'seed':s['seed'],'gpu_seconds':event['cumulative_gpu'],'macro_f1':best_so_far})
            for limit in ([5.,10.,20.] if followup else [2.,4.,8.]):
                available=[e for e in terminals if e['cumulative_gpu']<=limit]
                best=max(available,key=lambda e:e['fitness']) if available else None
                curves.append({'variant':variant,'seed':s['seed'],'gpu_seconds':limit,'macro_f1':best['fitness'] if best else None})
    frame=pd.DataFrame(records)
    if frame.empty:
        return
    # Retain all seeds in summaries. No conditional-success-only cost averaging.
    frame.to_csv(root/'seed_results.csv',index=False)
    save_json(root/'seed_results.json',frame.astype(object).where(pd.notnull(frame),None).to_dict('records'))
    rows=[]
    for (experiment,target,variant),group in frame.groupby(['experiment','target','variant'],dropna=False):
        base=frame[(frame.experiment==experiment)&(frame.variant=='bpso')]
        if pd.notna(target):base=base[base.target==target]
        sh=frame[(frame.experiment==experiment)&(frame.variant=='bpso_sh')]
        if pd.notna(target):sh=sh[sh.target==target]
        saving=100*(base.gpu_seconds.mean()-group.gpu_seconds.mean())/base.gpu_seconds.mean()
        restricted_saving=100*(base.restricted_gpu_time.mean()-group.restricted_gpu_time.mean())/base.restricted_gpu_time.mean()
        rows.append({'experiment':experiment,'target':target,'variant':variant,'seeds':len(group),'gpu_seconds':group.gpu_seconds.mean(),
                     'gpu_sd':group.gpu_seconds.std(),'wall_seconds':group.wall_seconds.mean(),'macro_f1':group.macro_f1.mean(),
                     'f1_sd':group.macro_f1.std(),'selected_architectures':int(group.macro_f1.notna().sum()),'reached':int(group.target_reached.sum()),'restricted_gpu_time':group.restricted_gpu_time.mean(),
                     'saving_percent':saving,'restricted_time_saving_percent':restricted_saving,
                     'f1_difference_vs_sh':group.macro_f1.mean()-sh.macro_f1.mean(),'epoch_mean':group.epochs.mean(),
                     'historical_archive_gpu':403.8190207847874 if variant not in ['bpso','bpso_sh','hyperband','dehb'] else 0.})
    table=pd.DataFrame(rows)
    table['unamortized_gpu_seconds']=table.gpu_seconds+table.historical_archive_gpu
    table.to_csv(root/'comparison.csv',index=False)
    pd.DataFrame(curves).to_csv(root/'anytime.csv',index=False)
    save_json(root/'audit.json',{'verified_prediction_archives':verified,'all_match':True,'searches':len(frame),
                               'all_jobs_started_within_gpu_cap':True,'max_single_epoch_gpu_overrun':float(frame.gpu_overrun_seconds.max())})
    md=['| Experiment | Target | Algorithm | GPU s | Wall s | Macro-F1 (available winners) | Winners | GPU saving | Reached |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in table.itertuples():
        md.append(f'| {row.experiment} | {row.target if pd.notna(row.target) else "—"} | {row.variant} | {row.gpu_seconds:.3f} | {row.wall_seconds:.3f} | {row.macro_f1:.4f} | {row.selected_architectures}/{row.seeds} | {row.saving_percent:.1f}% | {row.reached}/{row.seeds} |')
    (root/'comparison.md').write_text('\n'.join(md)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    import matplotlib.pyplot as plt
    plot=pd.DataFrame(curves)
    fig,ax=plt.subplots(figsize=(9,5))
    if not plot.empty:
        for variant,group in plot.groupby('variant'):
            stats=group.groupby('gpu_seconds').macro_f1.agg(['mean','std','count'])
            stats=stats[stats['count']==3]
            ax.errorbar(stats.index,stats['mean'],yerr=stats['std'].fillna(0),marker='o',capsize=2,label=variant)
    ax.set(xlabel='Cumulative measured GPU elapsed budget (s)',ylabel='Best observed 30-epoch validation macro-F1',ylim=(0,1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root/'gpu_anytime.png',dpi=170)
    fig.savefig(root/'gpu_anytime.pdf')
    plt.close(fig)
    pd.DataFrame(traces).to_csv(root/'gpu_seed_traces.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    trace_frame=pd.DataFrame(traces)
    if not trace_frame.empty:
        colors=plt.get_cmap('tab10')
        for color,(variant,group) in enumerate(trace_frame.groupby('variant')):
            for index,(seed,trace) in enumerate(group.groupby('seed')):
                xs=trace.gpu_seconds.tolist()+[float(frame.gpu_cap.max())]
                ys=trace.macro_f1.tolist()+[trace.macro_f1.iloc[-1]]
                ax.step(xs,ys,where='post',alpha=.55,color=colors(color),label=variant if index==0 else None)
    ax.set(xlabel='Actual cumulative GPU elapsed time (s)',ylabel='Best actual30-epoch validation macro-F1',ylim=(0,1))
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root/'gpu_seed_traces.png',dpi=170)
    fig.savefig(root/'gpu_seed_traces.pdf')
    plt.close(fig)
    print(root,'verified',len(frame),'searches;',verified,'prediction archives',flush=True)


def main():
    global ROOT
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default=str(ROOT))
    args=parser.parse_args()
    ROOT=Path(args.input)
    analyze(ROOT)
    if (ROOT/'followup20').exists():analyze(ROOT/'followup20',True)
    profile_path=ROOT/'profile_results_completed.json'
    if not profile_path.exists():
        return
    profiles=json.loads(profile_path.read_text())
    data=pd.DataFrame(profiles)
    good=data[data.failure.isna()].copy()
    baseline=good[good.profile=='legacy'].set_index('index').gpu_seconds
    good['gpu_saving_percent']=[100*(baseline.loc[r.index]-r.gpu_seconds)/baseline.loc[r.index] for r in good.itertuples()]
    # Resumed last profile has a continuation-only wall measurement. Do not call
    # it full-trial wall time; the cumulative GPU epochs/time remain measured.
    good['wall_scope']='complete trial'
    good.loc[(good.profile=='combined')&(good['index']==10),'wall_scope']='continuation invocation only; partial initial wall not available'
    good.to_csv(ROOT/'profile_comparison.csv',index=False)


if __name__=='__main__':
    main()
