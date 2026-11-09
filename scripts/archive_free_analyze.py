"""Verify actual outputs and report online/end-to-end costs without dropping misses."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.optimization.encoding import BinaryCodec
from src.optimization.curve_predictor import curve_features
from src.utils.core import digest, save_json

ROOT=Path('results/ua_amf_online_20261108')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',default=str(ROOT))
    args=parser.parse_args()
    root=Path(args.input)
    rows,anytime,trace,checks,gates=[],[],[],[],[]
    verified=0
    for path in sorted(root.glob('*/*/seed_*/search_summary.json')):
        folder=path.parent
        phase,variant=folder.parent.parent.name,folder.parent.name
        cfg=json.loads((folder/'resolved_config.json').read_text())
        s=json.loads(path.read_text())
        events=json.loads((folder/'evaluations.json').read_text())
        cap=cfg['compute']['max_gpu_seconds']
        gpu=0.
        terminal=[]
        for event in events:
            assert gpu<cap+.001,'A new training job started beyond the GPU cap'
            assert event['completed_budget']<=event['requested_budget']<=30
            gpu+=event['gpu_seconds']
            if event['fitness'] is not None and event['failure'] is None and event['completed_budget']==30:
                terminal.append(dict(event,cumulative_gpu=gpu))
        assert abs(gpu-s['gpu_seconds'])<1e-8
        assert sum(e['charged_epochs'] for e in events)==s['charged_epochs']
        valid=[r for r in terminal if r['cumulative_gpu']<=cap and r['search_wall_seconds']<=cfg['compute']['max_seconds']]
        winner=max(valid,key=lambda e:e['fitness']) if valid else None
        hit=next((e for e in valid if e['fitness']>=.6),None)
        if winner:save_json(folder/'strict_winner.json',winner)
        meta=json.loads((Path(cfg['data']['processed'])/'metadata.json').read_text())
        for candidate in (folder/'candidates').glob('*/trajectory.json'):
            trajectory=json.loads(candidate.read_text())
            for prediction in candidate.parent.glob('validation_epoch_*.npz'):
                budget=int(prediction.stem.split('_')[-1])
                expected=next(r['macro_f1'] for r in trajectory if r['epoch']==budget)
                with np.load(prediction) as p:
                    observed=classification_metrics(p['y'],p['probabilities'].argmax(1),meta['classes'])['macro_f1']
                assert abs(expected-observed)<1e-12
                verified+=1
        online_path=folder/'online_summary.json'
        info=json.loads(online_path.read_text()) if online_path.exists() else {}
        archive=403.8190207847874 if variant=='offline_ua' else 0.
        prep=info.get('predictor_preparation_seconds',None if variant=='offline_ua' else 0.)
        offline_state=json.loads((folder/'adaptive_state.json').read_text()) if variant=='offline_ua' else {}
        archive_wall=420.7223319799996 if variant=='offline_ua' else 0.
        rows.append({'phase':phase,'variant':variant,'seed':s['seed'],'gpu_cap':cap,'gpu_seconds':gpu,'wall_seconds':s['wall_seconds'],
                     'candidate_gpu_seconds':gpu,'separate_predictor_gpu_seconds':archive,'total_end_to_end_gpu_seconds':gpu+archive,
                     'separate_archive_wall_seconds':archive_wall,'search_plus_archive_wall_seconds':s['wall_seconds']+archive_wall,
                     'predictor_preparation_cpu_wall_seconds':prep,'predictor_inference_seconds':info.get('predictor_inference_seconds',offline_state.get('prediction_seconds',0.)),
                     'scheduler_and_setup_seconds':s['scheduler_and_setup_seconds'],
                     'macro_f1':winner['fitness'] if winner else None,'winner_parameters':winner.get('parameters') if winner else None,
                     'time_to_best_gpu_seconds':winner['cumulative_gpu'] if winner else None,
                     'time_to_best_wall_seconds':winner['search_wall_seconds'] if winner else None,
                     'target_reached':hit is not None,'target_gpu_seconds':hit['cumulative_gpu'] if hit else None,
                     'target_wall_seconds':hit['search_wall_seconds'] if hit else None,
                     'restricted_target_gpu_time':hit['cumulative_gpu'] if hit else cap,
                     'censored':hit is None,'epochs':s['charged_epochs'],'unique_architectures':s['unique_architectures'],
                     'promoted_architectures':len({e['candidate'] for e in events if e['completed_budget']>3}),
                     'cpu_fit_count':info.get('cpu_fit_count',0),'mature_architectures':info.get('mature_architectures',0),
                     'prediction_decisions':info.get('prediction_applied_decisions',0),'fallback_decisions':info.get('fallback_decisions',0),
                     'peak_gpu_memory_bytes':s['peak_gpu_memory_bytes'],
                     'numerical_failures':sum(bool(e['failure']) and 'time budget' not in e['failure'] for e in events),
                     'gpu_epoch_overrun_seconds':max(0.,gpu-cap)})
        if online_path.exists():
            assert 'predictor_bundle' not in cfg['search']
            observations=json.loads((folder/'online_observations.json').read_text())
            fits=json.loads((folder/'online_fit_audit.json').read_text())
            for fit in fits:
                assert not (set(fit['fit_ids']) & set(fit['calibration_ids']))
                assert not (set(fit['excluded_query_ids']) & set(fit['fit_ids']+fit['calibration_ids']))
                assert all(observations[k]['partition']=='fit' and observations[k]['target'] is not None for k in fit['fit_ids'])
                assert all(observations[k]['partition']=='calibration' and observations[k]['target'] is not None for k in fit['calibration_ids'])
            for key,record in observations.items():
                architecture=record['architecture']
                assert key==digest(architecture)
                # Recompute snapshots from the cutoff; no later row can enter.
                candidate_id=digest({'architecture':architecture,'seed':cfg['seed'],'training':cfg['training'],'data':meta['fingerprint']})
                trajectory=json.loads((folder/'candidates'/candidate_id/'trajectory.json').read_text())
                for budget,snapshot in record['snapshots'].items():
                    budget=int(budget)
                    before=[r for r in trajectory if r['epoch']<=budget]
                    previous=0 if budget==3 else 3
                    features=curve_features(architecture,before,budget,BinaryCodec(cfg['space']))
                    incremental=sum(r['training_seconds']+r['validation_seconds'] for r in before if r['epoch']>previous)
                    prefix=sum(r['training_seconds']+r['validation_seconds'] for r in before)
                    rebuilt=np.r_[features,np.log1p(incremental),np.log1p(prefix)]
                    assert np.allclose(rebuilt,snapshot['features'],rtol=1e-12,atol=1e-12)
                for budget,forecast in record['forecasts'].items():
                    assert record['partition']=='check' and forecast['made_before_target']
                    assert key not in forecast['fit_ids']+forecast['calibration_ids']
            decisions=json.loads((folder/'online_decisions.json').read_text())
            for decision in decisions:
                gates.append({'phase':phase,'variant':variant,'seed':cfg['seed'],'generation':decision['generation'],
                              'budget':decision['budget'],'prediction_applied':decision['prediction_applied'],**decision['gate']})
            checks.append({'path':str(folder),'fit_partitions_disjoint':True,'query_exclusions_verified':True,
                           'feature_cutoffs_verified':True,'no_offline_bundle':True})
        if phase=='compute':
            for limit in [5.,10.,20.]:
                available=[r for r in valid if r['cumulative_gpu']<=limit]
                best=max((r['fitness'] for r in available),default=None)
                anytime.append({'variant':variant,'seed':s['seed'],'gpu_budget':limit,'macro_f1':best})
            best=-1.
            for event in valid:
                best=max(best,event['fitness'])
                trace.append({'variant':variant,'seed':s['seed'],'gpu_seconds':event['cumulative_gpu'],'macro_f1':best})
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'seed_results.csv',index=False)
    save_json(root/'seed_results.json',frame.astype(object).where(pd.notnull(frame),None).to_dict('records'))
    pd.DataFrame(gates).to_csv(root/'online_gates.csv',index=False)
    pd.DataFrame(anytime).to_csv(root/'anytime.csv',index=False)
    pd.DataFrame(trace).to_csv(root/'seed_traces.csv',index=False)
    table=[]
    for (phase,variant),group in frame.groupby(['phase','variant']):
        base=frame[(frame.phase==phase)&(frame.variant=='bpso')]
        sh=frame[(frame.phase==phase)&(frame.variant=='bpso_sh')]
        if base.empty or sh.empty:continue
        table.append({'phase':phase,'variant':variant,'seeds':len(group),'gpu_mean':group.gpu_seconds.mean(),
                      'gpu_sd':group.gpu_seconds.std(),'wall_mean':group.wall_seconds.mean(),
                      'f1_mean':group.macro_f1.mean(),'f1_sd':group.macro_f1.std(),'selected':int(group.macro_f1.notna().sum()),
                      'reached':int(group.target_reached.sum()),'total_gpu_mean':group.total_end_to_end_gpu_seconds.mean(),
                      'prep_cpu_mean':group.predictor_preparation_cpu_wall_seconds.mean(),'epochs_mean':group.epochs.mean(),
                      'online_saving_vs_bpso':100*(base.gpu_seconds.mean()-group.gpu_seconds.mean())/base.gpu_seconds.mean(),
                      'total_saving_vs_bpso':100*(base.total_end_to_end_gpu_seconds.mean()-group.total_end_to_end_gpu_seconds.mean())/base.total_end_to_end_gpu_seconds.mean(),
                      'total_saving_vs_sh':100*(sh.total_end_to_end_gpu_seconds.mean()-group.total_end_to_end_gpu_seconds.mean())/sh.total_end_to_end_gpu_seconds.mean(),
                      'prediction_decisions':int(group.prediction_decisions.sum()),'cpu_fits':int(group.cpu_fit_count.sum())})
    summary=pd.DataFrame(table)
    summary.to_csv(root/'comparison.csv',index=False)
    # Charge a reusable archive once per campaign, rather than once per seed.
    campaigns=[]
    for (phase,variant),group in frame.groupby(['phase','variant']):
        archive=float(group.separate_predictor_gpu_seconds.max())
        campaigns.append({'phase':phase,'variant':variant,'seeds':len(group),'online_gpu_sum':group.gpu_seconds.sum(),
                          'archive_gpu_charged_once':archive,'campaign_gpu_total':group.gpu_seconds.sum()+archive,
                          'amortized_per_search_gpu':group.gpu_seconds.mean()+archive/len(group)})
    pd.DataFrame(campaigns).to_csv(root/'campaign_costs.csv',index=False)
    save_json(root/'leakage_audit.json',checks)
    save_json(root/'metric_budget_audit.json',{'verified_prediction_archives':verified,'all_metrics_match':True,
              'all_jobs_started_within_gpu_cap':True,'searches':len(frame),'no_external_online_gpu_archive':True})
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})
    import matplotlib.pyplot as plt
    if anytime:
        plot=pd.DataFrame(anytime)
        fig,ax=plt.subplots(figsize=(9,5))
        for variant,group in plot.groupby('variant'):
            stats=group.groupby('gpu_budget').macro_f1.agg(['mean','std','count'])
            stats=stats[stats['count']==3]
            ax.errorbar(stats.index,stats['mean'],yerr=stats['std'].fillna(0),marker='o',capsize=2,label=variant)
        ax.set(xlabel='Actual GPU elapsed budget (s)',ylabel='Best observed 30-epoch validation macro-F1',ylim=(0,1))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root/'anytime.pdf')
        fig.savefig(root/'anytime.png',dpi=170)
        plt.close(fig)
    quality=summary[summary.phase=='quality']
    if not quality.empty:
        fig,ax=plt.subplots(figsize=(9,5))
        x=np.arange(len(quality))
        ax.bar(x,quality.total_gpu_mean)
        ax.set_xticks(x,quality.variant,rotation=25,ha='right')
        ax.set(ylabel='Standalone total GPU elapsed including archive (s)',title='Target 0.6; all seeds retained, misses censored')
        for i,r in enumerate(quality.itertuples()):ax.text(i,r.total_gpu_mean+2,f'{r.reached}/3',ha='center',fontsize=8)
        fig.tight_layout()
        fig.savefig(root/'total_cost.pdf')
        fig.savefig(root/'total_cost.png',dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5))
        for variant,group in frame[frame.phase=='quality'].groupby('variant'):
            hits=np.sort(group.loc[group.target_reached,'target_gpu_seconds'].to_numpy())
            ax.step(np.r_[0,hits,20],np.r_[0,np.arange(1,len(hits)+1)/len(group),len(hits)/len(group)],where='post',label=variant)
        ax.set(xlabel='Online GPU elapsed (s)',ylabel='Fraction reaching terminal macro-F1 >=0.6',ylim=(0,1.05))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(root/'target_attainment.pdf')
        fig.savefig(root/'target_attainment.png',dpi=170)
        plt.close(fig)
    print('Verified',len(frame),'runs and',verified,'prediction archives.',flush=True)


if __name__=='__main__':main()
