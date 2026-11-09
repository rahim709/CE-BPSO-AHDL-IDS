"""Render the completed validation extension; --snapshot works without run dirs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import hashlib
import numpy as np
import pandas as pd
from src.evaluation.metrics import classification_metrics
from src.utils.core import save_json
from scripts.build_submission_assets import ROOT, EVIDENCE, table, figure
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',action='store_true')
    args=parser.parse_args()
    if not args.snapshot:
        source=Path('results/research_extension')
        assert json.loads((source/'progress.json').read_text())['state']=='complete', 'Extension is incomplete'
        for name in ['protocol','environment','measurements','ranking_results','learning_curves','progress']:
            shutil.copyfile(source/f'{name}.json',EVIDENCE/f'extended_{name}.json')
        group=Path('results/state_grouped_validation')
        assert json.loads((group/'complete.json').read_text())['completed']
        for name in ['protocol','environment','measurements','split_audit']:
            shutil.copyfile(group/f'{name}.json',EVIDENCE/f'grouped_{name}.json')
        shutil.copyfile(group/'data/metadata.json',EVIDENCE/'grouped_data_metadata.json')
        shutil.copyfile('results/submission_review/extension_tests.xml',EVIDENCE/'tests_extension.xml')
    rows=json.loads((EVIDENCE/'extended_measurements.json').read_text())
    groups=json.loads((EVIDENCE/'grouped_measurements.json').read_text())
    group_meta=json.loads((EVIDENCE/'grouped_data_metadata.json').read_text())
    protocol=json.loads((EVIDENCE/'extended_protocol.json').read_text())
    ranks=json.loads((EVIDENCE/'extended_ranking_results.json').read_text())
    curves=json.loads((EVIDENCE/'extended_learning_curves.json').read_text())
    assert len(rows)==85 and len(groups)==9
    if not args.snapshot:
        audited=[]
        for row in rows:
            path=Path(row['run'])/f"validation_epoch_{row['epoch']}.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['prediction_sha256']
            with np.load(path) as p:
                measured=classification_metrics(p['y'],p['probabilities'].argmax(1),[str(i) for i in range(8)],'0')
            assert np.isclose(measured['macro_f1'],row['metrics']['macro_f1'],atol=1e-12,rtol=0)
            assert measured['confusion_matrix']==row['metrics']['confusion_matrix']
            audited.append({'file':str(path),'matches':True,'sha256':row['prediction_sha256']})
        for row in groups:
            path=Path('results/state_grouped_validation')/f"{row['method']}_seed_{row['seed']}_validation.npz"
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['prediction_sha256']
            with np.load(path) as p:
                measured=classification_metrics(p['y'],p['prediction'],[str(i) for i in range(8)],'0')
            assert measured['confusion_matrix']==row['metrics']['confusion_matrix']
            audited.append({'file':str(path),'matches':True,'sha256':row['prediction_sha256']})
        save_json(EVIDENCE/'extended_prediction_audit.json',audited)
    def select(name,view,epoch=30):
        return [r for r in rows if r['preset']==name and r['normalization']==view and r['epoch']==epoch]
    def stats(items,key):
        vals=np.array([r['metrics'][key] for r in items])
        return float(vals.mean()),float(vals.std(ddof=1)) if len(vals)>1 else None
    def value(items,key):
        mean,sd=stats(items,key)
        return f'{mean:.4f}' + (f' $\\pm$ {sd:.4f}' if sd is not None else ' (n=1)')
    views=[('column_l2','Column L2'),('minmax','Min--Max'),('row_l2','Row L2')]
    table('extended_normalization','lrrr','Scaling & Macro-F1 & Accuracy & FPR',[
        f"{label} & {value(select('reference',view),'macro_f1')} & {stats(select('reference',view),'accuracy')[0]:.4f} & "
        f"{np.mean([r['metrics']['attack_detection']['fpr'] for r in select('reference',view)]):.4f}" for view,label in views])
    table('extended_components','lrrrr','Model & Macro-F1 & Accuracy & FPR & Params.',[
        f"{label} & {value(select(name,'minmax'),'macro_f1')} & {stats(select(name,'minmax'),'accuracy')[0]:.4f} & "
        f"{np.mean([r['metrics']['attack_detection']['fpr'] for r in select(name,'minmax')]):.4f} & {select(name,'minmax')[0]['parameters']:,}"
        for name,label in [('mlp','MLP'),('cnn','CNN'),('gru','GRU'),('reference','CNN--GRU')]])
    def number(v): return '---' if v is None else f'{v:.4f}'
    def regret(row):
        return (number(row['selection_regret']) if row['selection_regret'] is not None
                else '['+', '.join(f'{v:.3f}' for v in row['selection_regret_range'])+']')
    table('extended_rankings','rrrrr','Seed & Early epoch & Spearman & Kendall & Regret',[
        f"{r['seed']} & {r['early_epoch']} & {number(r['spearman'])} & {number(r['kendall'])} & {regret(r)}" for r in ranks])
    table('extended_long','lrrrr','Scaling & Seeds & Epochs & Macro-F1 & Accuracy',[
        f"{label} & {len(select('reference',view,200))} & 200 & {value(select('reference',view,200),'macro_f1')} & {stats(select('reference',view,200),'accuracy')[0]:.4f}"
        for view,label in views[:2]])
    def group_select(method): return [r for r in groups if r['method']==method]
    table('extended_grouped','lrrr','Model & Macro-F1 & Accuracy & FPR',[
        f"{label} & {value(group_select(name),'macro_f1')} & {stats(group_select(name),'accuracy')[0]:.4f} & "
        f"{np.mean([r['metrics']['attack_detection']['fpr'] for r in group_select(name)]):.4f}"
        for name,label in [('random_forest','Random forest'),('logistic_regression','Logistic'),('reference_hybrid','CNN--GRU')]])
    cohort=list(protocol['cohort'])
    table('extended_cohort','lrrr','Preset & Epoch 3 F1 & Epoch 10 F1 & Epoch 30 F1',[
        name.capitalize()+' & '+' & '.join(value(select(name,'column_l2',epoch),'macro_f1') for epoch in [3,10,30]) for name in cohort])
    summary={'normalization':{v:{'macro_f1':stats(select('reference',v),'macro_f1'),'accuracy':stats(select('reference',v),'accuracy')} for v,_ in views},
        'components':{n:{'macro_f1':stats(select(n,'minmax'),'macro_f1'),'parameters':select(n,'minmax')[0]['parameters']} for n in ['mlp','cnn','gru','reference']},
        'long':{v:{'n':len(select('reference',v,200)),'macro_f1':stats(select('reference',v,200),'macro_f1')} for v,_ in views[:2]},
        'grouped':{n:{'macro_f1':stats(group_select(n),'macro_f1'),'accuracy':stats(group_select(n),'accuracy')} for n in ['random_forest','logistic_regression','reference_hybrid']},
        'scope':'Means and sample SD across training seeds on fixed validation partitions; no significance claim or new held-out evaluation.'}
    save_json(EVIDENCE/'extended_summary.json',summary)
    macros={'ExtColumnThirty':stats(select('reference','column_l2'),'macro_f1')[0],
            'ExtMinmaxThirty':stats(select('reference','minmax'),'macro_f1')[0],
            'ExtRowThirty':stats(select('reference','row_l2'),'macro_f1')[0],
            'ExtColumnLong':stats(select('reference','column_l2',200),'macro_f1')[0],
            'ExtMinmaxLong':stats(select('reference','minmax',200),'macro_f1')[0],
            'ExtGroupedHybrid':stats(group_select('reference_hybrid'),'macro_f1')[0]}
    (ROOT/'extended_macros.tex').write_text('% Generated from completed validation measurements.\n'+
        '\n'.join('\\newcommand{\\'+key+'}{'+f'{val:.4f}'+'}' for key,val in macros.items())+'\n')
    table('extended_costs','lrr','Model & Parameters & Train + validation (s)',[
        f"{label} & {select(name,'minmax')[0]['parameters']:,} & {np.mean([r['training_validation_seconds'] for r in select(name,'minmax')]):.2f}"
        for name,label in [('mlp','MLP'),('cnn','CNN'),('gru','GRU'),('reference','CNN--GRU')]])
    pd.DataFrame([{'preset':r['preset'],'normalization':r['normalization'],'seed':r['seed'],'epoch':r['epoch'],
                  'macro_f1':r['metrics']['macro_f1'],'accuracy':r['metrics']['accuracy'],
                  'fpr':r['metrics']['attack_detection']['fpr'],'parameters':r['parameters']} for r in rows]).to_csv(EVIDENCE/'extended_individual_results.csv',index=False)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(3.5,2.4))
    for view,label in views:
        matrix=np.array([[r['macro_f1'] for r in curves[f'reference/{view}/{seed}'][:30]] for seed in [42,43,44]])
        x=np.arange(1,31); mean=matrix.mean(0);sd=matrix.std(0,ddof=1)
        ax.plot(x,mean,label=label.replace('--','-'))
        ax.fill_between(x,np.maximum(0,mean-sd),np.minimum(1,mean+sd),alpha=.12)
    ax.set(xlabel='Training epoch',ylabel='Validation macro-F1',ylim=(0,1));ax.legend(fontsize=8)
    figure(fig,'extended_normalization')
    fig,ax=plt.subplots(figsize=(3.5,2.4))
    for name in cohort:
        means=[stats(select(name,'column_l2',b),'macro_f1')[0] for b in [3,10,30]]
        ax.plot([3,10,30],means,marker='o',label=name.capitalize())
    ax.set(xlabel='Training epoch',ylabel='Mean validation macro-F1',xticks=[3,10,30],ylim=(0,1));ax.legend(fontsize=8)
    figure(fig,'extended_fidelity')
    fig,ax=plt.subplots(figsize=(3.5,2.4))
    for view,label,seeds in [('column_l2','Column L2: 3 seeds',[42,43,44]),('minmax','Min-Max: seed 42',[42])]:
        matrix=np.array([[r['macro_f1'] for r in curves[f'reference/{view}/{s}']] for s in seeds])
        ax.plot(np.arange(1,matrix.shape[1]+1),matrix.mean(0),label=label)
        if len(seeds)>1:
            sd=matrix.std(0,ddof=1);ax.fill_between(np.arange(1,201),np.maximum(0,matrix.mean(0)-sd),np.minimum(1,matrix.mean(0)+sd),alpha=.12)
    ax.set(xlabel='Training epoch',ylabel='Validation macro-F1',ylim=(0,1));ax.legend(fontsize=8)
    figure(fig,'extended_long')
    names=json.loads((EVIDENCE/'gas_class_names.json').read_text())['classes']
    table('extended_grouped_splits','llrr','Code & Class & Train & Validation',[
        f"{c} & {names[c]['abbreviation']} & {group_meta['splits']['train']['classes'][c]:,} & {group_meta['splits']['val']['classes'][c]:,}" for c in names])
    fig,ax=plt.subplots(figsize=(3.5,2.4))
    x=np.arange(8)
    for i,(view,label) in enumerate(views[:2]):
        items=select('reference',view)
        means=[np.mean([r['metrics']['per_class'][str(c)]['recall'] for r in items]) for c in x]
        ax.bar(x+(i-.5)*.36,means,.36,label=label.replace('--','-'))
    ax.set(xticks=x,xticklabels=[names[str(c)]['abbreviation'] for c in x],ylabel='Mean validation recall',ylim=(0,1.12))
    plt.setp(ax.get_xticklabels(),rotation=40,ha='right');ax.legend(fontsize=8)
    figure(fig,'extended_class_recall')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
