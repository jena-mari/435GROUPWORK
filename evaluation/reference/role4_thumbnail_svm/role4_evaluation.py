"""Role 4 evaluation. Requires numpy, matplotlib, scikit-learn.
Examples:
 python role4_evaluation.py --svm Dhan-Shomadhan/svm_results --frozen Dhan-Shomadhan/experiments/frozen_resnet18
 python role4_evaluation.py --svm Dhan-Shomadhan/svm_results --frozen Dhan-Shomadhan/experiments/frozen_resnet18 --finetuned Dhan-Shomadhan/experiments/finetuned_resnet18
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

CLASSES = ['Brown Spot','Leaf Scald','Rice Blast','Rice Tungro','Sheath Blight']
SEEDS = [11,22,33,44,55]

def read_predictions(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty predictions: {path}')
    if 'true_label' in rows[0]:
        norm=[dict(image_id=r['relative_path'], true=int(r['true_label']), pred=int(r['pred_label']), background=r['background'].strip().lower()) for r in rows]
    elif 'disease_id' in rows[0]:
        norm=[dict(image_id=r['image_id'], true=int(r['disease_id']), pred=int(r['predicted_disease_id']), background=str(r['background_id']).strip().lower()) for r in rows]
    else:
        raise ValueError(f'Unrecognised columns in {path}: {list(rows[0])}')
    for r in norm:
        if r['background'] in ('0','field','f'): r['background']='field'
        elif r['background'] in ('1','white','w'): r['background']='white'
        else: raise ValueError(f"Unknown background {r['background']} in {path}; verify coding")
        if r['true'] not in range(5) or r['pred'] not in range(5):
            raise ValueError(f'Invalid class in {path}')
    if len({r['image_id'] for r in norm}) != len(norm):
        raise ValueError(f'Duplicate image IDs in {path}')
    return norm

def locate(root, model, i, seed):
    if model=='SVM':
        return root / f'predictions_split_{i}.csv'
    candidates=[root / f'seed_{seed}' / 'test_predictions.csv', root / f'run_{i}' / 'test_predictions.csv']
    return next((p for p in candidates if p.exists()), candidates[0])

def metrics(rows):
    y=np.array([r['true'] for r in rows]); p=np.array([r['pred'] for r in rows])
    return dict(n=len(y),accuracy=accuracy_score(y,p),precision=precision_score(y,p,labels=list(range(5)),average='macro',zero_division=0),recall=recall_score(y,p,labels=list(range(5)),average='macro',zero_division=0),macro_f1=f1_score(y,p,labels=list(range(5)),average='macro',zero_division=0),matrix=confusion_matrix(y,p,labels=list(range(5))))

def write_csv(path, rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--svm',type=Path)
    parser.add_argument('--frozen',type=Path)
    parser.add_argument('--finetuned',type=Path)
    parser.add_argument('--out',type=Path,default=Path('role4_results'))
    args=parser.parse_args()
    sources={'SVM':args.svm,'Frozen ResNet18':args.frozen,'Fine-tuned ResNet18':args.finetuned}
    sources={k:v for k,v in sources.items() if v is not None}
    if not sources: parser.error('Supply at least one model results directory')
    args.out.mkdir(parents=True,exist_ok=True)
    individual=[]; aggregate=[]; manifests={}
    for model,root in sources.items():
        for i,seed in enumerate(SEEDS,1):
            path=locate(root,model,i,seed)
            if not path.is_file():
                print(f'MISSING {model} split {i}: {path}');continue
            rows=read_predictions(path)
            manifests[(model,seed)]={r['image_id']:r['true'] for r in rows}
            for scenario in ('white','field','all'):
                subset=rows if scenario=='all' else [r for r in rows if r['background']==scenario]
                if not subset: raise ValueError(f'No {scenario} images: {path}')
                m=metrics(subset)
                individual.append(dict(model=model,seed=seed,scenario=scenario,n=m['n'],**{k:round(m[k],6) for k in ('accuracy','precision','recall','macro_f1')}))
                aggregate.append((model,scenario,m['matrix']))
    if not individual: raise SystemExit('No prediction files found. Run models first and provide results directories.')
    write_csv(args.out/'per_run_metrics.csv',individual)
    summary=[]
    for model in sources:
        for scenario in ('white','field','all'):
            matches=[r for r in individual if r['model']==model and r['scenario']==scenario]
            if not matches: continue
            item=dict(model=model,scenario=scenario,runs=len(matches))
            for metric in ('accuracy','precision','recall','macro_f1'):
                values=np.array([r[metric] for r in matches])
                item[metric+'_mean']=round(float(values.mean()),6)
                item[metric+'_std']=round(float(values.std(ddof=1)),6) if len(values)>1 else ''
            summary.append(item)
            cm=sum((matrix for m,s,matrix in aggregate if m==model and s==scenario),np.zeros((5,5),dtype=int))
            fig,ax=plt.subplots(figsize=(8,7))
            ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,xticks_rotation=35,cmap='Blues',colorbar=False)
            ax.set_title(f'{model} - {scenario} (pooled test predictions)')
            fig.tight_layout()
            fig.savefig(args.out/(model.lower().replace(' ','_')+'_'+scenario+'_confusion.png'),dpi=160)
            plt.close(fig)
    write_csv(args.out/'five_run_summary.csv',summary)
    for seed in SEEDS:
        available=[(m,manifests[(m,seed)]) for m in sources if (m,seed) in manifests]
        for i in range(len(available)):
            for j in range(i+1,len(available)):
                a,ma=available[i];b,mb=available[j]
                if set(ma)!=set(mb) or ma!=mb:
                    print(f'WARNING: {a} and {b} seed {seed} do NOT have identical test IDs/labels; comparison may be unfair.')
    print(f'Wrote {args.out}/per_run_metrics.csv, five_run_summary.csv and confusion matrices')
    for r in summary:
        print(f"{r['model']:20} {r['scenario']:5} runs={r['runs']} accuracy={r['accuracy_mean']:.3f} macro-F1={r['macro_f1_mean']:.3f}")
    if any(r['runs']!=5 for r in summary):
        print('WARNING: Some results have fewer than 5 runs; not yet report-ready.')

if __name__=='__main__': main()
