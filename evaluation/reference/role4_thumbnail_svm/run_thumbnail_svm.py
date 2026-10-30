import sys,os,csv,json,time
from pathlib import Path
import numpy as np,pandas as pd,cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
sys.path.insert(0,'/mnt/data')
from svm_baseline import extract_features,C_VALUES,GAMMA_VALUES
base=Path(__file__).parent/'Dhan-Shomadhan'/'preprocessing'
root=base/'conservative_splits'/'bad260e9eda16df2'
out=Path('/mnt/data/role4_thumbnail_results');out.mkdir(exist_ok=True)
inv=pd.read_csv(root/'eligible_inventory.csv')
features={}
for j,r in enumerate(inv.itertuples(index=False)):
 p=base/'thumbnails'/f'{r.image_id}.jpg'
 img=cv2.imread(str(p))
 if img is None: raise RuntimeError(str(p))
 features[r.image_id]=extract_features(img).astype('float32')
 if j%250==0:print('features',j,flush=True)
rows=[];preds=[]; matrices={}
for seed in [11,22,33,44,55]:
 splits={k:pd.read_csv(root/f'seed_{seed}'/f'{k}.csv') for k in ['train','validation','test']}
 print(seed,[(k,v.shape) for k,v in splits.items()],flush=True)
 X={k:np.stack([features[z] for z in df.image_id]) for k,df in splits.items()}
 y={k:df.disease_id.to_numpy() for k,df in splits.items()}
 sc=StandardScaler().fit(X['train']);X={k:sc.transform(v) for k,v in X.items()}
 best=(-1,None,None,None)
 for c in [1]:
  for gamma in ["scale"]:
   model=SVC(kernel='rbf',C=c,gamma=gamma,class_weight='balanced',cache_size=400)
   model.fit(X['train'],y['train'])
   score=f1_score(y['validation'],model.predict(X['validation']),average='macro',labels=list(range(5)),zero_division=0)
   if score>best[0]:best=(score,c,gamma,model)
 pred=best[3].predict(X['test'])
 for i,(r,p) in enumerate(zip(splits['test'].itertuples(index=False),pred)):
  preds.append({'seed':seed,'image_id':r.image_id,'background':r.background,'true_label':int(r.disease_id),'pred_label':int(p)})
 for bg in ['white','field','all']:
  mask=np.ones(len(pred),dtype=bool) if bg=='all' else splits['test'].background.to_numpy()==bg
  yt=y['test'][mask]; yp=pred[mask]
  row={'seed':seed,'background':bg,'n':len(yt),'accuracy':accuracy_score(yt,yp),'macro_precision':precision_score(yt,yp,average='macro',labels=list(range(5)),zero_division=0),'macro_recall':recall_score(yt,yp,average='macro',labels=list(range(5)),zero_division=0),'macro_f1':f1_score(yt,yp,average='macro',labels=list(range(5)),zero_division=0),'best_C':best[1],'best_gamma':best[2]}
  rows.append(row)
  matrices[f'{seed}_{bg}']=confusion_matrix(yt,yp,labels=list(range(5))).tolist()
 print('finished',seed,best[:3],flush=True)
pd.DataFrame(rows).to_csv(out/'metrics_by_run.csv',index=False)
pd.DataFrame(preds).to_csv(out/'test_predictions.csv',index=False)
summary=pd.DataFrame(rows).groupby('background',sort=False)[['accuracy','macro_precision','macro_recall','macro_f1']].agg(['mean','std'])
summary.to_csv(out/'summary.csv')
(out/'confusion_matrices.json').write_text(json.dumps(matrices,indent=2))
print(summary.to_string(),flush=True)
