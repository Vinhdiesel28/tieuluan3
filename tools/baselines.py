from pathlib import Path
import sys,json,time,joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment import load,metrics
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
for key in ['diabetes','housing']:
    d=load(key);tr,va,te=d['train'],d['val'],d['test'];rows=[]
    out=d['out']/'classical';out.mkdir(exist_ok=True)
    models=[('LogisticRegression',LogisticRegression(max_iter=500,class_weight='balanced',random_state=42)),('RandomForest',RandomForestClassifier(n_estimators=100,max_depth=12,class_weight='balanced',random_state=42,n_jobs=2))] if key=='diabetes' else [('Ridge',Ridge(alpha=1)),('RandomForest',RandomForestRegressor(n_estimators=100,max_depth=12,random_state=42,n_jobs=2))]
    for name,model in models:
        start=time.perf_counter();model.fit(d['x'][tr],d['y'][tr]);seconds=time.perf_counter()-start
        pred=np.log(np.maximum(model.predict_proba(d['x'][te]),1e-12)) if key=='diabetes' else model.predict(d['x'][te])[:,None]
        val=np.log(np.maximum(model.predict_proba(d['x'][va]),1e-12)) if key=='diabetes' else model.predict(d['x'][va])[:,None]
        config=dict(name=name,kind='forest' if name=='RandomForest' else 'linear',task=d['meta']['task'])
        if name=='RandomForest':
            params={}
            for i,estimator in enumerate(model.estimators_):
                tree=estimator.tree_;v=tree.value[:,0,:]
                if key=='diabetes':v=v/np.maximum(v.sum(axis=1,keepdims=True),1e-12)
                for field,value in dict(left=tree.children_left,right=tree.children_right,feature=tree.feature,threshold=tree.threshold,value=v).items():params[f'{i}_{field}']=value
            config['trees']=len(model.estimators_)
        else:params=dict(coef=np.atleast_2d(model.coef_),intercept=np.atleast_1d(model.intercept_))
        np.savez_compressed(out/f'{name}.npz',**params);joblib.dump(model,out/f'{name}.joblib')
        np.savez_compressed(out/f'{name}_predictions.npz',indices=te,logits=pred)
        (out/f'{name}_config.json').write_text(json.dumps(config,indent=2))
        (out/f'{name}.json').write_text(json.dumps(dict(name=name,validation=metrics(d,va,val),test=metrics(d,te,pred),seconds=seconds),indent=2))
        rows.append(dict(Model=name,**metrics(d,te,pred)))
    pd.DataFrame(rows).to_csv(d['out']/'classical_baselines.csv',index=False);print(key,rows)
