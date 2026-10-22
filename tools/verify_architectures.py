"""Verify held-out predictions, portable models, native checkpoints and API routing."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import sys,json,joblib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'web'))
from architectures import build,torch
from architecture_inference import predict_arch
from experiment import load,metrics
from app import app,ARCH,CAT
records=[];client=app.test_client();assert client.get('/health').json['models']==38
for key,models in ARCH.items():
    d=load(key);te=d['test'];x=d['x'][te]
    for name,item in models.items():
        folder=ROOT/'results'/key/'classical' if key in ['diabetes','housing'] else ROOT/'results/architectures'/key
        saved=np.load(folder/f'{name}_predictions.npz');np.testing.assert_array_equal(te,saved['indices'])
        with np.load(ROOT/'web/bundle'/item['file']) as z:w=dict(z)
        if key in ['diabetes','housing']:
            model=joblib.load(folder/f'{name}.joblib')
            native=np.log(np.maximum(model.predict_proba(x),1e-12)) if key=='diabetes' else model.predict(x)[:,None]
        else:
            model=build(name,d['x'].shape[1:],item['config']['outputs']);model.load_state_dict(torch.load(folder/f'{name}.pth',weights_only=True));model.eval()
            with torch.no_grad():native=np.concatenate([model(torch.tensor(a)).numpy() for a in np.array_split(x,max(1,len(x)//128))])
        portable=np.concatenate([predict_arch(w,item['config'],a) for a in np.array_split(x,max(1,len(x)//128))])
        np.testing.assert_allclose(native,saved['logits'],atol=1e-4,rtol=1e-4)
        np.testing.assert_allclose(portable,native,atol=1e-4,rtol=1e-4)
        recalculated=metrics(d,te,native)
        for m,v in recalculated.items():np.testing.assert_allclose(v,item['test'][m],atol=1e-5,rtol=1e-5)
        for e in CAT[key]['examples']:
            payload=dict(dataset=key,model=name,framework=item['framework'],example=e['id'])
            r=client.post('/api/predict',json=payload);assert r.status_code==200,r.json
        records.append(dict(dataset=key,model=name,test_samples=len(te),max_export_error=float(abs(portable-native).max())))
        print('Verified',key,name,flush=True)
invalid=[dict(dataset='mnist',model='not_a_model',framework='pytorch'),dict(dataset='mnist',model='vgg_style',framework='keras'),dict(dataset='housing',model=['RandomForest'],framework='scikit-learn')]
for p in invalid:assert client.post('/api/predict',json=p).status_code==400
report=dict(status='passed',extra_models=20,models_total=38,records=records,extra_api_examples=120,invalid_cases=3)
(ROOT/'results/architecture_verification.json').write_text(json.dumps(report,indent=2));print('All architecture verification passed')
