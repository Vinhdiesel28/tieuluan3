from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment import load,metrics,forward,TorchModel,torch,tf,initialize
records=[]
for key in ['diabetes','housing','mnist','eurosat','stock','customer']:
    d=load(key);kind=d['meta']['kind'];tr,va,te=[d[k] for k in ['train','val','test']]
    assert not(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
    if key in ['stock','customer']:assert max(d['times'][tr])<min(d['times'][va]) and max(d['times'][va])<min(d['times'][te])
    for fw in ['scratch','keras','pytorch']:
        name=fw+'_42';p=dict(np.load(d['out']/f'{name}.npz'));saved=np.load(d['out']/f'{name}_predictions.npz');row=json.loads((d['out']/f'{name}.json').read_text());history=pd.read_csv(d['out']/f'{name}_history.csv')
        assert int(history.loc[history.val_loss.idxmin(),'epoch'])==row['best_epoch']
        np.testing.assert_array_equal(te,saved['indices']);recalculated=metrics(d,te,saved['logits'])
        for k,v in recalculated.items():np.testing.assert_allclose(v,row['test'][k],atol=1e-8)
        probe=d['x'][te[:32]];z=forward(p,probe,kind);np.testing.assert_allclose(z,saved['logits'][:32],atol=2e-5,rtol=2e-5)
        if fw=='pytorch':
            model=TorchModel(p,kind);model.load_state_dict(torch.load(d['out']/f'{name}.pth',weights_only=True));native=model(torch.tensor(probe)).detach().numpy()
        elif fw=='keras':native=tf.keras.models.load_model(d['out']/f'{name}.keras',compile=False)(probe).numpy()
        else:native=z
        np.testing.assert_allclose(z,native,atol=2e-5,rtol=2e-5);records.append(dict(dataset=key,framework=fw,max_native_error=float(abs(z-native).max())))
sys.path.insert(0,str(ROOT/'web'))
from app import app,CAT
client=app.test_client();assert client.get('/health').json['models']==38
valid=0
for key,v in CAT.items():
    for fw in ['scratch','keras','pytorch']:
        for e in v['examples']:
            r=client.post('/api/predict',json=dict(dataset=key,framework=fw,example=e['id']));assert r.status_code==200,r.json;valid+=1
            if 'csv' in e:
                s=client.post('/api/predict',json=dict(dataset=key,framework=fw,csv=e['csv']));assert s.status_code==200,s.json;valid+=1
                if 'probabilities' in r.json:np.testing.assert_allclose(r.json['probabilities'],s.json['probabilities'],atol=1e-5)
                else:np.testing.assert_allclose(r.json['prediction'],s.json['prediction'],atol=1e-4)
for p in [None,[],{},dict(dataset='bad',framework='scratch'),dict(dataset='diabetes',framework='keras',example=-1),dict(dataset='stock',framework='pytorch',csv='x\nNaN')]:assert client.post('/api/predict',json=p).status_code==400
books=[]
for path in (ROOT/'notebooks').glob('*.ipynb'):
    book=nbformat.read(path,as_version=4);cells=[c for c in book.cells if c.cell_type=='code'];assert all(c.execution_count is not None for c in cells),path.name
    assert not any(o.output_type=='error' for c in cells for o in c.outputs),path.name
    books.append(dict(notebook=path.name,cells=len(cells)))
result=dict(status='passed',models=18,checkpoints=records,api_valid=valid,api_invalid=6,notebooks=books,gradient_checks=json.loads((ROOT/'results/gradient_checks.json').read_text()))
(ROOT/'results/verification.json').write_text(json.dumps(result,indent=2));print('Verified 18 models,',valid,'valid API requests and',len(books),'notebooks')
