"""Run controlled comparisons across architectures, separate from framework parity."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import sys,json,time,copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from experiment import load,metrics,restore_target
from architectures import build,CNN_NAMES,RNN_NAMES,LABELS,cnn_spec,torch
from architecture_inference import predict_arch
torch.set_num_threads(2)

def run(key,name,epochs=20):
    torch.manual_seed(42);np.random.seed(42)
    d=load(key);meta=d['meta'];task=meta['task'];tr,va,te=[d[s] for s in ['train','val','test']]
    out=ROOT/'results/architectures'/key;out.mkdir(parents=True,exist_ok=True)
    outputs=len(meta['classes']) if task=='classification' else 1
    config=dict(name=name,label=LABELS[name],kind=meta['kind'],shape=list(d['x'].shape[1:]),outputs=outputs,hidden=32)
    if meta['kind']=='cnn':config['spec']=cnn_spec(name,d['x'].shape[-1],outputs)
    model=build(name,d['x'].shape[1:],outputs);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    x=torch.tensor(d['x'],dtype=torch.float32)
    y=torch.tensor(d['y'],dtype=torch.long if task=='classification' else torch.float32)
    weights=None
    if task=='classification':weights=torch.tensor(len(tr)/(outputs*np.bincount(d['y'][tr].astype(int),minlength=outputs)),dtype=torch.float32)
    def loss_fn(z,target):
        return ((z[:,0]-target)**2).mean() if task=='regression' else (torch.nn.functional.cross_entropy(z,target,reduction='none')*weights[target]).mean()
    def predict(ix):
        model.eval()
        with torch.no_grad():return torch.cat([model(x[chunk]) for chunk in np.array_split(ix,max(1,int(np.ceil(len(ix)/256))))]).numpy()
    best=np.inf;best_state=None;history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train();order=np.random.default_rng(42+epoch).permutation(tr);total=0
        for i in range(0,len(order),128):
            ix=order[i:i+128];optimizer.zero_grad();loss=loss_fn(model(x[ix]),y[ix]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();total+=float(loss.detach())*len(ix)
        z=predict(va);vl=float(loss_fn(torch.tensor(z),y[va]))
        history.append(dict(epoch=epoch,train_loss=total/len(tr),val_loss=vl))
        if vl<best:best=vl;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
        print(key,name,epoch,round(vl,6),flush=True)
    seconds=time.perf_counter()-start;model.load_state_dict(best_state);model.eval()
    val=predict(va);test=predict(te);p={k:v.detach().numpy().copy() for k,v in best_state.items()}
    native=predict(te[:16]);portable=predict_arch(p,config,d['x'][te[:16]])
    np.testing.assert_allclose(portable,native,atol=3e-5,rtol=3e-5)
    torch.save(best_state,out/f'{name}.pth');np.savez_compressed(out/f'{name}.npz',**p)
    np.savez_compressed(out/f'{name}_predictions.npz',indices=te,logits=test)
    (out/f'{name}_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
    row=dict(name=name,label=LABELS[name],seed=42,epochs=epochs,best_epoch=best_epoch,optimizer='Adam',lr=.001,batch=128,
             parameters=sum(v.numel() for v in model.parameters()),seconds=seconds,validation=metrics(d,va,val),test=metrics(d,te,test),
             numpy_max_error=float(abs(native-portable).max()))
    (out/f'{name}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    return row

def summarize(key):
    d=load(key);out=ROOT/'results/architectures'/key;names=CNN_NAMES if d['meta']['kind']=='cnn' else RNN_NAMES
    rows=[json.loads((out/f'{n}.json').read_text(encoding='utf-8')) for n in names]
    criterion='RMSE' if d['meta']['task']=='regression' else 'MacroF1'
    chosen=min(rows,key=lambda r:r['validation'][criterion]) if criterion=='RMSE' else max(rows,key=lambda r:r['validation'][criterion])
    (out/'selection.json').write_text(json.dumps(dict(name=chosen['name'],criterion=criterion,split='validation'),indent=2))
    table=pd.DataFrame([dict(Model=r['name'],**r['test'],Parameters=r['parameters'],TrainSeconds=r['seconds'],BestEpoch=r['best_epoch']) for r in rows]);table.to_csv(out/'comparison.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(10,6))
    for ax,n in zip(axs.flat,names):
        h=pd.read_csv(out/f'{n}_history.csv');ax.plot(h.epoch,h.train_loss,label='train');ax.plot(h.epoch,h.val_loss,label='validation');ax.set_title(n);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3));ax.bar(table.Model,table[criterion]);ax.set_ylabel('Test '+criterion);ax.set_title(key+' / identical split, Adam 0.001, 20 epochs');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=140);plt.close(fig)
    actual=d['actual_close'][d['test']] if key=='stock' else d['y'][d['test']]
    z=np.load(out/f"{chosen['name']}_predictions.npz")['logits']
    if key=='stock':
        pred=restore_target(d,d['test'],z);indices=np.argsort(abs(pred-actual))[-6:][::-1]
    else:pred=z.argmax(1);indices=np.flatnonzero(pred!=actual)[:6]
    (out/'errors.json').write_text(json.dumps([dict(index=int(d['test'][i]),actual=float(actual[i]),prediction=float(pred[i])) for i in indices],indent=2))
    return table

if __name__=='__main__':
    for key in sys.argv[1:] or ['mnist','eurosat','stock','customer']:
        for name in CNN_NAMES if key in ['mnist','eurosat'] else RNN_NAMES:run(key,name)
        print(summarize(key).to_string(index=False),flush=True)
