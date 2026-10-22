from pathlib import Path
import json,time,copy,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,average_precision_score,roc_auc_score,mean_squared_error,mean_absolute_error,r2_score,confusion_matrix
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from scratch import initialize,forward,backward,objective,step,probabilities
from frameworks import TorchModel,keras_model,export,torch,tf
ROOT=Path(__file__).resolve().parents[1]
FRAMEWORKS=['scratch','keras','pytorch']
def load(key):
    z=np.load(ROOT/'data'/f'{key}.npz');d={k:z[k] for k in z.files};d['meta']=json.loads((ROOT/'results'/key/'summary.json').read_text(encoding='utf-8'));d['out']=ROOT/'results'/key;return d
def restore_target(d,indices,z):
    meta=d['meta'];v=z[:,0]*meta['y_std']+meta['y_mean']
    return d['last_close'][indices]*np.exp(v) if meta['key']=='stock' else np.expm1(v)
def metrics(d,ix,z):
    if d['meta']['task']=='regression':
        y=d['actual_close'][ix] if d['meta']['key']=='stock' else d['raw_y'][ix];p=restore_target(d,ix,z)
        return dict(RMSE=float(np.sqrt(mean_squared_error(y,p))),MAE=float(mean_absolute_error(y,p)),R2=float(r2_score(y,p)))
    y=d['y'][ix];prob=probabilities(z);pred=prob.argmax(1)
    m=dict(Accuracy=float(accuracy_score(y,pred)),MacroF1=float(f1_score(y,pred,average='macro',zero_division=0)))
    if prob.shape[1]==2:m.update(F1=float(f1_score(y,pred,zero_division=0)),Precision=float(precision_score(y,pred,zero_division=0)),Recall=float(recall_score(y,pred,zero_division=0)),AP=float(average_precision_score(y,prob[:,1])),ROC_AUC=float(roc_auc_score(y,prob[:,1])))
    return m
def predict(p,x,kind,batch=256):return np.concatenate([forward(p,x[i:i+batch],kind) for i in range(0,len(x),batch)])
def train(key,framework,seed=42):
    d=load(key);meta=d['meta'];kind=meta['kind'];task=meta['task'];x=d['x'];y=d['y'];tr,va=d['train'],d['val']
    outputs=len(meta['classes']) if task=='classification' else 1;p=initialize(kind,x.shape[1:],outputs,seed)
    epochs=20 if kind=='cnn' else 18;lr=.04 if kind=='cnn' else .02;batch=128
    weight=None
    if task=='classification':
        counts=np.bincount(y[tr].astype(int),minlength=outputs);weight=(len(tr)/(outputs*counts)).astype(np.float32)
    model=None
    if framework=='pytorch':
        model=TorchModel(p,kind);optimizer=torch.optim.SGD(model.parameters(),lr=lr)
        def update(xb,yb):
            optimizer.zero_grad();z=model(torch.tensor(xb))
            loss=((z[:,0]-torch.tensor(yb))**2).mean() if task=='regression' else (torch.nn.functional.cross_entropy(z,torch.tensor(yb,dtype=torch.long),reduction='none')*torch.tensor(weight)[torch.tensor(yb,dtype=torch.long)]).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();return float(loss.detach())
    elif framework=='keras':
        model=keras_model(p,kind,x.shape[1:]);optimizer=tf.keras.optimizers.SGD(learning_rate=lr,global_clipnorm=1.)
        @tf.function(reduce_retracing=True)
        def update(xb,yb):
            with tf.GradientTape() as tape:
                z=model(xb,training=True)
                loss=tf.reduce_mean(tf.square(z[:,0]-yb)) if task=='regression' else tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(yb,tf.int32),logits=z)*tf.gather(weight,tf.cast(yb,tf.int32)))
            optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables));return loss
    else:
        def update(xb,yb):
            z,cache=forward(p,xb,kind,True);loss,dz=objective(z,yb,task,weight);g=backward(p,cache,dz,kind);step(p,g,lr);return loss
    best=np.inf;history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(seed+epoch).permutation(tr);total=0
        for i in range(0,len(order),batch):
            ix=order[i:i+batch];total+=float(update(x[ix],y[ix]))*len(ix)
        if model is not None:p=export(model,framework,kind)
        val=objective(predict(p,x[va],kind),y[va],task,weight)[0]
        history.append(dict(epoch=epoch,train_loss=total/len(tr),val_loss=val))
        if val<best:best=val;bestp=copy.deepcopy(p);bestepoch=epoch
        print(key,framework,epoch,round(val,5),flush=True)
    seconds=time.perf_counter()-start;p=bestp;name=f'{framework}_{seed}';out=d['out'];np.savez_compressed(out/f'{name}.npz',**p)
    pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
    validation=metrics(d,va,predict(p,x[va],kind));test=predict(p,x[d['test']],kind);np.savez_compressed(out/f'{name}_predictions.npz',logits=test,indices=d['test'])
    row=dict(framework=framework,seed=seed,epochs=epochs,best_epoch=bestepoch,lr=lr,batch=batch,parameters=sum(v.size for v in p.values()),seconds=seconds,validation=validation,test=metrics(d,d['test'],test))
    (out/f'{name}.json').write_text(json.dumps(row,indent=2))
    if framework=='pytorch':torch.save(TorchModel(p,kind).state_dict(),out/f'{name}.pth')
    elif framework=='keras':keras_model(p,kind,x.shape[1:]).save(out/f'{name}.keras')
    return row

def summarize(key):
    d=load(key);out=d['out'];meta=d['meta'];rows=[json.loads((out/f'{fw}_42.json').read_text()) for fw in FRAMEWORKS]
    criterion='RMSE' if meta['task']=='regression' else 'MacroF1'
    selected=sorted(rows,key=lambda r:r['validation'][criterion],reverse=meta['task']=='classification')[0]['framework']
    (out/'selection.json').write_text(json.dumps(dict(framework=selected,criterion=criterion,split='validation')))
    table=pd.DataFrame([dict(Framework=r['framework'],**r['test'],Parameters=r['parameters'],TrainSeconds=r['seconds'],BestEpoch=r['best_epoch']) for r in rows]);table.to_csv(out/'comparison.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.2))
    for ax,fw in zip(axes,FRAMEWORKS):
        h=pd.read_csv(out/f'{fw}_42_history.csv');ax.plot(h.epoch,h.train_loss,label='train');ax.plot(h.epoch,h.val_loss,label='val');ax.set_title(fw);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    te=d['test'];z=np.load(out/f'{selected}_42_predictions.npz')['logits']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    if meta['task']=='classification':
        counts=np.bincount(d['y'].astype(int));axes[0].bar(np.arange(len(counts)),counts);axes[0].set_title('Benchmark label counts')
        cm=confusion_matrix(d['y'][te],z.argmax(1));axes[1].imshow(cm,cmap='Blues');axes[1].set_title(selected+' test confusion')
        for a in range(len(cm)):
            for b in range(len(cm)):axes[1].text(b,a,str(cm[a,b]),ha='center',va='center',fontsize=6)
        baseline_pred=np.full(len(te),np.bincount(d['y'][d['train']].astype(int)).argmax());base=dict(Accuracy=float(accuracy_score(d['y'][te],baseline_pred)),MacroF1=float(f1_score(d['y'][te],baseline_pred,average='macro',zero_division=0)))
        errors=np.flatnonzero(z.argmax(1)!=d['y'][te])[:8];examples=[dict(index=int(te[i]),actual=int(d['y'][te[i]]),prediction=int(z[i].argmax()),score=float(probabilities(z[i:i+1]).max())) for i in errors]
    else:
        actual=d['actual_close'][te] if key=='stock' else d['raw_y'][te];pred=restore_target(d,te,z)
        axes[0].hist(actual,bins=40);axes[0].set_title('Test target distribution')
        axes[1].scatter(actual,pred,s=5,alpha=.4);axes[1].plot([actual.min(),actual.max()],[actual.min(),actual.max()],'k--');axes[1].set(xlabel='Actual',ylabel='Prediction')
        b=d['last_close'][te] if key=='stock' else np.full(len(te),d['raw_y'][d['train']].mean());base=dict(RMSE=float(np.sqrt(mean_squared_error(actual,b))),MAE=float(mean_absolute_error(actual,b)),R2=float(r2_score(actual,b)))
        errors=np.argsort(abs(pred-actual))[-8:][::-1];examples=[dict(index=int(te[i]),actual=float(actual[i]),prediction=float(pred[i]),absolute_error=float(abs(pred[i]-actual[i]))) for i in errors]
    fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=150);plt.close(fig)
    (out/'baseline.json').write_text(json.dumps(base,indent=2));(out/'errors.json').write_text(json.dumps(examples,indent=2))
    # Three observed input/output examples, not predictions invented for the report.
    examples=[]
    for ix in te[:3]:examples.append(dict(index=int(ix),input=np.round(d['x'][ix].reshape(-1)[:8],3).tolist(),target=float(d['raw_y'][ix])))
    (out/'samples.json').write_text(json.dumps(examples,indent=2))
    if meta['kind']=='cnn':
        fig,axes=plt.subplots(2,5,figsize=(10,4))
        for label,ax in enumerate(axes.flat):
            i=te[np.flatnonzero(d['y'][te]==label)[0]];ax.imshow(d['images'][i].squeeze(),cmap='gray' if key=='mnist' else None);ax.set_title(meta['classes'][label],fontsize=8);ax.axis('off')
        fig.tight_layout();fig.savefig(out/'samples.png',dpi=150);plt.close(fig)
    return table

if __name__=='__main__':
    for key in sys.argv[1:] or ['diabetes','housing','mnist','eurosat','stock','customer']:
        for fw in FRAMEWORKS:train(key,fw)
        print(summarize(key),flush=True)
