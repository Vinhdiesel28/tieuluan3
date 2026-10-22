from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scratch import *
from frameworks import TorchModel,keras_model,torch,tf
rows=[]
for kind,shape in [('mlp',(5,)),('cnn',(8,8,2)),('rnn',(4,3))]:
    p=initialize(kind,shape,3);rng=np.random.default_rng(73);x=rng.normal(size=(3,*shape)).astype(np.float32);y=np.array([0,1,2]);weight=np.array([1.,.8,1.2],np.float32)
    z,cache=forward(p,x,kind,True);loss,dz=objective(z,y,'classification',weight);g=backward(p,cache,dz,kind)
    pt=TorchModel(p,kind);zp=pt(torch.tensor(x));losspt=(torch.nn.functional.cross_entropy(zp,torch.tensor(y),reduction='none')*torch.tensor(weight[y])).mean();losspt.backward()
    kt=keras_model(p,kind,shape)
    with tf.GradientTape() as tape:
        zk=kt(x);losskt=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=zk)*weight[y])
    grads=tape.gradient(losskt,kt.trainable_variables);kg=dict(zip(['w','u','b','v','c'] if kind=='rnn' else ['w','b','v','c'],[v.numpy() for v in grads]))
    error=0
    for k in p:
        np.testing.assert_allclose(g[k],pt.params[k].grad.numpy(),atol=3e-6,rtol=3e-4)
        np.testing.assert_allclose(g[k],kg[k],atol=3e-6,rtol=3e-4)
        error=max(error,float(np.abs(g[k]-kg[k]).max()))
    np.testing.assert_allclose(z,zp.detach().numpy(),atol=3e-6);np.testing.assert_allclose(z,zk.numpy(),atol=3e-6)
    # Central differences use float64 for an independent check.
    pd={k:v.astype(np.float64) for k,v in p.items()};xx=x.astype(np.float64);zz,cc=forward(pd,xx,kind,True);_,dd=objective(zz,y,'classification',weight);gg=backward(pd,cc,dd,kind);diff=0
    for k in pd:
        for index in [0,pd[k].size//2,pd[k].size-1]:
            old=pd[k].flat[index];eps=1e-5
            pd[k].flat[index]=old+eps;a=objective(forward(pd,xx,kind),y,'classification',weight)[0]
            pd[k].flat[index]=old-eps;b=objective(forward(pd,xx,kind),y,'classification',weight)[0];pd[k].flat[index]=old
            numerical=(a-b)/(2*eps);diff=max(diff,float(abs(numerical-gg[k].flat[index])))
            np.testing.assert_allclose(numerical,gg[k].flat[index],atol=1e-6,rtol=1e-4)
    rows.append(dict(kind=kind,framework_gradient_max_error=error,finite_difference_max_error=diff,status='passed'))
(ROOT/'results/gradient_checks.json').write_text(json.dumps(rows,indent=2));print(rows)
