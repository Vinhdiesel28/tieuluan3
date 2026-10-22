"""NumPy deployment of exported PyTorch architectures (no training dependency)."""
import numpy as np

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-60,60)))

def conv(x,w,b,padding=0,groups=1):
    if padding:x=np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    k=w.shape[-1]; patch=np.lib.stride_tricks.sliding_window_view(x,(k,k),axis=(2,3))
    if groups==1:y=np.einsum('nchwij,ocij->nohw',patch,w,optimize=True)
    elif groups==x.shape[1] and w.shape[1]==1:
        y=np.einsum('nchwij,cij->nchw',patch,w[:,0],optimize=True)
    else:raise ValueError('Unsupported convolution groups')
    return y+b[None,:,None,None]

def predict_arch(weights,config,x):
    if config['kind'] in ['linear','forest']:
        if config['kind']=='linear':
            z=x@weights['coef'].T+weights['intercept']
            if config['task']=='regression':return z
            p=sigmoid(z[:,0]);return np.log(np.maximum(np.column_stack((1-p,p)),1e-12))
        values=[]
        for i in range(config['trees']):
            left=weights[f'{i}_left'];right=weights[f'{i}_right'];feature=weights[f'{i}_feature'];threshold=weights[f'{i}_threshold']
            node=np.zeros(len(x),dtype=int)
            while True:
                active=np.flatnonzero(left[node]>=0)
                if not len(active):break
                at=node[active];node[active]=np.where(x[active,feature[at]]<=threshold[at],left[at],right[at])
            values.append(weights[f'{i}_value'][node])
        z=np.mean(values,axis=0)
        return np.log(np.maximum(z,1e-12)) if config['task']=='classification' else z
    if config['kind']=='cnn':
        x=x.transpose(0,3,1,2);residual=None
        for s in config['spec']:
            op=s['op'];prefix='layers.'+s.get('name','')
            if op=='conv':x=conv(x,weights[prefix+'.weight'],weights[prefix+'.bias'],s['padding'],s['groups'])
            elif op=='linear':x=x@weights[prefix+'.weight'].T+weights[prefix+'.bias']
            elif op=='relu':x=np.maximum(x,0)
            elif op=='pool':
                n,c,h,w=x.shape;assert h%2==w%2==0
                x=x.reshape(n,c,h//2,2,w//2,2).mean((3,5))
            elif op=='flatten':x=x.reshape(len(x),-1)
            elif op=='gap':x=x.mean((2,3))
            elif op=='save':residual=x
            elif op=='add':x=x+residual
        return x
    states=[];name=config['name']
    for reverse in ([False,True] if name=='bilstm' else [False]):
        suffix='_reverse' if reverse else ''
        w=weights['rnn.weight_ih_l0'+suffix];u=weights['rnn.weight_hh_l0'+suffix]
        b=weights['rnn.bias_ih_l0'+suffix];c=weights['rnn.bias_hh_l0'+suffix]
        h=np.zeros((len(x),u.shape[1]),dtype=np.float32);cell=np.zeros_like(h)
        for t in (range(x.shape[1]-1,-1,-1) if reverse else range(x.shape[1])):
            a=x[:,t]@w.T+b;v=h@u.T+c
            if name=='simple_rnn':h=np.tanh(a+v)
            elif name in ['lstm','bilstm']:
                i,f,g,o=np.split(a+v,4,axis=1)
                cell=sigmoid(f)*cell+sigmoid(i)*np.tanh(g);h=sigmoid(o)*np.tanh(cell)
            else:
                ar,az,an=np.split(a,3,axis=1);vr,vz,vn=np.split(v,3,axis=1)
                r=sigmoid(ar+vr);z=sigmoid(az+vz);n=np.tanh(an+r*vn);h=(1-z)*n+z*h
        states.append(h)
    h=np.concatenate(states,axis=1)
    return h@weights['head.weight'].T+weights['head.bias']
