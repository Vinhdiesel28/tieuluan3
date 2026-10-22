"""NumPy forward/backward/SGD. No automatic differentiation in this module."""
import numpy as np

def initialize(kind,shape,outputs,seed=42):
    rng=np.random.default_rng(seed)
    def w(s,scale):return (rng.normal(size=s)*scale).astype(np.float32)
    if kind=='mlp':return {'w':w((shape[0],32),np.sqrt(2/shape[0])),'b':np.zeros(32,np.float32),'v':w((32,outputs),np.sqrt(1/32)),'c':np.zeros(outputs,np.float32)}
    if kind=='cnn':
        height,width,channels=shape;flat=((height-2)//2)*((width-2)//2)*8
        return {'w':w((3,3,channels,8),np.sqrt(2/(9*channels))),'b':np.zeros(8,np.float32),'v':w((flat,outputs),np.sqrt(1/flat)),'c':np.zeros(outputs,np.float32)}
    return {'w':w((shape[-1],16),np.sqrt(1/shape[-1])),'u':w((16,16),.15),'b':np.zeros(16,np.float32),'v':w((16,outputs),.2),'c':np.zeros(outputs,np.float32)}

def patches(x):
    return np.lib.stride_tricks.sliding_window_view(x,(3,3),axis=(1,2)).transpose(0,1,2,4,5,3)

def forward(p,x,kind,cache=False):
    if kind=='mlp':
        z=x@p['w']+p['b'];h=np.maximum(z,0);aux=(x,z,h)
    elif kind=='cnn':
        cols=patches(x);z=np.tensordot(cols,p['w'],axes=([3,4,5],[0,1,2]))+p['b']
        a=np.maximum(z,0);n,hh,ww,ch=a.shape
        h=a.reshape(n,hh//2,2,ww//2,2,ch).mean((2,4)).reshape(n,-1);aux=(cols,z,h)
    else:
        h=np.zeros((len(x),16),dtype=x.dtype);states=[h]
        for t in range(x.shape[1]):
            h=np.tanh(x[:,t]@p['w']+h@p['u']+p['b']);states.append(h)
        aux=(x,states,h)
    y=h@p['v']+p['c']
    return (y,aux) if cache else y

def objective(logits,y,task,weight=None):
    n=len(y)
    if task=='regression':
        delta=logits[:,0]-y
        return float(np.mean(delta**2)),(2*delta/n)[:,None]
    a=logits-logits.max(1,keepdims=True);exp=np.exp(a);prob=exp/exp.sum(1,keepdims=True)
    weights=np.ones(n) if weight is None else weight[y.astype(int)]
    loss=-(weights*(a[np.arange(n),y.astype(int)]-np.log(exp.sum(1)))).mean()
    grad=prob.copy();grad[np.arange(n),y.astype(int)]-=1;grad*=weights[:,None]/n
    return float(loss),grad

def backward(p,aux,dy,kind):
    x,z,h=aux;g={'v':h.T@dy,'c':dy.sum(0)};dh=dy@p['v'].T
    if kind=='mlp':
        dz=dh*(z>0);g.update(w=x.T@dz,b=dz.sum(0))
    elif kind=='cnn':
        n,hh,ww,ch=z.shape
        pooled=dh.reshape(n,hh//2,ww//2,ch)
        dz=np.repeat(np.repeat(pooled,2,axis=1),2,axis=2)/4*(z>0)
        g.update(w=np.tensordot(x,dz,axes=([0,1,2],[0,1,2])),b=dz.sum((0,1,2)))
    else:
        states=z;g.update(w=np.zeros_like(p['w']),u=np.zeros_like(p['u']),b=np.zeros_like(p['b']))
        for t in range(x.shape[1]-1,-1,-1):
            dz=dh*(1-states[t+1]**2)
            g['w']+=x[:,t].T@dz;g['u']+=states[t].T@dz;g['b']+=dz.sum(0)
            dh=dz@p['u'].T
    return g

def step(p,g,lr,clip=1.):
    norm=np.sqrt(sum(np.sum(v.astype(np.float64)**2) for v in g.values()))
    scale=min(1.,clip/(norm+1e-12))
    for k in p:p[k]-=lr*scale*g[k]

def probabilities(z):
    a=np.exp(z-z.max(1,keepdims=True));return a/a.sum(1,keepdims=True)
