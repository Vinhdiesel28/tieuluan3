"""Matched architectures with one canonical recurrent bias in every implementation."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import torch
from torch import nn
import tensorflow as tf
import numpy as np
torch.set_num_threads(2)

class TorchModel(nn.Module):
    def __init__(self,p,kind):
        super().__init__();self.kind=kind
        self.params=nn.ParameterDict({k:nn.Parameter(torch.tensor(v.copy())) for k,v in p.items()})
    def forward(self,x):
        p=self.params
        if self.kind=='mlp':h=torch.relu(x@p['w']+p['b'])
        elif self.kind=='cnn':
            a=nn.functional.conv2d(x.permute(0,3,1,2),p['w'].permute(3,2,0,1),p['b'])
            h=nn.functional.avg_pool2d(torch.relu(a),2).permute(0,2,3,1).reshape(len(x),-1)
        else:
            h=torch.zeros((len(x),16),dtype=x.dtype,device=x.device)
            for t in range(x.shape[1]):h=torch.tanh(x[:,t]@p['w']+h@p['u']+p['b'])
        return h@p['v']+p['c']

def keras_model(p,kind,shape):
    layers=tf.keras.layers
    inputs=layers.Input(shape=shape)
    if kind=='mlp':h=layers.Dense(32,activation='relu',name='hidden')(inputs)
    elif kind=='cnn':h=layers.Flatten()(layers.AveragePooling2D(2)(layers.Conv2D(8,3,activation='relu',name='hidden')(inputs)))
    else:h=layers.SimpleRNN(16,activation='tanh',name='hidden')(inputs)
    model=tf.keras.Model(inputs,layers.Dense(p['c'].size,name='head')(h))
    model.get_layer('hidden').set_weights([p['w'],p['u'],p['b']] if kind=='rnn' else [p['w'],p['b']])
    model.get_layer('head').set_weights([p['v'],p['c']]);return model

def export(model,framework,kind):
    if framework=='pytorch':return {k:v.detach().numpy().copy() for k,v in model.params.items()}
    values=model.get_layer('hidden').get_weights();p=dict(zip(['w','u','b'] if kind=='rnn' else ['w','b'],values));p['v'],p['c']=model.get_layer('head').get_weights();return p
