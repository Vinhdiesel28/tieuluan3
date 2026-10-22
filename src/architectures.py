"""Small, train-from-scratch architecture comparisons for the existing datasets.

VGG/MobileNet/ResNet-style networks illustrate architectural mechanisms. They
are deliberately reduced for 16x16 inputs, not full named ImageNet networks.
"""
import torch
from torch import nn

CNN_NAMES = ['basic_cnn', 'vgg_style', 'mobilenet_style', 'resnet_style']
RNN_NAMES = ['simple_rnn', 'lstm', 'gru', 'bilstm']
LABELS = dict(basic_cnn='CNN cơ bản', vgg_style='VGG-style',
              mobilenet_style='MobileNet-style', resnet_style='ResNet-style',
              simple_rnn='Simple RNN', lstm='LSTM', gru='GRU', bilstm='BiLSTM')

def cnn_spec(name, channels, outputs=10):
    def conv(key, a, b, k=3, pad=1, groups=1):
        return dict(op='conv', name=key, inputs=a, outputs=b, kernel=k, padding=pad, groups=groups)
    def dense(key, a, b): return dict(op='linear', name=key, inputs=a, outputs=b)
    relu=dict(op='relu'); pool=dict(op='pool')
    if name == 'basic_cnn':
        return [conv('conv', channels, 8, pad=0), relu, pool,
                dict(op='flatten'), dense('head', 8*7*7, outputs)]
    if name == 'vgg_style':
        return [conv('c1', channels, 16), relu, conv('c2', 16, 16), relu, pool,
                conv('c3', 16, 32), relu, conv('c4', 32, 32), relu, pool,
                dict(op='flatten'), dense('fc', 32*4*4, 64), relu, dense('head', 64, outputs)]
    if name == 'mobilenet_style':
        return [conv('stem', channels, 16), relu, dict(op='save'),
                conv('expand1', 16, 32, 1, 0), relu,
                conv('depth1', 32, 32, groups=32), relu,
                conv('project1', 32, 16, 1, 0), dict(op='add'), pool,
                conv('expand2', 16, 48, 1, 0), relu,
                conv('depth2', 48, 48, groups=48), relu,
                conv('project2', 48, 24, 1, 0), dict(op='gap'), dense('head', 24, outputs)]
    if name == 'resnet_style':
        return [conv('stem', channels, 16), relu, dict(op='save'),
                conv('r1a', 16, 16), relu, conv('r1b', 16, 16), dict(op='add'), relu, pool,
                conv('transition', 16, 32, 1, 0), dict(op='save'),
                conv('r2a', 32, 32), relu, conv('r2b', 32, 32), dict(op='add'), relu,
                dict(op='gap'), dense('head', 32, outputs)]
    raise ValueError(name)

class CNN(nn.Module):
    def __init__(self, spec):
        super().__init__(); self.spec=spec; self.layers=nn.ModuleDict()
        for s in spec:
            if s['op']=='conv':
                self.layers[s['name']]=nn.Conv2d(s['inputs'],s['outputs'],s['kernel'],
                                                padding=s['padding'],groups=s['groups'])
            elif s['op']=='linear':self.layers[s['name']]=nn.Linear(s['inputs'],s['outputs'])
    def forward(self, x):
        x=x.permute(0,3,1,2); residual=None
        for s in self.spec:
            op=s['op']
            if op in ['conv','linear']:x=self.layers[s['name']](x)
            elif op=='relu':x=torch.relu(x)
            elif op=='pool':x=nn.functional.avg_pool2d(x,2)
            elif op=='flatten':x=x.flatten(1)
            elif op=='gap':x=x.mean((2,3))
            elif op=='save':residual=x
            elif op=='add':x=x+residual
        return x

class Recurrent(nn.Module):
    def __init__(self, name, inputs, outputs, hidden=32):
        super().__init__(); self.name=name
        layer={'simple_rnn':nn.RNN,'lstm':nn.LSTM,'gru':nn.GRU,'bilstm':nn.LSTM}[name]
        self.rnn=layer(inputs,hidden,batch_first=True,bidirectional=name=='bilstm')
        self.head=nn.Linear(hidden*(2 if name=='bilstm' else 1),outputs)
    def forward(self,x):
        _, state=self.rnn(x)
        h=state[0] if isinstance(state,tuple) else state
        h=torch.cat((h[-2],h[-1]),dim=1) if self.name=='bilstm' else h[-1]
        return self.head(h)

def build(name, shape, outputs):
    return CNN(cnn_spec(name,shape[-1],outputs)) if name in CNN_NAMES else Recurrent(name,shape[-1],outputs)
