from pathlib import Path
import json,sys,io,zipfile,hashlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split,GroupShuffleSplit
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'
def save(key,x,y,split,meta,extras=None):
    out=ROOT/'results'/key;out.mkdir(parents=True,exist_ok=True)
    meta.update(key=key,shape=list(x.shape[1:]),samples=len(y),splits={k:len(v) for k,v in split.items()})
    if meta['task']=='classification':meta['counts']={k:np.bincount(y[v].astype(int),minlength=len(meta['classes'])).tolist() for k,v in split.items()}
    np.savez_compressed(DATA/f'{key}.npz',x=x.astype(np.float32),y=y,**split,**(extras or {}))
    (out/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(key,meta['splits'],flush=True)

def split_random(y):
    ix=np.arange(len(y));tr,te=train_test_split(ix,test_size=.2,random_state=42,stratify=y)
    tr,va=train_test_split(tr,test_size=.2,random_state=42,stratify=y[tr]);return dict(train=tr,val=va,test=te)

def tabular(key):
    if key=='diabetes':
        df=pd.read_csv(DATA/'diabetes.csv');raw_count=len(df);df=df.drop_duplicates()
        target=(df.pop('Diabetes_012').to_numpy()>0).astype(np.int64)
        features=list(df);raw=df.to_numpy(np.float32);groups=pd.util.hash_pandas_object(df,index=False).to_numpy()
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(raw,target,groups))
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(raw[tr],target[tr],groups[tr]));split=dict(train=tr[a],val=tr[b],test=te)
        # Subset each split by labels after grouping; no identical predictor crosses splits.
        for k,n in [('train',16000),('val',4000),('test',5000)]:
            ix=split[k]
            if len(ix)>n:split[k]=train_test_split(ix,train_size=n,random_state=42,stratify=target[ix])[0]
        chosen=np.concatenate(list(split.values()));mapping={int(v):i for i,v in enumerate(chosen)}
        split={k:np.array([mapping[int(v)] for v in ix]) for k,ix in split.items()};raw=raw[chosen];y=target[chosen]
        meta=dict(kind='mlp',task='classification',classes=['No diabetes','Prediabetes or diabetes'],raw_rows=raw_count,features=features,source='https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset',protocol='Group by identical predictors; bounded stratified subsets within splits',transform='train mean/std')
    else:
        df=pd.read_csv(DATA/'house_prices.csv');raw_count=len(df);df=df.drop_duplicates()
        df=df[(df.Area>0)&(df.Price>0)].copy()
        features=['Area','Frontage','Access Road','Floors','Bedrooms','Bathrooms']
        raw=df[features].to_numpy(np.float32);raw[raw<0]=np.nan
        groups=df.Address.fillna('Unknown').astype(str).to_numpy()
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df,groups=groups))
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(df.iloc[tr],groups=groups[tr]));split=dict(train=tr[a],val=tr[b],test=te)
        median=np.nanmedian(raw[split['train']],axis=0);raw=np.where(np.isnan(raw),median,raw);y=df.Price.to_numpy(np.float32)
        meta=dict(kind='mlp',task='regression',raw_rows=raw_count,features=features,median=median.tolist(),source='https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024',protocol='Group split by exact Address, 64/16/20 approximate rows; no location feature',transform='log1p numeric, train mean/std',unit='billion VND')
        raw=np.log1p(raw)
    mean=raw[split['train']].mean(0);std=raw[split['train']].std(0);std[std<1e-6]=1
    meta.update(mean=mean.tolist(),std=std.tolist())
    if key=='housing':
        z=np.log1p(y);ym=z[split['train']].mean();ys=z[split['train']].std();meta.update(y_mean=float(ym),y_std=float(ys));extras={'raw_y':y};y=(z-ym)/ys
    else:extras={'raw_y':y}
    save(key,(raw-mean)/std,y,split,meta,extras)

def images(key):
    if key=='mnist':
        z=np.load(DATA/'mnist_raw.npz');x=np.concatenate([z['x_train'],z['x_test']]);y=np.concatenate([z['y_train'],z['y_test']]);raw_count=len(y)
        tr,va=train_test_split(np.arange(60000),train_size=10000,test_size=2000,random_state=42,stratify=y[:60000])
        te=train_test_split(np.arange(60000,70000),train_size=2000,random_state=42,stratify=y[60000:])[0]
        ids=np.r_[tr,va,te];imgs=np.stack([np.asarray(Image.fromarray(x[i]).resize((16,16),Image.Resampling.BILINEAR))[:,:,None] for i in ids]);y=y[ids];classes=list(map(str,range(10)))
        source='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz';sizes=[len(tr),len(va),len(te)]
    else:
        with zipfile.ZipFile(DATA/'EuroSAT_RGB.zip') as z:
            names=sorted(n for n in z.namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n);classes=sorted(set(n.split('/')[-2] for n in names));labels=np.array([classes.index(n.split('/')[-2]) for n in names]);raw_count=len(names)
            idx=train_test_split(np.arange(len(names)),train_size=10000,random_state=42,stratify=labels)[0];y=labels[idx];split=split_random(y);ids=np.concatenate(list(split.values()));sizes=[len(v) for v in split.values()]
            imgs=np.stack([np.asarray(Image.open(io.BytesIO(z.read(names[idx[i]]))).convert('RGB').resize((16,16),Image.Resampling.BILINEAR)) for i in ids]);y=y[ids]
        source='https://zenodo.org/records/7711810'
    a,b=np.cumsum(sizes)[:2];split=dict(train=np.arange(a),val=np.arange(a,b),test=np.arange(b,len(y)))
    x=imgs.astype(np.float32)/255;mean=x[split['train']].mean((0,1,2));std=x[split['train']].std((0,1,2));std=np.maximum(std,1e-6)
    save(key,(x-mean)/std,y,split,dict(kind='cnn',task='classification',classes=classes,raw_rows=raw_count,mean=mean.tolist(),std=std.tolist(),source=source,protocol='Fixed stratified benchmark subset; MNIST preserves official test',transform='Resize16 bilinear /255 then train channel mean/std'),{'raw_y':y,'images':imgs})

def sequences(key):
    from prepare_sequences import stock,customer
    {'stock':stock,'customer':customer}[key]()
    z=np.load(DATA/f'{key}_prepared.npz');old=json.loads((ROOT/'results'/key/'data_summary.json').read_text());x=z['x'];y=z['y'];sp={k:z[k] for k in ['train','val','test']}
    meta=dict(kind='rnn',task='regression' if key=='stock' else 'classification',classes=['No purchase','Purchase'],raw_rows=old.get('raw_rows',old.get('audit',{}).get('raw_rows')),source='https://github.com/tuananhtrieu1305/ISD_Assignment01/tree/main/A06/datasets',protocol=old['split'],transform=old.get('transform','Causal OHLCV indicators, train standardization'),features=old['features'],mean=old['x_mean'],std=old['x_std'],y_mean=old['y_mean'],y_std=old['y_std'],periods=old['periods'],details=old)
    extra={k:z[k] for k in z.files if k not in ['x','y','train','val','test']}
    save(key,x,y if key=='stock' else y.astype(int),sp,meta,extra)

if __name__=='__main__':
    for key in sys.argv[1:] or ['diabetes','housing','mnist','eurosat','stock','customer']:
        if key in ['diabetes','housing']:tabular(key)
        elif key in ['mnist','eurosat']:images(key)
        else:sequences(key)
