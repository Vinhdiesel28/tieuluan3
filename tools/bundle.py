from pathlib import Path
import json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'web/bundle';out.mkdir(parents=True,exist_ok=True);catalog={}
    for key in ['diabetes','housing','mnist','eurosat','stock','customer']:
        meta=json.loads((ROOT/'results'/key/'summary.json').read_text(encoding='utf-8'));d=np.load(ROOT/'data'/f'{key}.npz');ix=d['test'][np.linspace(0,len(d['test'])-1,6,dtype=int)];examples=[]
        for j,i in enumerate(ix):
            e=dict(id=j,x=d['x'][i].tolist(),actual=float(d['actual_close'][i] if key=='stock' else d['raw_y'][i]))
            if key=='stock':e['last_close']=float(d['last_close'][i]);e['date']=str(d['times'][i])
            if meta['kind']=='cnn':e['pixels']=d['images'][i].tolist()
            elif key!='stock':
                raw=d['x'][i]*np.array(meta['std'])+np.array(meta['mean'])
                if key in ['housing','customer']:raw=np.maximum(np.expm1(raw),0)
                e['csv']=','.join(meta['features'])+'\n'+'\n'.join(','.join(f'{v:.6f}' for v in row) for row in np.atleast_2d(raw))
            examples.append(e)
        for fw in ['scratch','keras','pytorch']:shutil.copyfile(ROOT/'results'/key/f'{fw}_42.npz',out/f'{key}_{fw}.npz')
        catalog[key]=dict(meta=meta,examples=examples,selected=json.loads((ROOT/'results'/key/'selection.json').read_text())['framework'])
    (out/'catalog.json').write_text(json.dumps(catalog,ensure_ascii=False),encoding='utf-8');print('Bundled 18 trained models')
if __name__=='__main__':main()
