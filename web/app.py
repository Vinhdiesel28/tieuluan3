from pathlib import Path
import sys,json,io,csv,base64,datetime
import numpy as np
from PIL import Image,UnidentifiedImageError
from flask import Flask,request,jsonify,render_template
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from scratch import forward,probabilities
from architecture_inference import predict_arch
from sequence_core import stock_features,STOCK_COLUMNS
from multi_agent import CoordinatorAgent
from knowledge_pipeline import validate_request
app=Flask(__name__);app.config['MAX_CONTENT_LENGTH']=3*1024*1024
AGENT=CoordinatorAgent()
CAT=json.loads((ROOT/'web/bundle/catalog.json').read_text(encoding='utf-8'))
ARCH=json.loads((ROOT/'web/bundle/architectures.json').read_text(encoding='utf-8')) if (ROOT/'web/bundle/architectures.json').exists() else {}
WEIGHTS={}
for key in CAT:
    for fw in ['scratch','keras','pytorch']:
        z=np.load(ROOT/'web/bundle'/f'{key}_{fw}.npz',allow_pickle=False);WEIGHTS[key,fw]={k:z[k] for k in z.files}
EXTRA={}
for key,models in ARCH.items():
    for name,item in models.items():
        with np.load(ROOT/'web/bundle'/item['file'],allow_pickle=False) as z:EXTRA[key,name]=dict(z)
def custom(payload,key,meta):
    context={}
    if meta['kind']=='cnn':
        value=payload.get('image')
        if not isinstance(value,str) or len(value)>2800000:raise ValueError('Ảnh phải dưới 2 MB.')
        blob=base64.b64decode(value,validate=True)
        with Image.open(io.BytesIO(blob)) as im:
            if im.width*im.height>16000000:raise ValueError('Ảnh tối đa 16 megapixel.')
            im=im.convert('L' if key=='mnist' else 'RGB').resize((16,16),Image.Resampling.BILINEAR);raw=np.asarray(im,np.float32)
        if raw.ndim==2:raw=raw[:,:,None]
        raw=raw/255
    else:
        text=payload.get('csv')
        if not isinstance(text,str) or len(text)>800000:raise ValueError('CSV cần có header và dưới 800 KB.')
        reader=csv.DictReader(io.StringIO(text.strip()));cols=STOCK_COLUMNS if key=='stock' else meta['features']
        if not reader.fieldnames or not set(cols).issubset(reader.fieldnames):raise ValueError('Thiếu cột: '+','.join(cols))
        rows=list(reader)
        if len(rows)>5000:raise ValueError('Tối đa 5000 dòng.')
        raw=np.array([[float(r[c]) for c in cols] for r in rows],dtype=np.float64)
        if not np.isfinite(raw).all():raise ValueError('Dữ liệu phải là số hữu hạn.')
        if key=='stock':
            dates=[datetime.date.fromisoformat(r['Date']) for r in rows]
            if dates!=sorted(set(dates)):raise ValueError('Date tăng dần, không trùng, YYYY-MM-DD.')
            context['last_close']=float(raw[-1,3]);raw=stock_features(raw)[-30:]
        elif key=='customer':
            if raw.shape!=(8,6) or (raw<0).any() or raw[:,0].sum()<1 or (raw[:,4]>7).any():raise ValueError('Cần 8 tuần hợp lệ, không âm, có ít nhất một đơn, ActiveDays ≤7.')
            raw=np.log1p(raw)
        else:
            if raw.shape!=(1,len(cols)):raise ValueError('Cần đúng một dòng dữ liệu.')
            raw=raw[0]
            if (raw<0).any():raise ValueError('Đặc trưng không được âm.')
            if key=='housing':raw=np.log1p(raw)
    x=(raw-np.array(meta['mean']))/np.array(meta['std'])
    if x.shape!=tuple(meta['shape']) or not np.isfinite(x).all():raise ValueError('Sai kích thước hoặc dữ liệu không hữu hạn.')
    return x.astype(np.float32),context
@app.get('/')
def home():return render_template('index.html')
@app.get('/health')
def health():return jsonify(status='ok',models=len(WEIGHTS)+len(EXTRA),agents=7,knowledge_documents=len(AGENT.knowledge.engine.documents),rag_systems=5)
@app.get('/api/catalog')
def catalog():
    result={}
    for k,v in CAT.items():
        criterion='RMSE' if v['meta']['task']=='regression' else 'MacroF1'
        models=ARCH.get(k,{})
        selected=min(models,key=lambda n:models[n]['validation'][criterion]) if criterion=='RMSE' and models else max(models,key=lambda n:models[n]['validation'][criterion]) if models else 'matched'
        result[k]=dict(selected=v['selected'],selected_model=selected,kind=v['meta']['kind'],features=v['meta'].get('features',[]),
            models=[dict(id=n,label=m['label'],framework=m['framework']) for n,m in models.items()]+[dict(id='matched',label={'mlp':'MLP','cnn':'CNN','rnn':'Simple RNN'}[v['meta']['kind']]+' — đối chiếu 3 thư viện',framework=None)],
            examples=[{a:b for a,b in e.items() if a!='x'} for e in v['examples']])
    return jsonify(result)
@app.post('/api/predict')
def predict():
    try:
        p=request.get_json(silent=True)
        if not isinstance(p,dict):raise ValueError('Gửi JSON object.')
        key=p.get('dataset');fw=p.get('framework');name=p.get('model','matched')
        if not isinstance(key,str) or key not in CAT or not isinstance(name,str):raise ValueError('Dataset/model không hợp lệ.')
        if name=='matched':
            if fw not in ['scratch','keras','pytorch']:raise ValueError('Framework không hợp lệ.')
        elif name not in ARCH.get(key,{}) or fw!=ARCH[key][name]['framework']:raise ValueError('Model/framework không hợp lệ.')
        meta=CAT[key]['meta'];context={}
        if 'image' in p or 'csv' in p:x,context=custom(p,key,meta)
        else:
            idx=p.get('example',0)
            if isinstance(idx,bool) or not isinstance(idx,int) or not 0<=idx<len(CAT[key]['examples']):raise ValueError('Mẫu không hợp lệ.')
            e=CAT[key]['examples'][idx];x=np.asarray(e['x'],np.float32);context={a:e[a] for a in ['actual','last_close','date'] if a in e}
        z=forward(WEIGHTS[key,fw],x[None],meta['kind']) if name=='matched' else predict_arch(EXTRA[key,name],ARCH[key][name]['config'],x[None])
        if not np.isfinite(z).all():raise ValueError('Đầu vào vượt miền số học của model.')
        if meta['task']=='classification':
            prob=probabilities(z)[0];label=int(prob.argmax());result=dict(prediction=meta['classes'][label],label=label,probabilities=prob.tolist(),classes=meta['classes'])
        else:
            v=float(z[0,0])*meta['y_std']+meta['y_mean']
            if abs(v)>30:raise ValueError('Đầu vào quá khác dữ liệu train.')
            result=dict(prediction=float(context['last_close']*np.exp(v) if key=='stock' else np.expm1(v)),unit='USD' if key=='stock' else 'tỷ VND')
        return jsonify(dataset=key,framework=fw,model=name,model_label=ARCH[key][name]['label'] if name!='matched' else meta['kind'].upper()+' (đối chiếu thư viện)',**result,**context)
    except (ValueError,TypeError,KeyError,IndexError,UnidentifiedImageError,OverflowError) as e:return jsonify(error=str(e)),400
@app.post('/api/agent')
def agent():
    try:
        payload=request.get_json(silent=True)
        if not isinstance(payload,dict):raise ValueError('Gửi JSON object.')
        question=payload.get('question','')
        if not isinstance(question,str) or not question.strip():raise ValueError('Câu hỏi không được để trống.')
        if len(question)>2000:raise ValueError('Câu hỏi tối đa 2000 ký tự.')
        return jsonify(AGENT.run(question.strip()))
    except (ValueError,TypeError,KeyError) as e:return jsonify(error=str(e)),400
@app.post('/api/rag')
def rag():
    try:
        payload=validate_request(request.get_json(silent=True))
        question=payload.pop('question')
        return jsonify(AGENT.knowledge.engine.answer(question,**payload))
    except (ValueError,TypeError,KeyError) as e:return jsonify(error=str(e)),400
@app.errorhandler(413)
def large(e):return jsonify(error='Yêu cầu quá lớn.'),413
if __name__=='__main__':app.run(host='127.0.0.1',port=5007,debug=False)
