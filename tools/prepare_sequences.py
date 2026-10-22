"""Freeze chronological train/validation/test samples; fit only on training data."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sequence_core import stock_features, STOCK_COLUMNS, STOCK_FEATURES, CUSTOMER_FEATURES

def save(key, x, y, times, metadata, extra):
    out=ROOT/'results'/key;out.mkdir(parents=True,exist_ok=True)
    unique=np.unique(times)
    a,b=int(len(unique)*.7),int(len(unique)*.85)
    train=np.flatnonzero(times<unique[a]);val=np.flatnonzero((times>=unique[a])&(times<unique[b]));test=np.flatnonzero(times>=unique[b])
    mean=x[train].mean(axis=(0,1),dtype=np.float64).astype(np.float32)
    std=x[train].std(axis=(0,1),dtype=np.float64).astype(np.float32);std[std<1e-8]=1
    xs=((x-mean)/std).astype(np.float32)
    ym=float(y[train].mean()) if key=='stock' else 0.
    ys=float(y[train].std()) if key=='stock' else 1.
    ys=max(ys,1e-8)
    metadata.update({'key':key,'samples':len(y),'input_shape':list(x.shape[1:]),'train':len(train),'validation':len(val),'test':len(test),
                     'x_mean':mean.tolist(),'x_std':std.tolist(),'y_mean':ym,'y_std':ys,
                     'periods':{k:[str(times[ix[0]]),str(times[ix[-1]])] for k,ix in [('train',train),('validation',val),('test',test)]}})
    if key=='customer':metadata['positive_rates']={k:float(y[ix].mean()) for k,ix in [('train',train),('validation',val),('test',test)]}
    np.savez_compressed(ROOT/'data'/f'{key}_prepared.npz',x=xs,y=((y-ym)/ys).astype(np.float32),raw_y=y,times=times.astype(str),train=train,val=val,test=test,**extra)
    np.savez_compressed(out/'split_indices.npz',train=train,val=val,test=test)
    (out/'data_summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(key,metadata['samples'],metadata['train'],metadata['validation'],metadata['test'],flush=True)

def stock():
    df=pd.read_csv(ROOT/'data/AAPL_2015_2025.csv',parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
    assert not df.Date.duplicated().any() and df[STOCK_COLUMNS].notna().all().all()
    raw=df[STOCK_COLUMNS].to_numpy(dtype=np.float64);features=stock_features(raw)
    targets=np.arange(79,len(df))
    x=np.stack([features[t-30:t] for t in targets]).astype(np.float32)
    y=np.log(raw[targets,3]/raw[targets-1,3]).astype(np.float32)
    assert np.isfinite(x).all()
    metadata={'dataset':'AAPL historical stock','raw_rows':len(df),'start':str(df.Date.iloc[0].date()),'end':str(df.Date.iloc[-1].date()),
              'features':STOCK_FEATURES,'lookback':30,'target':'Next-session log return; convert to Close = last Close * exp(return)',
              'split':'Chronological 70/15/15 by target date; features only from earlier dates','price_column':'Close (not Adj Close)'}
    save('stock',x,y,df.Date.dt.strftime('%Y-%m-%d').to_numpy()[targets],metadata,{'last_close':raw[targets-1,3],'actual_close':raw[targets,3],'raw_index':targets})

def customer():
    cache=ROOT/'data/retail_weekly.npz';audit_path=ROOT/'data/retail_audit.json'
    if cache.exists():
        z=np.load(cache,allow_pickle=False);dense=z['dense'];weeks=z['weeks'];ids=z['ids'];audit=json.loads(audit_path.read_text())
    else:
        sheets=pd.read_excel(ROOT/'data/online_retail_II.xlsx',sheet_name=None,engine='openpyxl')
        df=pd.concat(sheets.values(),ignore_index=True);df.columns=[str(c).strip() for c in df.columns]
        df=df.rename(columns={'Customer ID':'CustomerID','Invoice':'InvoiceNo','Price':'UnitPrice'})
        audit={'raw_rows':len(df),'sheets':list(sheets),'missing_customer_rows':int(df.CustomerID.isna().sum()),'duplicate_rows':int(df.duplicated().sum())}
        df=df.drop_duplicates().dropna(subset=['CustomerID','InvoiceDate'])
        keep=(df.Quantity>0)&(df.UnitPrice>0)&~df.InvoiceNo.astype(str).str.startswith('C')
        audit['nonpurchase_rows_excluded_after_dedup_and_id_filter']=int((~keep).sum())
        df=df.loc[keep].copy();df['CustomerID']=df.CustomerID.astype(int)
        df['InvoiceDate']=pd.to_datetime(df.InvoiceDate)
        df['Week']=df.InvoiceDate.dt.to_period('W-SUN').dt.start_time
        df['Day']=df.InvoiceDate.dt.normalize();df['Revenue']=df.Quantity*df.UnitPrice
        weekly=df.groupby(['CustomerID','Week']).agg(Orders=('InvoiceNo','nunique'),Quantity=('Quantity','sum'),Revenue=('Revenue','sum'),DistinctItems=('StockCode','nunique'),ActiveDays=('Day','nunique')).reset_index()
        weekly['AverageOrderValue']=weekly.Revenue/weekly.Orders
        all_weeks=pd.date_range(weekly.Week.min(),weekly.Week.max(),freq='W-MON')
        # Last observed week is incomplete: do not use it as a future label.
        all_weeks=all_weeks[all_weeks+pd.Timedelta(days=6)<=df.InvoiceDate.max().normalize()]
        ids=np.sort(weekly.CustomerID.unique());weeks=all_weeks.strftime('%Y-%m-%d').to_numpy(dtype=str)
        dense=np.zeros((len(ids),len(weeks),6),dtype=np.float32)
        weekly=weekly[weekly.Week.isin(all_weeks)]
        ii=np.searchsorted(ids,weekly.CustomerID.to_numpy());jj=all_weeks.get_indexer(weekly.Week)
        dense[ii,jj]=weekly[CUSTOMER_FEATURES].to_numpy(np.float32)
        audit.update({'retained_purchase_rows':len(df),'customers':len(ids),'complete_weeks':len(weeks),'raw_end':str(df.InvoiceDate.max()),'raw_start':str(df.InvoiceDate.min())})
        np.savez_compressed(cache,dense=dense,weeks=weeks,ids=ids)
        audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    xs=[];ys=[];times=[];customers=[];anchor=[]
    for t in range(8,len(weeks),4):
        eligible=np.flatnonzero(dense[:,t-8:t,0].sum(1)>0)
        xs.append(np.log1p(dense[eligible,t-8:t]));ys.append((dense[eligible,t,0]>0).astype(np.float32))
        times.extend([str(weeks[t])]*len(eligible));customers.extend(ids[eligible].tolist());anchor.extend([t]*len(eligible))
    x=np.concatenate(xs).astype(np.float32);y=np.concatenate(ys)
    metadata={'dataset':'UCI Online Retail II','features':CUSTOMER_FEATURES,'lookback':8,'anchor_stride_weeks':4,
              'target':'At least one valid purchase in the next calendar week','eligibility':'At least one purchase in the previous 8 weeks',
              'split':'Chronological 70/15/15 by target week; returning customers may occur in different splits',
              'transform':'log1p nonnegative weekly aggregates, then train-only standardization','audit':audit}
    save('customer',x,y,np.array(times),metadata,{'customer_ids':np.array(customers),'week_index':np.array(anchor)})

if __name__=='__main__':
    for key in sys.argv[1:] or ['stock','customer']:{'stock':stock,'customer':customer}[key]()
