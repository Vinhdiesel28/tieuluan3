"""Lightweight, framework-independent preprocessing and checkpoint inference."""
import numpy as np

STOCK_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
STOCK_FEATURES = ['LogReturn', 'OpenGap', 'Range', 'Body', 'RelativeMA20', 'RelativeMA50', 'Volatility20', 'LogVolume']
CUSTOMER_FEATURES = ['Orders', 'Quantity', 'Revenue', 'DistinctItems', 'ActiveDays', 'AverageOrderValue']

def stock_features(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 5 or len(raw) < 80:
        raise ValueError('Cần ít nhất 80 dòng và 5 cột Open, High, Low, Close, Volume.')
    if not np.isfinite(raw).all() or np.any(raw[:, :4] <= 0) or np.any(raw[:, 4] < 0):
        raise ValueError('Giá phải dương, volume không âm và dữ liệu phải hữu hạn.')
    if np.any(raw[:,1] < raw[:,2]) or np.any(raw[:,1] < raw[:,[0,3]].max(1)) or np.any(raw[:,2] > raw[:,[0,3]].min(1)):
        raise ValueError('High/Low không phù hợp với Open/Close.')
    op, hi, lo, close, volume = raw.T
    returns = np.r_[np.nan, np.diff(np.log(close))]
    result = np.full((len(raw), 8), np.nan)
    for i in range(49, len(raw)):
        result[i] = [returns[i], op[i]/close[i-1]-1, (hi[i]-lo[i])/close[i],
                     (close[i]-op[i])/op[i], close[i]/close[i-19:i+1].mean()-1,
                     close[i]/close[i-49:i+1].mean()-1, returns[i-19:i+1].std(ddof=1), np.log1p(volume[i])]
    return result

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))

def recurrent_numpy(x, weights, kind, prefix='forward'):
    wx, wh = weights[prefix+'_wx'], weights[prefix+'_wh']
    bx, bh = weights[prefix+'_bx'], weights[prefix+'_bh']
    h = np.zeros((len(x), wh.shape[0]), dtype=np.float32)
    c = np.zeros_like(h)
    for t in range(x.shape[1]):
        a, b = x[:,t] @ wx + bx, h @ wh + bh
        if kind == 'rnn':
            h = np.tanh(a + b)
        elif kind == 'lstm':
            i, f, g, o = np.split(a + b, 4, axis=1)
            c = sigmoid(f)*c + sigmoid(i)*np.tanh(g)
            h = sigmoid(o)*np.tanh(c)
        elif kind == 'gru':
            # Canonical gate order z, r, n; reset-after formulation.
            az, ar, an = np.split(a, 3, axis=1)
            bz, br, bn = np.split(b, 3, axis=1)
            z, r = sigmoid(az+bz), sigmoid(ar+br)
            h = z*h + (1-z)*np.tanh(an+r*bn)
        else:
            raise ValueError(kind)
    return h

def predict_numpy(x, weights, kind):
    x = np.asarray(x, dtype=np.float32)
    if kind == 'bilstm':
        h = np.concatenate([recurrent_numpy(x, weights, 'lstm'), recurrent_numpy(x[:,::-1], weights, 'lstm', 'backward')], axis=1)
    else:
        h = recurrent_numpy(x, weights, kind)
    return (h @ weights['head_w'] + weights['head_b']).reshape(-1)
