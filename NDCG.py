import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed  
import multiprocessing
def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k=5, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def mean_NDCG(preds, truth, n_modes=5):
    assert(len(preds)==len(truth))
    r = pd.DataFrame(0, index=preds.index, columns=preds.columns, dtype=np.float64)
    for col in preds.columns:
        r[col] = (preds[col] == truth) * 1.0

    score = pd.Series(r.apply(ndcg_at_k, axis=1, reduce=True), name='score')
    return np.mean(score)

def one_partition_NDCG(x ,labels ,model ,i ,factor):
    le = LabelEncoder()
    y = le.fit_transform(labels)   
    piv_train = x.shape[0]
    trans_x = []
    trans_y = []
    test_x = []
    test_y = []
    if i == 0:
        trans_x = x[(i+1)*factor:] 
        trans_y = y[(i+1)*factor:] 
        test_x = x[:(i+1)*factor]
        test_y = y[:(i+1)*factor]
    elif i+1 == piv_train/factor:
        trans_x = x[:i*factor] 
        trans_y = y[:i*factor] 
        test_x = x[i*factor:]
        test_y = y[i*factor:]
    else:
        trans_x = pd.concat((x[:i*factor],x[(i+1)*factor:]), axis=0 ,ignore_index=True)
        trans_y = pd.concat((y[:i*factor],y[(i+1)*factor:]), axis=0 ,ignore_index=True)
        test_x = x[i*factor:(i+1)*factor]
        test_y = y[i*factor:(i+1)*factor]
    model.fit(trans_x,trans_y)
    y_pred = model.predict_proba(test_x)
    ids = []  
    cts = []  
    for j in range(factor):
        cts += [le.inverse_transform(np.argsort(y_pred[j])[::-1])[:5].tolist()]
    preds = pd.DataFrame(cts)
    truth = pd.Series(labels[i*factor:(i+1)*factor])
    #truth = pd.Series(le.inverse_transform(test_y).tolist())
    return mean_NDCG(preds, truth)

def cross_valation_score(x ,labels ,model ,partition):
    piv_train = x.shape[0]
    factor = piv_train / partition; 
    sum = Parallel(n_jobs=partition)(delayed(one_partition_NDCG)(x,labels,model,i,factor) for i in range(partition))
    return np.mean(sum)
    
# simple check because the excution time is too long.
def score_predictions(x ,labels ,model):
    piv_train = x.shape[0]
    factor = piv_train / 10; 
    return one_partition_NDCG(x,labels,model,9,factor)
    
