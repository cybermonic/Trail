import numpy as np
import torch
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
from xgboost import XGBClassifier

from lprop_feats_gnn import get_and_split
from models.ioc_encoder import FeatureGetter


def get_dataset_from_neighbors(idx,feats, events,labels):
    event_iocs = []
    for event in events:
        ioc_idx = idx[event]
        ioc_idx = ioc_idx[ioc_idx != -1]
        event_iocs.append(feats[ioc_idx])

    cnts = torch.tensor([ei.size(0) for ei in event_iocs])
    y = labels.repeat_interleave(cnts)
    x = torch.cat(event_iocs)

    return x,y

def predict_neighbors(models, ips,doms,urls, iocs):
    keys = ['ips', 'domains', 'urls']
    ys = []
    for ioc in iocs:
        votes = []
        for i, (idx,x) in enumerate([ ips,doms,urls ]):
            valid = idx[ioc]
            valid = valid[valid != -1]
            feats = x[valid]

            if feats.size(0) == 0:
                continue

            preds = models[keys[i]].predict(feats)
            votes.append(preds)

        votes = np.concatenate(votes)
        y = mode(votes,keepdims=False)[0]
        ys.append(y)

    return torch.tensor(ys)


def train_fold(models, g, feats, tr_idx, tr_y):
    # Use for debugging
    #subset = torch.randperm(tr_idx.size(0))[:100]
    #tr_idx = tr_idx[subset]
    #tr_y = tr_y[subset]

    iocs = g.edge_csr[tr_idx]
    lens = [ioc.size(0) for ioc in iocs]

    uq,inv = torch.cat(iocs).unique(return_inverse=True)
    ips,domains,urls = feats(g, uq)
    iocs = inv.split(lens)

    ip_x,ip_y = get_dataset_from_neighbors(*ips, iocs,tr_y)
    dom_x,dom_y = get_dataset_from_neighbors(*domains, iocs,tr_y)
    url_x,url_y = get_dataset_from_neighbors(*urls, iocs,tr_y)

    print(f"\nTraining IP CLS ({ip_x.size(0)} samples)")
    models['ips'].fit(ip_x, ip_y)

    print(f"Training Domain CLS ({dom_x.size(0)} samples)")
    models['domains'].fit(dom_x, dom_y)

    print(f"Training URL CLS ({url_x.size(0)} samples)")
    models['urls'].fit(url_x, url_y)

    return models

def eval_fold(models, g, feats, te_idx, te_y):
    iocs = g.edge_csr[te_idx]
    lens = [ioc.size(0) for ioc in iocs]

    uq,inv = torch.cat(iocs).unique(return_inverse=True)
    ips,domains,urls = feats(g, uq)
    iocs = inv.split(lens)

    y_hat = predict_neighbors(models, ips,domains,urls, iocs)

    acc = accuracy_score(te_y, y_hat)
    bac = balanced_accuracy_score(te_y, y_hat)

    print(f"Acc: {acc:0.4f}\tB-Acc: {bac:0.4f}")
    return acc,bac

if __name__ == '__main__':
    DATASET = 'otx_dataset-d2e'
    feat = FeatureGetter(10, 'otx_dataset-d2e')

    '''
    models = {
        'ips': RandomForestClassifier(n_jobs=64),
        'domains': RandomForestClassifier(n_jobs=64),
        'urls': RandomForestClassifier(n_jobs=64)
    }
    '''
    models = {
        'ips': XGBClassifier(n_jobs=64),
        'domains': XGBClassifier(n_jobs=64),
        'urls': XGBClassifier(n_jobs=64)
    }


    stats = dict(acc=[],bac=[])
    folds = get_and_split(DATASET, val=False)
    for g, tr,te in folds:
        train_fold(models, g,feat, *tr)
        acc,bac = eval_fold(models, g,feat, *te)

        stats['acc'].append(acc)
        stats['bac'].append(bac)

    df = pd.DataFrame(stats)
    print(df)
    print(df.mean())
    print(df.sem())

    with open('rf_events.txt', 'w+') as f:
        f.write(str(df) + '\n')
        f.write(str(df.mean()) + '\n')
        f.write(str(df.sem()) + '\n')
