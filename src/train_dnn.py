from collections import defaultdict
from math import ceil
import json 
import sys 
from types import SimpleNamespace

import numpy as np 
import torch 
from torch.optim import Adam 
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from models.dnn import IOCLinear
from config import config 

DATA = None 
HYPERPARAMS = SimpleNamespace(
    epochs=500, bs=2**8, patience=50,
    lr=0.001
)
torch.set_num_threads(16)

def remove_dupes(idxs,labels):
    idx_to_label = defaultdict(set)
    for i,idx in enumerate(idxs):
        idx_to_label[idx].add(labels[i])

    valid_idxs = []
    valid_labels = []
    for k,v in idx_to_label.items():
        if len(v) == 1: 
            valid_idxs.append(k)
            valid_labels.append(v.pop())

    return  torch.tensor(valid_idxs, dtype=torch.long), \
            torch.tensor(valid_labels, dtype=torch.long)

def train_one_fold(hp, tr_set, ioc_type):
    data = np.load(f'{DATA}/dense/{ioc_type}.npy')
    data = torch.from_numpy(data).detach()

    idxs = tr_set[ioc_type]['x']
    labels = tr_set[ioc_type]['y']
    idxs, labels = remove_dupes(idxs,labels)

    data = data[idxs]

    splitter = StratifiedKFold(n_splits=5)
    tr,va = next(splitter.split(data, labels))

    tr_x = data[tr]; tr_y = labels[tr]
    va_x = data[va]; va_y = labels[va]
    
    _, class_weights = tr_y.unique(return_counts=True)
    class_weights = class_weights.max() / class_weights

    model = IOCLinear(data.size(1), class_weights)
    model.preprocess_fit(tr_x)

    opt = Adam(model.parameters(), lr=hp.lr)

    best = (0,None,-1)
    n_batches = ceil(tr_x.size(0)/hp.bs)
    for e in range(hp.epochs): 
        model.train()
        
        # Shuffle
        order = torch.randperm(tr_x.size(0))
        tr_x = tr_x[order]
        tr_y = tr_y[order]

        for b in range(n_batches): 
            opt.zero_grad()
            x = tr_x[b*hp.bs : (b+1)*hp.bs]
            y = tr_y[b*hp.bs : (b+1)*hp.bs]
            loss = model.forward(x,y)
            loss.backward()
            opt.step()
            print(f"[{e}-{b}] {loss.item()}\r", end='')
        
        with torch.no_grad():
            model.eval()
            preds = model.inference(va_x)

        acc = accuracy_score(va_y, preds.argmax(dim=1))
        bacc = balanced_accuracy_score(va_y, preds.argmax(dim=1))

        print(f"[{e}] Loss: {loss.item():0.4f}  Acc: {acc:0.4f}  Bacc: {bacc:0.4f}")
        if bacc > best[0]: 
            best = (bacc, model.save(), e)
        if e-best[-1] > hp.patience: 
            print("Early stopping!")
            break 

    return best 

if __name__ == '__main__':
    DATA = config.get('ML_DATA')
    k = 'ips'

    with open(f'{DATA}/folds.json', 'r') as f:
        folds = json.load(f)

    out_dir = DATA.split('_')[0]
    for i in range(len(folds)):
        best = train_one_fold(HYPERPARAMS, folds[i]['tr'], k)
        torch.save(best, f'{config.get('RESULTS')}/trad_pretrained_{out_dir}/{k}/{k}_{i+1}.pt')