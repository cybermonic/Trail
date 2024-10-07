import json 

import pandas as pd 
import torch 
from torch_geometric.nn import MessagePassing
from tqdm import tqdm 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

torch.set_num_threads(16)

def nonzero_softmax(x):
    zeros = x == 0
    x = torch.softmax(x,dim=-1)
    x[zeros] = 0
    return x 

def prop_loop(g, x, te, hops=5):
    mp = MessagePassing(aggr='mean')

    event_feats = []
    for k in range(hops):
        x += mp.propagate(g.edge_index, x=x, size=None)
        print("[%d] Labeled events: %d/%d" % (
            k+1,
            (x[te].sum(dim=1) > 0).sum(),
            te.shape[0]
        ))

        event_feats.append(x[te].clone())
        
    return torch.stack(event_feats)

def kfold(hops, graph_file):
    g = torch.load(graph_file)
    #csr = torch.load('data/full_graph_csr.pt').edge_csr
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    #valid_mask = torch.tensor([len(e) for e in csr[g.event_ids]]) > 0
    valid_mask = g.sources == g.src_map['OTX']
    all_data = g.event_ids[valid_mask]
    valid_ys = g.y[valid_mask]

    other_data = g.event_ids[~valid_mask]
    other_ys = g.y[~valid_mask]

    prop_results = []
    ys = []
    stats = []
    i = 1
    for tr, te in kfold.split(all_data, valid_ys):
        print(
            "Fold %d  (tr %d, te %d)" % 
            (i, all_data.size(0)-len(te), len(te))
        )
        i += 1

        tr_idx = all_data[tr]
        y = valid_ys[te]
        ys.append(y)

        tr_x = torch.zeros(g.x.size(0), g.y.max()+1)
        tr_x[tr_idx, valid_ys[tr]] = 1. 
        tr_x[other_data, other_ys] = 1. 
        
        y_hats = prop_loop(g, tr_x, all_data[te], hops=hops)
        prop_results.append(y_hats) 

        stat = dict() 
        for l in range(hops):
            pred = torch.argmax(y_hats[l], dim=1)
            pred[y_hats[l].sum(dim=1) == 0] = -1

            stat[f"{l}-acc"] = accuracy_score(y, pred)
            stat[f"{l}-bac"] = balanced_accuracy_score(y, pred)
            #cm = confusion_matrix(y, pred)
            #individual_accs = cm.diagonal() / cm.sum(axis=1)
            
            #for idx,name in g.label_map.items():
            #    stat[f"{l}-{name}"] = individual_accs[idx]

        stats.append(stat)
        print(json.dumps(stat, indent=1))

    df = pd.DataFrame(stats)
    print(df.mean())
    with open('results/nonparametric_prop/out.csv', 'w') as f:
        f.write(df.to_csv())
        f.write(df.mean().to_csv())
        f.write(df.sem().to_csv())

    prop_results = torch.cat(prop_results, dim=1)
    ys = torch.cat(ys)
    torch.save((prop_results, ys), 'predictions/nonparametric_prop_new.pt')

if __name__ == '__main__':
    import sys 
    print('analyzing')
    kfold(5, f'otx_dataset-d2e/full_graph_ei.pt')