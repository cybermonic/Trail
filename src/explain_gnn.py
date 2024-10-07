import os

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm import tqdm

from models.gnn import ExplainableSAGE
from train_gnn import get_and_split

def get_node_names(feat_dir, ntypes, feat_idx, true_y, label_map, node_names):
    type_dict = {'ips': 0, 'urls': 1, 'domains': 2}

    # Label IOCs
    names = [''] * feat_idx.size(0)
    for type_str, type_idx in type_dict.items():
        print(f'Loading {type_str}')
        df = pd.read_csv(
            os.path.join(feat_dir, f'{type_str}.csv'),
            sep='\t'
        )

        for i in tqdm(range(feat_idx.size(0))):
            if ntypes[i] != type_idx:
                continue
            if (idx := feat_idx[i]) == -1:
                names[i] = f'{type_str.upper()}:{node_names[i]}'
                continue

            name = df['ioc'][idx.item()]
            names[i] = name if name != '' else f'{type_str.upper()}:{node_names[i]}'

    # Label events
    events = true_y.nonzero()
    node,apt = events.T
    for i in range(node.size(0)):
        names[node[i].item()] = label_map[apt[i].item()]

    # Label ASNs
    asns = (feat_idx == 5).nonzero()
    for asn in asns:
        names[asn.item()] = f'ASN {node_names[asn.item()]}'

    return names

@torch.no_grad()
def make_dataset(model_file, apt='APT28'):
    sd,args,kwargs = torch.load(model_file)
    dataset_dir = 'otx_dataset'

    # Load the first fold
    generator = get_and_split(dataset_dir, val=False)
    g, (tr_idx, tr_ys), (te_idx, te_ys) = next(generator)
    inv_labels = {v:k for k,v in g.label_map.items()}
    apt_idx = inv_labels[apt]
    of_interest = te_idx[te_ys == apt_idx]

    # Add labels from training set to data to allow lprop
    masked_labels = torch.zeros(g.x.size(0), len(inv_labels))
    masked_labels[tr_idx,tr_ys] = 1

    true_labels = masked_labels.clone()
    true_labels[te_idx, te_ys] = 1

    # Load saved model
    model = ExplainableSAGE(*args, **kwargs)
    model.load_state_dict(sd)
    model.eval()

    # torch_geometric.explainable requires models to take full
    # x and edge_index as arguments, so extract them outside of
    # the model rather than having model.forward(batch) do this for us
    ei,reindex = g.edge_csr.k_hop_subgraph(of_interest, 3)
    feats = model.net.feature_sampler(g, reindex)
    y_feats = masked_labels[reindex]
    true_y = true_labels[reindex]
    x = torch.cat([feats, y_feats], dim=1)

    batch = (reindex == te_idx.unsqueeze(-1)).sum(dim=0)
    batch = batch.nonzero().squeeze()

    node_names = [g.node_names.get(i.item(), -1) for i in reindex]
    node_str = get_node_names(dataset_dir, g.x[reindex], g.feat_map[reindex], true_y, g.label_map, node_names)
    data = Data(
        x=x,
        edge_index=ei,
        target=batch,
        y=torch.full(batch.size(), apt_idx),
        node_str=node_str
    )
    torch.save(data, 'xgnn_graph.pt')
    return data

def explain(g, model_file):
    sd, args, kwargs = torch.load(model_file)
    model = ExplainableSAGE(*args, **kwargs)
    model.load_state_dict(sd)
    model.eval()

    for i in range(10):
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',  # Model returns log probabilities.
            ),
        )

        # Generate explanation for the node at index `10`:
        explanation = explainer(g.x, g.edge_index, index=g.target[i])
        print(explanation.edge_mask)
        print(explanation.node_mask)

        torch.save(explanation, f'explaination-{i}.pt')

if __name__ == '__main__':
    MODEL_FILE = 'weights/3-layer/gnn_train-0.808_max_lprop+feats+ae-new-data.pt'

    #g = make_dataset(MODEL_FILE) # Only need to run once
    g = torch.load('xgnn_graph.pt')
    explain(g, MODEL_FILE)