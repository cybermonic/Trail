import os

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, MessagePassing
from torch_scatter import segment_csr

from .ioc_encoder import FeatureAutoencoder, FeatureSampler, VariationalFeatureAutoencoder


class BatchGNN(nn.Module):
    '''
    Uses both node features and label propagation
    '''

    # Prob wont change but for readability it a var
    # (oops, good thing we made it a variable)
    NUM_CLASS = 22
    def __init__(self, data_dir, in_dim, hidden, out_dim, layers, aggr='mean', autoencoder=False, variational=False, sm=True, sample_size=512):
        super().__init__()
        self.HOME = os.path.join(
            os.path.dirname(__file__),
            '..', data_dir
        )

        if autoencoder:
            if autoencoder == 'ignore':
                self.feature_sampler = None
            elif variational:
                self.feature_sampler = VariationalFeatureAutoencoder(in_dim, HOME)
            else:
                self.feature_sampler = FeatureAutoencoder(in_dim, self.HOME)
        else:
            self.feature_sampler = FeatureSampler(in_dim, self.HOME)
        self.autoencoder = autoencoder

        def get_net(i,o,nonlin=nn.ReLU, drop=0.1, kwargs=dict()):
            return nn.Sequential(
                nn.Linear(i,o),
                nn.Dropout(drop),
                nonlin(**kwargs)
            )

        self.net = nn.ParameterList(
            [get_net(2*(in_dim+self.NUM_CLASS), hidden)] +
            [get_net(2*hidden, hidden) for _ in range(layers-2)] +
            [get_net(2*hidden, out_dim, nonlin=nn.Identity)] # type: ignore
        )
        self.final_sm = sm
        self.sm = nn.Softmax(dim=-1)
        self.aggr = aggr
        self.sample_size = sample_size

    def forward(self, g, batch):
        h = self.message_pass(g, batch)
        return h

    def get_features(self, g, batch):
        features = self.feature_sampler(g, batch)

        oh = torch.zeros(batch.size(0), self.NUM_CLASS)

        events = (g.x[batch] == g.type_dict['EVENT'])
        event_idx = batch[events] # NIDs in batch that are events

        y_loc = (event_idx.unsqueeze(-1) == g.event_ids).nonzero()
        has_label = y_loc[:,0]
        label_loc = y_loc[:,1]

        event_labels = g.y[label_loc]
        oh[events.nonzero().squeeze()[has_label], event_labels] = 1

        return torch.cat([features, oh], dim=1)

    def message_pass(self, g, batch, layer=-1):
        if layer == -1:
            layer = len(self.net)

        if layer == 0:
            return self.get_features(g, batch)

        # Get f_{layer-1}(neighbors V batch)
        neighbors = g.edge_csr.sample_n(batch, n=self.sample_size) + list(batch.split(1))
        sizes = [n.size(0) for n in neighbors]

        idxptr = [0]
        for s in sizes:
            idxptr.append(s+idxptr[-1])

        unique_neighbors, idx = torch.cat(neighbors).unique(return_inverse=True)
        neighbor_z = self.message_pass(g, unique_neighbors, layer-1)

        # Aggregate node features (and single batch encoding)
        # output should be 2B x d
        h_prev = segment_csr(neighbor_z[idx], torch.as_tensor(idxptr), reduce=self.aggr)

        # Convert to B x 2d. Cats f(batch) to f(aggr(neighbors))
        h0_cat_hn = torch.cat(h_prev.split(batch.size(-1)), dim=-1)
        h = self.net[layer-1](h0_cat_hn)

        # L2 normalize
        l2 = torch.norm(h, 2, dim=-1, keepdim=True)+1e-8
        h = h / l2
        return h


class SelfAttnGNN(BatchGNN):
    def __init__(self, data_dir, in_dim, hidden, out_dim, layers, aggr='mean', autoencoder=False, variational=False, sm=True, sample_size=512, heads=8):
        super().__init__(data_dir, in_dim, hidden, out_dim, layers, aggr, autoencoder, variational, sm, sample_size)

        del self.net

        def get_net(i,o,nonlin=nn.ReLU, drop=0.1, kwargs=dict()):
            return nn.Sequential(
                nn.Linear(i,o),
                nn.Dropout(drop),
                nonlin(**kwargs)
            )

        self.proj_in = get_net(in_dim+self.NUM_CLASS, hidden)
        self.proj_out = get_net(hidden, out_dim, nonlin=nn.Softmax, kwargs=dict(dim=-1))
        self.attn = nn.ParameterList(
            [
                nn.MultiheadAttention(hidden, num_heads=heads, batch_first=True)
                for _ in range(layers)
            ]
        )

        self.norms = nn.ParameterList(
            [
                nn.LayerNorm(hidden)
                for _ in range(layers*2)
            ]
        )

        self.ffnns = nn.ParameterList(
            [
                nn.Linear(hidden, hidden)
                for _ in range(layers)
            ]
        )

    def forward(self, g, batch):
        h = self.message_pass(g, batch)
        return self.proj_out(h)

    def message_pass(self, g, batch, layer=-1):
        if layer == -1:
            layer = len(self.attn)

        if layer == 0:
            return self.proj_in(self.get_features(g, batch))

        # Get f_{layer-1}(neighbors V batch)
        neighbors = g.edge_csr.sample_exactly_n(batch, n=self.sample_size)
        neighbors = torch.cat([batch.unsqueeze(-1), neighbors], dim=1)
        unique_neighbors, idx = neighbors.unique(return_inverse=True)

        # B x n+1 x d
        qkv = self.message_pass(g, unique_neighbors, layer-1)[idx]
        q,kv = qkv[:,0:1,:], qkv[:,1:,:] # First column was self

        h,_ = self.attn[layer-1](q, kv, kv)
        h = h.squeeze(1)

        # Short circuit and norm
        h = h + q.squeeze(1)
        h = self.norms[(layer-1)*2](h)

        # Project and norm again
        h_ = self.ffnns[layer-1](h)
        h = h + h_
        h = self.norms[(layer-1)*2 + 1](h)

        return h

class SageClassifier(nn.Module):
    def __init__(
                self, data_dir, encoding_dim, hidden, out_dim, class_weights=None, layers=3, aggr='mean',
                autoencoder=True, sample_size=32, variational=False, heads=0
            ):

        super().__init__()
        self.args = (data_dir, encoding_dim, hidden, out_dim)
        self.kwargs = dict(
            class_weights=class_weights, layers=layers, aggr=aggr,
            autoencoder=autoencoder, sample_size=sample_size,
            variational=variational, heads=heads
        )

        self.ae = autoencoder

        if heads:
            self.net = SelfAttnGNN(data_dir, encoding_dim, hidden, out_dim, layers=layers, autoencoder=autoencoder, sample_size=sample_size, variational=variational, heads=heads)

        else:
            self.net = BatchGNN(data_dir, encoding_dim, hidden, out_dim, layers=layers, aggr=aggr, autoencoder=autoencoder, sample_size=sample_size, variational=variational)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, g, batch, labels):
        preds = self.net(g, batch)
        loss = self.criterion(preds, labels)

        if self.ae:
            recon_loss = self.net.feature_sampler.get_loss()
            loss = loss + recon_loss

        return loss

    def inference(self, g, batch):
        return self.net(g, batch)


class ExplainableGNN(BatchGNN):
    '''
    Need a model that can take arguments (x, ei, **kwargs) so it will
    work with the PyG explainability models. Hopefully this is an easy conversion
    from csr used in original BatchGNN to dense, edge index here.
    Luckilly, we assume x includes label info and stuff already so just need to update
    label prop function
    '''
    def __init__( self, data_dir, encoding_dim, hidden, out_dim, layers=3, aggr='mean',
                autoencoder=True, sample_size=32, variational=False):
        super().__init__(
            data_dir, encoding_dim, hidden, out_dim, layers=layers, aggr=aggr, autoencoder=autoencoder,
            sample_size=sample_size, variational=variational, sm=False
        )
        self.layers = layers
        self.mp = MessagePassing(aggr=aggr)

    def forward(self, x, edge_index):
        h = self.message_pass(x, edge_index)

        if self.final_sm:
            return self.sm(h)
        return h

    def message_pass(self, x, edge_index):
        for layer in range(self.layers):
            # Aggregate node features (and single batch encoding)
            # output should be 2B x d
            aggr = self.mp.propagate(edge_index, size=None, x=x)

            # Convert to B x 2d. Cats f(batch) to f(aggr(neighbors))
            h0_cat_hn = torch.cat([aggr, x], dim=-1)
            h = self.net[layer](h0_cat_hn)

            # L2 normalize
            l2 = torch.norm(h, 2, dim=-1, keepdim=True)+1e-8
            x = h / l2

        return x

class ExplainableSAGE(SageClassifier):
    def __init__(
                self, data_dir, encoding_dim, hidden, out_dim, class_weights=None, layers=3, aggr='mean',
                autoencoder=True, sample_size=32, variational=False, heads=0
            ):

        super().__init__(
            data_dir, encoding_dim, hidden, out_dim, class_weights, layers, aggr,
            'ignore', sample_size, variational, heads
        )
        self.args = (data_dir, encoding_dim, hidden, out_dim)
        self.kwargs = dict(
            class_weights=class_weights, layers=layers, aggr=aggr,
            autoencoder=autoencoder, sample_size=sample_size,
            variational=variational, heads=heads
        )

        self.net = ExplainableGNN(data_dir, encoding_dim, hidden, out_dim, layers=layers, aggr=aggr, autoencoder=autoencoder, sample_size=sample_size, variational=variational)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.sm = nn.Softmax(dim=1)
        self.scale = 1

    def inference(self, x, ei):
        return self.sm(self.net(x, ei)*self.scale)

    def forward(self, x,ei):
        return self.net(x,ei)