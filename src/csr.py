from collections import defaultdict
from collections.abc import Iterable

import torch 
from tqdm import tqdm 

class CSR:
    def __init__(self, ei=None):
        self.idx=None 
        self.ptr=None 
        self.ignore=None

        if ei is not None:
            self.ei_to_csr(ei)

    def ei_to_csr(self, ei):
        neighbors = defaultdict(set)

        # Sort into sets of src -> dst 
        for i in tqdm(range(ei.size(1)), desc='Organizing'):
            src,dst = ei[:,i]
            neighbors[src.item()].add(dst.item())

        # Make into efficient data structure
        self.idx = torch.zeros(ei.size(1), dtype=torch.long)
        self.ptr = [0]

        for src in tqdm(range(ei.max()+1), desc='Torchifying'):
            dsts = list(neighbors[src])
            
            n_v = len(dsts)
            start = self.ptr[-1]
            end = start + n_v

            self.idx[start:end] = torch.as_tensor(dsts)
            self.ptr.append(end)

        self.ptr = torch.tensor(self.ptr)

    def __drop_edge_sample_one(self, idx, p):
        neighbors = self.get(idx)
        rnd = torch.rand(neighbors.size(0))
        return neighbors[rnd > p]
    
    def drop_edge_sample(self, idx, p=0.25):
        if isinstance(idx, Iterable):
            return [self.__drop_edge_sample_one(i,p) for i in idx]
        return self.drop_edge_sample(idx, p)

    def __sample_n_one(self, idx, n):
        neighbors = self.get(idx)
        if neighbors.size(0) > n:
            idx = torch.randperm(neighbors.size(0))
            return neighbors[idx[:n]]
        return neighbors

    def sample_n(self, idx, n=32):
        if isinstance(idx, Iterable):
            return [self.__sample_n_one(i,n) for i in idx]
        return self.__sample_n_one(idx, n)
    
    def sample_exactly_n(self, idx, n=32):
        if isinstance(idx, Iterable):
            return torch.stack([self.sample_exactly_n(i.item(),n) for i in idx])
        
        neighbors = self.get(idx)

        if neighbors.size(-1) == 0:
            print("hmm")

        return neighbors[
            torch.randint(0, neighbors.size(0), (n,))
        ]

    def get(self, idx):
        st = self.ptr[idx]
        en = self.ptr[idx+1]
        ret = self.idx[st:en]

        if self.ignore is not None:
            ret = ret[~self.ignore[ret]]
        
        return ret 

    def __getitem__(self, idx):
        if isinstance(idx, Iterable):
            if isinstance(idx, torch.Tensor) and idx.dim()==0:
                return self.get(idx)
            return [self.get(i) for i in idx]
        return self.get(idx)
    
    def set_ignore(self, ignore_list):
        self.ignore = ignore_list

    def k_hop_subgraph(self, batch, k, n=None):
        '''
        Assumes batch is unique. Will return a k-hop
        subgraph reindexed s.t. batch[n] correlates to nid n
        '''
        edges = set()

        nodes = [b.item() for b in batch]
        num_nodes = [len(nodes)]
        reidx = {n:i for i,n in enumerate(nodes)}
        def get_idx(n):
            if (nid := reidx.get(n)) is None:
                nid = num_nodes[0]
                reidx[n] = nid
                num_nodes[0] += 1
                nodes.append(n)
            return nid 

        # Src/dst is arbitrary as we always use 
        # undirected graph
        frontier = set(nodes)
        next_frontier = set()
        for _ in range(k):
            for dst in frontier:
                if n is None:
                    srcs = self.__getitem__(dst)
                else: 
                    srcs = self.sample_n(dst, n)
                    
                dst_id = get_idx(dst)
                
                for s in srcs:
                    s = s.item()
                    if s not in reidx:
                        next_frontier.add(s) 
                    edges.add((get_idx(s), dst_id))
            
            frontier = next_frontier
            next_frontier = set()

        ei = torch.tensor(list(edges)).T 
        idx = torch.tensor(nodes)

        return ei,idx

    def to_ei(self):
        dst = self.idx 
        src = []
        for i in tqdm(range(self.ptr.size(0)-1)):
            cnt = self.ptr[i+1]-self.ptr[i] 
            src += [i] * cnt.item() 

        return torch.stack([torch.tensor(src), dst])