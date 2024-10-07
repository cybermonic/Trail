from math import ceil

from dateutil.parser import parser
from joblib import Parallel, delayed
from neo4j import GraphDatabase
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from csr import CSR
from .domain import RECORD_TYPES, nlp_features as domain_nlp_features, to_dense_gnn as domain_dense_gnn, to_dense as domain_to_dense
from .url import extract as extract_url, to_dense_gnn as url_dense_gnn, to_dense as url_to_dense
from .ip import to_dense_gnn as ip_dense_gnn, to_dense as ip_to_dense
from .utils import (
    get_ip_onehot, get_country_code_mapper, get_tld_feats
)

class GraphFeatureGetter():
    BATCH_SIZE = 2048 *8

    def __init__(self, config, testing=False):
        # Much faster to query using neo4j module than py2neo
        self.graph = GraphDatabase.driver(f"bolt://{config['NEO4j_URL']}:7687")

        self.iss_mapper = get_ip_onehot()
        self.cc_mapper = get_country_code_mapper()
        self.n_ccs = max(self.cc_mapper.values())+1
        self.date_parser = parser()

        self.testing = testing

    def __add_if_present(self, key, in_d, out_d):
        if key in in_d:
            out_d[key] = in_d[key]

    def get_ip_feats(self, nids=[], include_id=False):
        '''
        Given list of nids get IP features for those nodes
        If list is None, return all labled nodes
        '''
        if nids:
            query = f'''
                MATCH
                    (ip:IP)
                WHERE
                    id(ip) in $nids
                OPTIONAL MATCH
                    (ip) -- (e:EVENT),
                    (ip) -[:IN_ASN_GROUP]- (a:ASN) // And if they exist grab their ASN values
                WHERE
                    e.label <> "UNK"
                RETURN                              // Filter by nid later
                    distinct id(ip) as nid,
                    properties(ip) as feats, // Doesn't filter repeats
                    a.issuer as issuer,
                    e.label as apt
                '''


        else:
            # Note: issuer is stored on any neighbor ASN node (should be
            # exactly 1 or 0 per IP)
            query = f'''
                MATCH (n:IP) -- (e:EVENT)            // Grab all labeled IPs
                OPTIONAL MATCH (n) -[r:IN_ASN_GROUP]- (a:ASN)   // And if they exist grab their ASN values
                WHERE
                    e.label <> "UNK"
                RETURN
                    distinct id(n) as nid,
                    properties(n) as feats,
                    a.issuer as issuer,
                    e.label as apt
            '''
        if self.testing:
            query += 'LIMIT 100'

        rows = []

        # Takes about 10 seconds for all IPs
        with self.graph.session() as session:
            resp = session.run(query, nids=nids)

            for r in tqdm(resp, desc='Parsing IP query response', total=len(nids)):
                # This is actually a lot faster than giving it a list of nids to filter by
                iss = r['issuer']
                apt = r['apt']
                ip = r['feats']

                # Country code
                sparse_row = []
                if (cc := ip.get('country_code')):
                    if (cc_oh := self.cc_mapper.get(cc)):
                        sparse_row.append(cc_oh)

                # Issuer
                if (iss_oh := self.iss_mapper.get(iss)):
                    iss_oh += self.n_ccs
                    sparse_row.append(iss_oh)

                # Now that we have one-hot feats adding the rest is simple
                row = {'one_hot': sparse_row}
                [
                    self.__add_if_present(k, ip, row)
                    for k in ['latitude', 'longitude']
                ]
                row['apt'] = apt
                row['ioc'] = ip['value']

                if include_id:
                    row['nid'] = r['nid']

                rows.append(row)

        return pd.DataFrame(rows)


    def get_domain_feats(self, nids=[], include_id=False):
        '''
        Given list of nids get domain features for those nodes
        If list is None, return all labled nodes
        '''
        if nids:
            feat_query = f'''
                MATCH
                    (n:domain)  // Grab all domains
                WHERE
                    id(n) in $nids
                OPTIONAL MATCH
                    (n) -- (e:EVENT)
                WHERE
                    e.label <> "UNK"
                RETURN
                    distinct id(n) as nid,
                    n.value as ioc,
                    n.first_seen as first_seen,
                    n.last_seen as last_seen,
                    n.has_nxdomain as has_nxdomain,
                    e.label as apt
            '''
        else:
            feat_query = f'''
                MATCH
                    (n:domain) -- (e:EVENT)  // Grab all domains w labels
                WHERE
                    e.label <> "UNK"
                RETURN
                    distinct id(n) as nid,
                    n.value as ioc,
                    n.first_seen as first_seen,
                    n.last_seen as last_seen,
                    n.has_nxdomain as has_nxdomain,
                    e.label as apt
            '''

        if self.testing:
            feat_query += ' LIMIT 100'

        rows = []
        nid_to_idx = dict()

        # First grab individual node features
        with self.graph.session() as session:
            resp = session.run(feat_query, nids=nids)

            i = 0
            for r in tqdm(resp, desc='Processing domain query response', total=len(nids)):
                # Create dict of node features
                row = dict(r)
                nid = row.pop('nid')
                row.update(domain_nlp_features(r['ioc']))

                # Convert time stamp str to number
                for k in ['first_seen', 'last_seen']:
                    if row.get(k):
                        row[k] = self.date_parser.parse(row[k]).timestamp()

                if (tld := get_tld_feats([r['ioc']], sparse=True)[0]):
                    row[tld] = True

                # Map nid to index in list of rows
                nid_to_idx[nid] = i

                if include_id:
                    row['nid'] = nid
                rows.append(row)

                i += 1

        # Next, count the edge types for every node we just got
        neigh_query = f'''
        MATCH (n:domain) -[r]- ()
        WHERE
            id(n) in $nids and
            type(r) in $recs
        RETURN
            id(n) as nid,
            type(r) as rtype,
            count(r) as cnt
        '''

        n_batches = ceil(len(nid_to_idx) / self.BATCH_SIZE)
        nids = list(nid_to_idx.keys())
        prog = tqdm(desc='Counting domain records', total=len(nid_to_idx))
        for i in range(n_batches):
            batch = nids[self.BATCH_SIZE*i : self.BATCH_SIZE*(i+1)]
            with self.graph.session() as session:
                resp = session.run(
                    neigh_query,
                    nids=batch,
                    recs=RECORD_TYPES
                )
                for r in resp:
                    nid = r['nid']
                    rows[nid_to_idx[nid]][r['rtype']] = r['cnt']
                    prog.update()

        return pd.DataFrame(rows)


    def get_url_feats(self, nids=[], include_id=False):
        '''
        Given list of nids get URL features for those nodes
        If list is None, return all labled nodes
        '''
        if nids:
            query = f'''
                MATCH
                    (n:URL)
                WHERE
                    id(n) in $nids
                OPTIONAL MATCH
                    (n) -- (e:EVENT)
                WHERE
                    e.label <> "UNK"
                RETURN
                    distinct id(n) as nid,
                    properties(n) as feats,
                    e.label as apt
            '''
        else:
            query = f'''
                MATCH
                    (n:URL) -- (e:EVENT)  // Grab all URLs w labels
                WHERE
                    e.label <> "UNK"
                RETURN
                    distinct id(n) as nid,
                    properties(n) as feats,
                    e.label as apt
            '''

        if self.testing:
            query += ' LIMIT 100'

        # Pretty straighforward. Just grab all the features we want
        urls = []
        ordered_nids = []
        iocs = []

        with self.graph.session() as session:
            resp = session.run(query, nids=nids)

            for r in tqdm(resp, desc='Parsing URL query response', total=len(nids)):
                ioc = r['feats'].pop('value')
                url = r['feats']
                url['ioc'] = ioc
                url['apt'] = r['apt']

                ordered_nids.append(r['nid'])
                urls.append(url)
                iocs.append(url['ioc'])

        # Honestly, I don't know why I didn't just do this
        # with the other functions.
        df = extract_url(urls)
        if include_id:
            df['nid'] = ordered_nids

        df['ioc'] = iocs
        return df

    def __structured_bfs(self, edges, visited, frontier, nid_to_val):
        next_frontier = {
            'EVENT': set(),
            'IP': set(),
            'URL': set(),
            'domain': set(),
            'ASN': set()
        }

        # Add self-loops to all nodes so they are
        # represented in the graph even if they have
        # no neighbors
        for value in frontier.values():
            for n in value:
                edges.add((n,n))

        def add_q_results(resp):
            for edge in resp:
                s = edge.get('src')
                d = edge.get('dst')
                dtype = edge.get('dtype')

                edges.add((s,d))
                edges.add((d,s))

                if d not in visited[dtype[0]]:
                    next_frontier[dtype[0]].add(d)
                    nid_to_val[d] = edge.get('val')

                visited[dtype[0]].add(d)

        # Get IP neighbors
        if frontier['IP']:
            ip_q = f'''
                MATCH (ip:IP) -- (o)
                WHERE
                    id(ip) in $nids and
                    (o:EVENT or o:URL or o:domain or o:ASN)
                RETURN
                    id(ip) as src,
                    id(o) as dst,
                    o.value as val,
                    labels(o) as dtype
            '''

            with self.graph.session() as session:
                ip_neighbors = session.run(ip_q, nids=list(frontier['IP']))
                add_q_results(ip_neighbors)
            visited['IP'].update(frontier['IP'])

        # Get URL neighbors
        if frontier['URL']:
            url_q = f'''
                MATCH (url:URL) -- (o)
                WHERE
                    id(url) in $nids and
                    (o:EVENT or o:IP)
                RETURN
                    id(url) as src,
                    id(o) as dst,
                    o.value as val,
                    labels(o) as dtype
            '''

            with self.graph.session() as session:
                url_neighbors = session.run(url_q, nids=list(frontier['URL']))
                add_q_results(url_neighbors)

            visited['URL'].update(frontier['URL'])

        # Get domain neighbors
        if frontier['domain']:
            dom_q = f'''
                MATCH (d:domain) -- (o)
                WHERE
                    id(d) in $nids and
                    (o:URL or o:IP)
                RETURN
                    id(d) as src,
                    id(o) as dst,
                    o.value as val,
                    labels(o) as dtype
            '''

            with self.graph.session() as session:
                d_neighbors = session.run(dom_q, nids=list(frontier['domain']))
                add_q_results(d_neighbors)

            visited['domain'].update(frontier['domain'])

        # Get ASN neighbors
        if frontier['ASN']:
            asn_q = f'''
                MATCH (a:ASN) -- (o)
                WHERE
                    id(a) in $nids and
                    o:IP
                RETURN
                    id(a) as src,
                    id(o) as dst,
                    o.value as val,
                    labels(o) as dtype
            '''
            with self.graph.session() as session:
                a_neighbors = session.run(asn_q, nids=list(frontier['ASN']))
                add_q_results(a_neighbors)

            visited['ASN'].update(frontier['ASN'])

        if frontier['EVENT']:
            event_q = f'''
                MATCH (e:EVENT) -- (o)
                WHERE
                    id(e) in $nids and
                    (o:IP or o:URL)
                RETURN
                    id(e) as src,
                    id(o) as dst,
                    o.value as val,
                    labels(o) as dtype
            '''
            with self.graph.session() as session:
                e_neighbors = session.run(event_q, nids=list(frontier['EVENT']))
                add_q_results(e_neighbors)

            visited['EVENT'].update(frontier['EVENT'])

        return next_frontier

    def get_subgraph(self, nodes, k_hops=2, group_input=True, as_csr=False):
        '''
        Expects dict with keys 'IP', 'URL', 'domain', 'ASN'
        And values as lists of ioc strings
        and will return k-hop subgraph for GNN to use

            Only these relations are considered:
                    EVENT -> IP, URL
                    IP <-> URL
                    IP <-> ASN
                    IP <-> domain
                    URL <-> domain

        If group_input is true, assumes the input nodes are
        grouped in some way, and will add a synthetic "EVENT" node
        to the output subgraph, connecting to all the inputs
        '''
        nid_to_val = dict()
        frontier = {
            'EVENT': set(),
            'IP': set(),
            'URL': set(),
            'domain': set(),
            'ASN': set()
        }
        visited = {
            'EVENT': set(),
            'IP': set(),
            'URL': set(),
            'domain': set(),
            'ASN': set()
        }

        edges = set()
        input_ids = set()
        not_found = dict()

        # Get nids for everything
        for ntype in frontier.keys():
            if not nodes.get(ntype):
                continue

            query = f'''
                MATCH (n:{ntype})
                WHERE
                    n.value in $vals
                RETURN
                    id(n) as nid,
                    n.value as ioc
            '''
            with self.graph.session() as session:
                full_query = set(nodes[ntype])
                resp = session.run(query, vals=list(full_query))
                found = {r.get('nid'):r.get('ioc') for r in resp}

                not_found_ = full_query - set(found.values())
                not_found[ntype] = list(not_found_)

            input_ids.update(found.keys())
            frontier[ntype].update(found.keys())
            nid_to_val.update(found)

        # Populates edges/visited iteratively using BFS algorithm but only
        # following allowed edges in the schema
        for _ in range(k_hops):
            frontier = self.__structured_bfs(edges, visited, frontier, nid_to_val)

        # Build graph of edges
        g = self.__to_torch(
            [*zip(*edges)],
            visited['EVENT'],
            visited['IP'],
            visited['URL'],
            visited['domain'],
            visited['ASN']
        )

        # Normally just saved as neo_id -> graph_id
        names = [''] * g.x.size(0)
        for nid,gid in g.node_names.items():
            names[gid] = nid_to_val[nid]

        if group_input:
            gids = [g.node_names[n] for n in input_ids]
            fake_id = (g.edge_index.max()+1).item()
            e_edges = [fake_id] * len(gids)

            # Add edges to fake node into graph
            g.edge_index = torch.cat([
                g.edge_index, torch.tensor([gids+e_edges, e_edges+gids])
            ], dim=1)

            # Add event to end of list events
            fake_x = torch.tensor([g.ntypes['EVENT']])
            names.append('plan')

            g.feat_map = torch.cat([g.feat_map, torch.tensor([-1])])
            g.x = torch.cat([g.x, fake_x])

            important = torch.tensor([fake_id]+gids)

        g.node_names = names
        feats, feat_map = self.get_features_for_gnn(g, out_dir=None, skip_empty=True)
        g.feat_map = feat_map


        # Convert to CSR matrix
        if as_csr:
            g.edge_csr = CSR(g.edge_index)
            del g.edge_index

        if group_input:
            return g, feats, important, not_found

        return g, feats, not_found

    def __get_gnn_edges(self, sources):
        '''
        Build graph used by GNNs. Schema only includes the following ntypes and etypes:

                    EVENT -> IP, URL
                    IP <-> URL
                    IP <-> ASN
                    IP <-> domain
                    URL <-> domain

        '''
        sources_str = [f'"{s}"' for s in sources]
        if sources == []:
            condition = ''
        else:
            condition = f"e.source in [{','.join(sources_str)}] and "

        edges = set()
        uq_events = set()

        # First get 1-hop IP connections
        uq_ips = set()
        print("Getting EVENT -> IP")
        query = f'''
            MATCH (e:EVENT) -- (n)
            WHERE
                {condition}
                n:IP
            RETURN
                id(e) as eid,
                id(n) as nid
        '''
        if self.testing:
            query += ' LIMIT 100'

        with self.graph.session() as s:
            ips = s.run(query)

            for ip in ips:
                eid = ip['eid']
                ioc = ip['nid']

                # Add bi-directional edge
                edges.add((eid,ioc))
                edges.add((ioc,eid))

                # For tracking later
                uq_events.add(eid)
                uq_ips.add(ioc)

        # Next get 1-hop URL connections
        uq_urls = set()
        print("Getting EVENT -> URL")
        query = f'''
            MATCH (e:EVENT) -- (n)
            WHERE
                {condition}
                n:URL
            RETURN
                id(e) as eid,
                id(n) as nid
        '''
        if self.testing:
            query += ' LIMIT 100'

        with self.graph.session() as s:
            urls = s.run(query)
            for url in urls:
                eid = url['eid']
                ioc = url['nid']

                # Add bi-directional edge
                edges.add((eid,ioc))
                edges.add((ioc,eid))

                # For tracking later
                uq_events.add(eid)
                uq_urls.add(ioc)

        # Next get edges between urls/IPs
        # Minibatched
        all_neighbors = [s for s in uq_urls]
        n_batches = ceil(len(all_neighbors)/self.BATCH_SIZE)
        prog = tqdm(desc='URL -> IP', total=n_batches)
        for i in range(n_batches):
            batch = all_neighbors[i*self.BATCH_SIZE : (i+1)*self.BATCH_SIZE]
            with self.graph.session() as s:
                query = f'''
                    MATCH (u:URL) -- (i:IP)
                    WHERE
                        id(u) in $batch
                    RETURN
                        id(u) as uid,
                        id(i) as iid
                '''
                url_ip = s.run(query, batch=batch)
                prog.update()

                for edge in url_ip:
                    url = edge['uid']
                    ip = edge['iid']

                    uq_urls.add(url)
                    uq_ips.add(ip)

                    edges.add((url,ip))
                    edges.add((ip,url))

        prog.close()

        # Next get edges between urls/IPs
        # Minibatched
        all_neighbors = [s for s in uq_ips]
        n_batches = ceil(len(all_neighbors)/self.BATCH_SIZE)
        prog = tqdm(desc='IP -> URL', total=n_batches)

        for i in range(n_batches):
            batch = all_neighbors[i*self.BATCH_SIZE : (i+1)*self.BATCH_SIZE]
            query = f'''
                MATCH (u:URL) -- (i:IP)
                WHERE
                    id(i) in $batch
                RETURN
                    id(u) as uid,
                    id(i) as iid
            '''

            with self.graph.session() as s:
                url_ip = s.run(query, batch=batch)
                prog.update()

                for edge in url_ip:
                    url = edge['uid']
                    ip = edge['iid']

                    uq_urls.add(url)
                    uq_ips.add(ip)

                    edges.add((url,ip))
                    edges.add((ip,url))

        prog.close()

        # Get domains connected to IP/URLs we've seen
        # (Minibatch so queries don't take a million years)
        uq_domains = set()
        uq_asns = set()

        all_neighbors = [s for s in uq_ips.union(uq_urls)]
        n_batches = ceil(len(all_neighbors)/self.BATCH_SIZE)
        prog = tqdm(desc='ASNs & Domains', total=n_batches*2)
        for i in range(n_batches):
            batch = all_neighbors[(i*self.BATCH_SIZE) : ((i+1)*self.BATCH_SIZE)]
            query = f'''
                MATCH (n) -- (d:domain)
                WHERE
                    id(n) in $batch
                RETURN
                    id(n) as nid,
                    id(d) as did
            '''

            with self.graph.session() as s:
                domains = s.run(query, batch=batch)
                prog.update()

                for d in domains:
                    dom = d['did']
                    ioc = d['nid']

                    uq_domains.add(dom)
                    edges.add((dom,ioc))
                    edges.add((ioc,dom))

            # Get ASNs attached to IP/URLs
            query = f'''
                MATCH (n) -- (a:ASN)
                WHERE
                    id(n) in $batch
                RETURN
                    id(n) as nid,
                    id(a) as aid
            '''

            with self.graph.session() as s:
                asns = s.run(query, batch=batch)
                prog.update()

                for asn in asns:
                    a = asn['aid']
                    ioc = asn['nid']

                    uq_asns.add(a)
                    edges.add((a,ioc))
                    edges.add((ioc,a))

        prog.close()
        return self.__to_torch([*zip(*edges)], uq_events, uq_ips, uq_urls, uq_domains, uq_asns)


    def __to_torch(self, ei, events, ips, urls, domains, asns):
        print("Building torch.Data object")
        ei = torch.tensor(ei)
        _,uq = ei.unique(return_inverse=True)
        nmap = dict()

        IP=0
        URL=1
        DOMAIN=2
        ASN=3
        EVENT=4
        TYPE_MAP = {'ips': 0, 'urls': 1, 'domains': 2, 'ASN':3, 'EVENT': 4}

        # Build mapping from neo nid to 0-N
        for i in range(ei.size(0)):
            for j in range(ei.size(1)):
                nmap[ei[i][j].item()] = uq[i][j].item()

        # Fill in feature vector (just have ntypes, query for feats using ids later)
        ei = uq
        x = torch.zeros(uq.max()+1, dtype=torch.long)
        feat_map = torch.zeros(uq.max()+1, dtype=torch.long)

        emap = dict()
        eid = 0
        event_ids = []
        for u in urls:
            x[nmap[u]] = URL
            feat_map[nmap[u]] = u
        for i in ips:
            x[nmap[i]] = IP
            feat_map[nmap[i]] = i
        for a in asns:
            x[nmap[a]] = ASN
            feat_map[nmap[a]] = -1 # ASNs dont have features
        for d in domains:
            x[nmap[d]] = DOMAIN
            feat_map[nmap[d]] = d
        for e in events:
            x[nmap[e]] = EVENT
            feat_map[nmap[e]] = -1 # Events don't have features
            event_ids.append(nmap[e])

            # Used for labelling
            emap[e] = eid
            eid += 1

        event_ids = torch.tensor(event_ids)

        # No longer need to go from neo-id to uuid, instead need to go backward
        inv_map = {v:k for k,v in nmap.items()}
        inv_map = torch.tensor([inv_map[i] for i in range(len(nmap))])

        # Finally, just need to get labels
        query = f'''
            MATCH (e:EVENT)
            WHERE
                id(e) in $events
            return
                id(e) as eid,
                e.label as apt,
                e.source as src
        '''
        print("Getting labels")

        src_map = dict()
        src_id = 0
        with self.graph.session() as s:
            resp = s.run(query, events=[e for e in events])

            labels = []
            sources = []
            idx = []
            for y in resp:
                labels.append(y['apt'])

                if (sid := src_map.get(y['src'])) is None:
                    sid = src_map[y['src']] = src_id
                    src_id += 1

                sources.append(sid)
                idx.append(y['eid'])

        # So GNN is consistant with labels
        lmap = {
            0:'APT28',
            1:'TA511',
            2:'ICEFOG',
            3:'APT35',
            4:'COBALT GROUP',
            5:'APT38',
            6:'MOLERATS',
            7:'TA551',
            8:'APT41',
            9:'FIN11',
            10:'GOLD WATERFALL',
            11:'FIN7',
            12:'TEAMTNT',
            13:'APT29',
            14:'APT27',
            15:'TURLA',
            16:'KIMSUKY',
            17:'MUSTANG PANDA',
            18:'APT37',
            19:'BLACKENERGY',
            20:'MAGECART',
            21:'MUDDYWATER',
            22:'APT34',
            23:'SAPPHIRE MUSHROOM'
        }
        inv_lmap = {v:k for k,v in lmap.items()}

        ys = torch.zeros(event_ids.size(0), dtype=torch.long)
        for i in range(len(idx)):
            if labels[i] in inv_lmap:
                ys[emap[idx[i]]] = inv_lmap[labels[i]]

        print("Done")
        return Data(
            x=x,
            edge_index=ei,
            label_map=lmap,
            feat_map=feat_map,
            y=ys,
            sources=torch.tensor(sources),
            src_map=src_map,
            event_ids=event_ids,
            ntypes=TYPE_MAP,
            node_names=nmap
        )

    def get_features_for_gnn(self, g, out_dir=None, skip_empty=False):
        # Start building full feature dataframes for each node
        getters = {
            'ips': self.get_ip_feats,
            'urls': self.get_url_feats,
            'domains': self.get_domain_feats
        }
        densifiers = {
            'ips': ip_dense_gnn,
            'urls': url_dense_gnn,
            'domains': domain_dense_gnn
        }

        dfs = dict()
        feat_map = g.feat_map
        ntypes = g.ntypes
        x = g.x
        del g

        for ftype in ['domains', 'urls', 'ips']:
            # Get dataframe with sparse data for ioc type
            iocs = feat_map[x == ntypes[ftype]]
            iocs = [i.item() for i in iocs]

            if not len(iocs) and skip_empty:
                continue

            print(f"Getting features for {len(iocs)} {ftype}...")
            df = getters[ftype](nids=iocs, include_id=True)

            # This is already held in the graph, no need to double-save it
            df.pop('apt')

            aggs = dict.fromkeys(df, 'first')                               # Returns first non-None value
            aggs.pop('nid')                                                 # Don't need to agg on primary key
            #aggs['apt'] = lambda apts : list(set([x for x in apts if x]))   # Want to retain multi APT values
            df = df.groupby('nid', as_index=False).agg(aggs)

            # But need to figure out which row each ioc's data was saved on
            nids = df.pop('nid')
            nids = {n:i for i,n in enumerate(nids)}

            new_feat_map = []
            for ioc in iocs:
                new_feat_map.append(
                    nids.get(ioc, -1)
                )

            # Update to new locations within dataframe
            feat_map[x == ntypes[ftype]] = torch.tensor(new_feat_map)

            if out_dir:
                # Save and go to next ioc type
                df.to_csv(f'{out_dir}/{ftype}.csv', sep='\t')
            else:
                # We assume few enough nodes to hold dense features in memory
                dfs[ftype] = torch.from_numpy(densifiers[ftype](df)[0])

        if out_dir:
            # Update feat map and save
            g = torch.load(f'{out_dir}/full_graph_csr.pt')
            g.feat_map = feat_map
            torch.save(g, f'{out_dir}/full_graph_csr.pt')

        return dfs, feat_map

    def build_gnn_data(self, out_dir, sources=[]):
        '''
        Runs in about 20 minutes for just OTX
        '''
        # Builds edge_index Data object with pointers to
        # nids in the neo graph for feature data
        g = self.__get_gnn_edges(sources)
        print("Converting edge_index -> edge_csr")
        g.edge_csr = CSR(g.edge_index)
        del g.edge_index

        # Save before doing features so we can free some mem
        torch.save(g, f'{out_dir}/full_graph_csr.pt')

        #g = torch.load(f'{out_dir}/test.pt')
        self.get_features_for_gnn(g, out_dir=out_dir)

        # Finally, save graph
        torch.save(g, f'{out_dir}/full_graph_csr.pt')

    def ioc_str_to_gid(self, iocs, node_type):
        '''
        Returns mapping of ioc -> id(node)
        Need to do this for iocs of a single node type (e.g. IPs, URLs, etc.)
        because in the database, they are all stored in different relation tables.

        Should be pretty fast, ioc strings are the primary key
        '''
        query = f'''
            match (n:{node_type})
            where n.value in $iocs
            return
                id(n) as nid,
                n.value as ioc
        '''

        ret_map = dict()

        # If iocs == [] just return empty dict
        if not iocs:
            return ret_map

        with self.graph.session() as s:
            ret = s.run(query, iocs=iocs)
            for r in ret:
                ret_map[r['ioc']] = r['nid']

        return ret_map

    def get_features(self, iocs):
        '''
        Expects list of [
         {'ioc': ioc string, 'type': ioc type}
        ]
        where ioc type in [IP,URL,domain,host] (otherwise returns no features)
        '''
        ips, urls, domains, unk = [],[],[],[]
        for ioc in iocs:
            ioc_type = ioc['type'].upper()
            ioc_val = ioc['ioc']

            if ioc_type == 'IP':
                ips.append(ioc_val)
            elif ioc_type == 'URL':
                urls.append(ioc_val)
            elif ioc_type == 'DOMAIN' or ioc_type == 'HOST':
                domains.append(ioc_val)


        fn_map = {
            'IP': (self.get_ip_feats, ip_to_dense),
            'URL': (self.get_url_feats, url_to_dense),
            'domain': (self.get_domain_feats, domain_to_dense)
        }

        def get_one_type(ioc_strs, ioc_type):
            if ioc_strs == []:
                return (None,None,None),None

            nid_map = self.ioc_str_to_gid(ioc_strs, ioc_type)

            feat_getter, densifier = fn_map[ioc_type]
            df = feat_getter(nids=list(nid_map.values()))

            # Add any values that weren't in database as rows
            # with NaN values (can be removed if not useful...)
            iocs_wanted = set(nid_map.keys())
            iocs_captured = set(df['ioc'])
            missed_iocs = iocs_wanted - iocs_captured

            if missed_iocs:
                df = pd.concat([
                    df,
                    pd.DataFrame([{'ioc':ioc} for ioc in missed_iocs])
                ])

            iocs = df.pop('ioc').tolist()
            return densifier(df),iocs

        # Do in parallel
        dfs = Parallel(
            n_jobs=3 if not self.testing else 1,
            prefer='threads'
        )(
            delayed(get_one_type)(*args)
            for args in [(ips, 'IP'), (urls, 'URL'), (domains, 'domain')]
        )

        def structured(dense_ret, iocs):
            return {
                'X': dense_ret[0],
                'y': dense_ret[1],
                'columns': dense_ret[2],
                'iocs': iocs
            }

        # Make a little easier to deal with before returning
        return {
            'IP': structured(*dfs[0]),
            'URL': structured(*dfs[1]),
            'domain': structured(*dfs[2])
        }