from math import log2

from dateutil.parser import parser
import numpy as np 
import pandas as pd 

from .utils import get_tld_feats, order_df_cols

# All record types present in the training set
RECORD_TYPES = ['A', 'AAAA', 'NS', 'CNAME', 'SOA', 'MX', 'TXT', 'PTR', 'DNAME']
RECORD_MAP = {k:i for i,k in enumerate(RECORD_TYPES)}

def extract(domains, dense=False):
    '''
    Expects dict or list of dicts of domain/host data returned from 
    greyhound_graph.ingest.enrichment.otx_alien_vault.py:enrich_host()
    '''
    if isinstance(domains, dict):
        domains = [domains]

    rows = []

    # Featurizer info (when use_whois=False)
    for domain in domains:
        rows.append(nlp_features(domain['ioc']))

    # Get TLD one-hot
    tlds = get_tld_feats([domain['ioc'] for domain in domains])
    [row.update(tlds[i]) for i,row in enumerate(rows)]

    # Try to extract the last seen date if available and 
    # num unique domain records
    p = parser()
    for i,domain in enumerate(domains):
        has_nxdomain = False 
        records = {r:set() for r in RECORD_TYPES}
        start = float('inf')
        end = 0 

        for rec in domain.get('dns_records',[]):
            if rec['address'] == 'NXDOMAIN':
                has_nxdomain = True 

                # First occurence of NXDOMAIN is end of domain's 
                # life--this is the first time the domain does not 
                # resolve, so it's a good metric for 'last_seen'
                date = p.parse(rec['first']).timestamp()
                end = end if date < end else date 
                continue 

            first = p.parse(rec['first']).timestamp()
            last = p.parse(rec['last']).timestamp()
            
            # Add record to record info, add start/end if they 
            # are further in the past/future than what is previously
            # on record 
            records[rec['record_type']].add(rec['address'])
            start = start if first > start else first 
            end = end if last < end else last 

        update_dict = {k:len(v) for k,v in records.items()}
        update_dict['has_nxdomain'] = has_nxdomain
        if start != float('inf'):
            update_dict['first_seen'] = start 
        if end != 0:
            update_dict['last_seen'] = end 

        rows[i].update(update_dict)

    df = pd.DataFrame(rows)
    if dense:
        return to_dense(df)
    return df

def nlp_features(s):
    probs = [s.count(c) / len(s) for c in set(s)]
    entropy = -sum([p * log2(p) for p in probs])

    return {
        'domain_entropy': entropy,
        'domain_length': len(s),
        'num_digits': len([d for d in s if d.isdigit()]),
        'subdomains': s.count('.')
    }

def to_dense(df):
    # Luckilly domain df is already dense. No work here
    if 'apt' in df:
        y = df.pop('apt')
    else:
        y = None
    df = order_df_cols(df, COL_ORDER)
    x = np.nan_to_num(df.to_numpy(dtype=float))
    cols = df.columns 

    return x,y,cols

def to_dense_gnn(df):
    df = order_df_cols(df.copy(), COL_ORDER)

    # Luckilly domain df is already dense. No work here
    if 'apt' in df:
        y = df.pop('apt')
    else: 
        y = None 
        
    if 'last_seen' in df and 'first_seen' in df:
        df['age'] = (df['last_seen'] - df['first_seen']).abs() / (60*60*24*365)
        df.pop('first_seen'); df.pop('last_seen')
    else:
        df['age'] = None 

    x = np.nan_to_num(df.to_numpy(dtype=float))
    cols = df.columns 

    return x,y,cols

# Pulled out of first to_dense to make sure to_numpy is deterministic
COL_ORDER = ['domain_entropy', 'domain_length', 'num_digits', 'subdomains', 'COM', 'NET', 'ORG', 'JP', 'DE', 'UK', 'FR', 'BR', 'IT', 'RU', 'ES', 'ME', 'GOV', 'PL', 'CA', 'AU', 'CN', 'CO', 'IN', 'NL', 'EDU', 'INFO', 'EU', 'CH', 'ID', 'AT', 'KR', 'CZ', 'MX', 'BE', 'TV', 'SE', 'TR', 'TW', 'AL', 'UA', 'IR', 'VN', 'CL', 'SK', 'LY', 'CC', 'TO', 'NO', 'FI', 'US', 'PT', 'DK', 'AR', 'HU', 'TK', 'GR', 'IL', 'NEWS', 'RO', 'MY', 'BIZ', 'IE', 'ZA', 'NZ', 'SG', 'EE', 'TH', 'IO', 'XYZ', 'PE', 'BG', 'HK', 'RS', 'LT', 'LINK', 'PH', 'CLUB', 'SI', 'SITE', 'MOBI', 'BY', 'CAT', 'WIKI', 'LA', 'GA', 'XXX', 'CF', 'HR', 'NG', 'JOBS', 'ONLINE', 'KZ', 'UG', 'GQ', 'AE', 'IS', 'LV', 'PRO', 'FM', 'TIPS', 'MS', 'SA', 'APP', 'LAT', 'A', 'AAAA', 'NS', 'CNAME', 'SOA', 'TXT', 'PTR', 'DNAME', 'has_nxdomain', 'first_seen', 'last_seen', 'apt']