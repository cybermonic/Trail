import numpy as np
import pandas as pd

from .utils import get_ip_onehot, get_country_code_mapper, order_df_cols
from .columns import IP_COLS

CC_MAP = None
ISS_MAP = None

def extract(ips, dense=False):
    '''
    Expects dict or list of dicts of IP data returned from
    greyhound_graph.ingest.enrichment.otx_alien_vault.py:enrich_ip()
    '''
    if isinstance(ips, dict):
        ips = [ips]

    # Only load once
    global CC_MAP, ISS_MAP
    if CC_MAP is None:
        CC_MAP = get_country_code_mapper()
    if ISS_MAP is None:
        ISS_MAP = get_ip_onehot()

    # Get one-hot features
    one_hot = []
    for ip in ips:
        # Country code
        sparse_row = []
        if (cc := ip.get('country_code')):
            if (cc_oh := CC_MAP.get(cc)):
                sparse_row.append(cc_oh)

        # Issuer
        if (asn := ip.get('asn')):
            _, issuer = asn.split(' ', 1)
            if (iss_oh := ISS_MAP.get(issuer)):
                iss_oh += max(CC_MAP.values())+1
                sparse_row.append(iss_oh)

        one_hot.append(sparse_row)

    rows = [{'one_hot': oh} for oh in one_hot]

    # Get value features
    for i,ip in enumerate(ips):
        row = rows[i]
        row['latitude'] = ip.get('latitude')
        row['longitude'] = ip.get('longitude')

    df = pd.DataFrame(rows)
    if dense:
        return to_dense(df)
    return df

def to_dense(df):
    # Only load once
    global CC_MAP, ISS_MAP
    if CC_MAP is None:
        CC_MAP = get_country_code_mapper()
    if ISS_MAP is None:
        ISS_MAP = get_ip_onehot()

    oh = df.pop('one_hot')

    # When reading in precomputed csv
    if type(oh.iloc[0]) == str:
        oh = [eval(o) for o in oh]

    if 'apt' in df:
        y = df.pop('apt')
    else:
        y = None

    df = order_df_cols(df, IP_COLS)
    vec = np.nan_to_num(df.to_numpy(dtype=float))

    cc_feats = max(CC_MAP.values())+1
    iss_feats = max(ISS_MAP.values())+1
    n_feats = cc_feats+iss_feats

    dense = np.zeros((vec.shape[0], n_feats))
    for i,hot in enumerate(oh):
        dense[i,hot] = 1.

    X = np.concatenate([vec, dense], axis=1)
    inv_cc = {v:k for k,v in CC_MAP.items()}
    inv_iss = {v:k for k,v in ISS_MAP.items()}

    columns = \
        list(df.columns) + \
        [inv_cc.get(i, 'missing') for i in range(cc_feats)] + \
        [inv_iss.get(i, 'missing') for i in range(iss_feats)]

    return X,y,columns

def to_dense_gnn(df): return to_dense(df)

COL_ORDER = ['one_hot', 'latitude', 'longitude', 'apt']