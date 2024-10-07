from collections.abc import Iterable
from math import log2
import re

from dateutil.parser import parse
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from .utils import get_url_onehot, get_tld_feats, order_df_cols
from .columns import URL_COLS

# Keep on heap after loading in once
ONE_HOT_ENCODER = None

def _ts_fmt(ts):
    '''
    For some reason, formatted differently between
    URLs and domains >>

    e.g. Thu, 01 Jan 1970 00:00:01 GMT
    '''
    if ts in ['None', '-1', -1, '0', 0, None]:
        return None

    # Too many formats to try to do this by hand
    try:
        ts = parse(ts)
    except:
        # Sometimes ts is a list
        # e.g. Thu, 01 Jan 1970 00:00:00 GMT, Thu, 01 Jan 1970 00:00:00 GMT
        # in that case just grab the first one (they're usually the same anyway)
        og_ts = ts
        ts = ts.split(',')
        if len(ts) == 1:
            return None

        ts = ts[1]
        try:
            ts = parse(ts)
        except:
            return None

    return ts.timestamp()

def _server_parse(servers, label_encoder):
    '''
    Arguments:
        servers: list of strings denoting the server data of a URL

    Server data is *usually* (but not always) formatted
    [ServerSoftware]/[version] ([OS]) [[Service]/[Version], ...]

    E.g.
        'Microsoft-IIS/8.5'
        'Apache/2.4.10 (Win32) OpenSSL/1.0.1i PHP/5.6.3'
        'openresty'
        'Apache/2.2.22 (Debian)'
        'Apache/2.4.25 (Win64) OpenSSL/1.0.2j PHP/5.6.19'
        'Apache Phusion_Passenger/4.0.10 mod_bwlimited/1.4'
    '''
    cols = []

    # Sometimes it'll say Apache (2.1.2) or something. We don't really
    # care what version it's running (right now). Maybe a feature for
    # a future date?
    # Right now removes any number followed by a dot (e.g. XXX 2.0) or any value
    # in parenthesis (e.g Apache (v1))
    server_version_pattern = re.compile(r"(( *|)\d\..*)|( *\(.*\))")

    # Sometimes get timestamps instead of os names (?)
    yyyy_dd_mm = re.compile(r"\d{4}-\d{2}-\d{2}")

    for s in servers:
        # Skip Null/nan values
        if not s or type(s) != str:
            cols.append(dict())
            continue

        s = s.upper()
        d = dict()

        server_software = s.split('/')
        d['server-software'] = re.sub(
            server_version_pattern, '', server_software[0]
        ).strip()

        if len(server_software) == 1:
            cols.append(d)
            continue

        # Edge case if os == (LINUX/SUSE) need to rejoin
        version_os_service = '/'.join(server_software[1:]).split(' ')
        if len(version_os_service) == 1:
            cols.append(d)
            continue

        if version_os_service[1].startswith('('):
            # Trim out parentheses
            os = version_os_service[1][1:-1]
            if os:
                # Sometimes get timestamps after instead of OS
                if os[:3] in ['H;M', 'TH;', 'TM;', 'TG;'] or re.match(yyyy_dd_mm, os):
                    cols.append(d)
                    continue

                # I don't know why servers aren't consistant. This hurts my soul.
                # for reasons only known to god ruby gets formatted with parenthesis
                # so it looks like an OS.
                if os[:4] == 'RUBY':
                    version_os_service[1] = version_os_service[1][1:-1]
                    services = version_os_service[1:]

                else:
                    d['os'] = os
                    services = version_os_service[2:]
        else:
            services = version_os_service[1:]

        services = [
            s.split('/')[0]
            for s in services
            if '/' in s and s[0] != '(' # Edge case for long list of multiple servers(?)
        ]

        d['services'] = services
        cols.append(d)

    rows = []
    for c in cols:
        sparse_row = []

        if (srv := c.get('server-software')):
            if (oh := label_encoder.get('server-' + srv)):
                sparse_row.append(oh)

        if (os := c.get('os')):
            if (oh := label_encoder.get('os-' + os)):
                sparse_row.append(oh)

        if (services := c.get('services')):
            oh = [label_encoder.get('service-'+s) for s in services]
            oh = [o for o in oh if o]
            sparse_row += oh

        rows.append(sparse_row)

    return rows

def extract(urls, dense=False):
    '''
    Expects dict or list of dicts of URL data returned from
    greyhound_graph.ingest.enrichment.otx_alien_vault.py:enrich_url()
    '''
    # Only load in once
    global ONE_HOT_ENCODER
    if ONE_HOT_ENCODER is None:
        ONE_HOT_ENCODER = get_url_onehot()

    if isinstance(urls, dict):
        urls = [urls]

    # Get the easier one-hot features
    sparse_rows = []
    for url in urls:
        row = []
        for k in ['filetype', 'fileclass', 'http_code', 'encoding']:
            if k in url and (oh := ONE_HOT_ENCODER.get(f'{k}-{url[k]}')):
                row.append(oh)

        sparse_rows.append(row)

    # Parse out features from server
    servers = [url.get('server','') for url in urls]
    server_oh = _server_parse(servers, ONE_HOT_ENCODER)
    sparse_rows = [sparse_rows[i] + server_oh[i] for i in range(len(urls))]

    # Prepare for turning into dataframe
    rows = [{'one_hot': oh} for oh in sparse_rows]

    # Add in non-one-hot feature (just 'expiration' for URLs)
    # needs to be formatted to convert to timestamp
    for i,url in enumerate(urls):
        rows[i]['expiration'] = _ts_fmt(url.get('expiration'))
        rows[i]['apt'] = url.get('apt')

        # TODO can maybe do some nlp on these?
        # for now omitting
        #rows[i]['raw_server'] = url.get('server')
        #rows[i]['title'] = url.get('extracted-title')

    # Extract lexical features using featurizer (simplified)
    for i,url in enumerate(urls):
        rows[i].update(url_lexical_features(url['ioc']))

    # Get TLD features
    tlds = get_tld_feats([url['ioc'] for url in urls])
    [row.update(tlds[i]) for i,row in enumerate(rows)]

    df = pd.DataFrame(rows)
    if dense and not df.empty:
        return to_dense(df)
    return df

def to_dense(df):
    # Only load in once
    global ONE_HOT_ENCODER
    if ONE_HOT_ENCODER is None:
        ONE_HOT_ENCODER = get_url_onehot()

    oh = df.pop('one_hot')

    # When reading in precomputed csv
    if type(oh.iloc[0]) == str:
        oh = [eval(o) for o in oh]

    if 'apt' in df:
        y = df.pop('apt')
    else:
        y = None

    df = order_df_cols(df, URL_COLS)
    vec = np.nan_to_num(df.to_numpy(dtype=float))

    n_feats = max(ONE_HOT_ENCODER.values())+1
    dense = np.zeros((vec.shape[0], n_feats))
    for i,hot in enumerate(oh):
        dense[i,hot] = 1.

    reverse_enc = {v:k for k,v in ONE_HOT_ENCODER.items()}
    X = np.concatenate([vec, dense], axis=1)
    columns = list(df.columns) + [reverse_enc.get(i, 'missing') for i in range(n_feats)]

    return X,y,columns

def to_dense_gnn(df):
    # Only load in once
    global ONE_HOT_ENCODER
    if ONE_HOT_ENCODER is None:
        ONE_HOT_ENCODER = get_url_onehot()

    df = order_df_cols(df, COL_ORDER)
    df.pop('expiration') # All NaN
    oh = df.pop('one_hot')

    # When reading in precomputed csv
    if type(oh.iloc[0]) == str:
        oh = [eval(o) for o in oh]

    if 'apt' in df:
        y = df.pop('apt')
    else:
        y = None

    vec = np.nan_to_num(df.to_numpy(dtype=float))

    n_feats = max(ONE_HOT_ENCODER.values())+1
    dense = np.zeros((vec.shape[0], n_feats))
    for i,hot in enumerate(oh):
        dense[i,hot] = 1.

    reverse_enc = {v:k for k,v in ONE_HOT_ENCODER.items()}
    X = np.concatenate([vec, dense], axis=1)
    columns = list(df.columns) + [reverse_enc.get(i, 'missing') for i in range(n_feats)]

    return X,y,columns

def url_lexical_features(ioc):
    '''
    Copied out of featurizer
    '''
    parsed = urlparse(ioc)

    has_port = parsed.netloc.split(':')
    has_port = int(len(has_port) > 1 and has_port[-1].isdigit())

    dirs = parsed.path.strip('/').split('/')
    num_subdirectories = len(dirs)

    frags = parsed.fragment
    num_fragments = 0 if frags == '' else len(frags.split('#'))

    params = parsed.query
    num_params = 0 if params == '' else len(params.split('&'))

    def entropy(s):
        probs = [s.count(c) / len(s) for c in set(s)]
        return -sum([p * log2(p) for p in probs])

    return {
        'url_entropy': entropy(ioc),
        'url_path_entropy': entropy(parsed.path),
        'url_length': len(ioc),
        'num_periods': ioc.count('.'),
        'num_subdir': num_subdirectories,
        'num_digits': len([i for i in ioc if i.isdigit()]),
        'num_frag': num_fragments,
        'num_params': num_params,
        'url_path_length': len(parsed.path),
        'url_host_length': len(parsed.netloc),
        'has_port': has_port
    }

COL_ORDER = ['one_hot', 'expiration', 'url_entropy', 'url_path_entropy', 'url_length', 'num_periods', 'num_subdir', 'num_digits', 'num_frag', 'num_params', 'url_path_length', 'url_host_length', 'has_port', 'COM', 'NET', 'ORG', 'JP', 'DE', 'UK', 'FR', 'BR', 'IT', 'RU', 'ES', 'ME', 'GOV', 'PL', 'CA', 'AU', 'CN', 'CO', 'IN', 'NL', 'EDU', 'INFO', 'EU', 'CH', 'ID', 'AT', 'KR', 'CZ', 'MX', 'BE', 'TV', 'SE', 'TR', 'TW', 'AL', 'UA', 'IR', 'VN', 'CL', 'SK', 'LY', 'CC', 'TO', 'NO', 'FI', 'US', 'PT', 'DK', 'AR', 'HU', 'TK', 'GR', 'IL', 'NEWS', 'RO', 'MY', 'BIZ', 'IE', 'ZA', 'NZ', 'SG', 'EE', 'TH', 'IO', 'XYZ', 'PE', 'BG', 'HK', 'RS', 'LT', 'LINK', 'PH', 'CLUB', 'SI', 'SITE', 'MOBI', 'BY', 'CAT', 'WIKI', 'LA', 'GA', 'XXX', 'CF', 'HR', 'NG', 'JOBS', 'ONLINE', 'KZ', 'UG', 'GQ', 'AE', 'IS', 'LV', 'PRO', 'FM', 'TIPS', 'MS', 'SA', 'APP', 'LAT', 'apt']