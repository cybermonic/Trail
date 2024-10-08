import pandas as pd 
import numpy as np 

from feature_extraction import ip, domain, url 
from feature_extraction.utils import order_df_cols
from feature_extraction.columns import CMAP

from config import config 

'''
Builds dense numpy matrices from sparse csv files saved on disk
as they can be memmapped and more efficiently queried by the 
GNN model which needs them available. 
'''

READ = config.get('DATASET')
WRITE = config.get('ML_DATA')

fn_map = {
    'ips': ip.to_dense,
    'domains': domain.to_dense,
    'urls': url.to_dense
}

has_onehot = lambda x : False if x == 'domains' else True 

def build(fstr):
    print("Building ", fstr)
    df = pd.read_csv(
        READ+fstr+'.csv',
        sep='\t'
    )
    arr = fn_map[fstr](df)[0]
    np.save(WRITE+fstr, arr)

def get_oh_df(ohs): 
    rows = []
    for oh in ohs:
        if oh == '[]':
            rows.append(dict())
            continue 

        oh = eval(oh)
        row =  {o:True for o in oh}
        rows.append(row)

    return pd.DataFrame(rows)


def build_trad(dataset, fstr):
    print("Building ", fstr)
    df = pd.read_csv(
        dataset+'/'+fstr+'.csv',
        sep='\t'
    )

    if has_onehot(fstr):
        oh_rows = get_oh_df(df.pop('one_hot'))
        df = pd.concat([df, oh_rows], axis=1)

    if 'apt' in df:
        df.pop('apt')
        
    df = df.reindex(columns = CMAP[fstr])
    df = df.astype('float32')
    df = df.to_numpy()
    np.nan_to_num(df, copy=False)

    np.save(f'{dataset}/{fstr}', df)


def main(dataset, trad=False):
    if trad:
        return [build_trad(dataset, k) for k in fn_map.keys()]
    
    [build(k) for k in fn_map.keys()]

def test():
    build('ips')