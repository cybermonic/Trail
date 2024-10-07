import os 
from math import log2

import numpy as np 

def get_country_code_mapper():
    fname = os.path.join(
        os.path.dirname(__file__),
        'helper_files',
        'country_codes.csv'
    )
    f = open(fname, 'r')

    ccs = {}
    
    i = 0 
    line = f.readline() 
    while(line):
        # E.g.  two = US; three = USA
        two,three = line.strip().split(',')
        ccs[two]=i; ccs[three]=i 
        
        i += 1
        line = f.readline() 

    return ccs 

def get_url_onehot():
    fname = os.path.join(
        os.path.dirname(__file__),
        'helper_files',
        'url_onehot_categories.txt'
    )

    with open(fname, 'r') as f:
        category_str = f.read().strip()
    
    categories = category_str.split(',')
    return {c:i for i,c in enumerate(categories)}

def get_ip_onehot(topk=256):
    fname = os.path.join(
        os.path.dirname(__file__),
        'helper_files',
        'ip_onehot_categories.txt'
    )

    with open(fname, 'r') as f:
        category_str = f.read().strip()

    categories = category_str.split('\n')[:topk]
    return {c.split(',')[0]:i for i,c in enumerate(categories)}


def build_onehot_from_db(table, column, outf):
    '''
    E.g. this is used to generate mapping for asn issuers
    '''
    from postgres.util import connect_ro
    
    outf = os.path.join(
        os.path.dirname(__file__),
        'helper_files',
        outf
    )

    conn = connect_ro()
    cur = conn.cursor()

    # Returns name,count of every unique element in column
    cur.execute(f'''
        SELECT {column}, COUNT({column}) 
        FROM {table}
        GROUP BY {column};
    ''')
    vals = cur.fetchall()
    vals.sort(key=lambda x:x[1], reverse=True)

    # Sanity check 
    if len(vals) > 4:
        print('[',vals[:2],'...',vals[-2:],']')
    else:
        print(vals)

    # Dump to file 
    with open(outf, 'w+') as f:
        for val in vals:
            name,cnt = val
            f.write(f"{name},{cnt}\n")
    

def get_tld_feats(urls, sparse=False):
    top_tlds = get_tld_headers()
    
    def make_row(tld):
        if sparse: 
            if tld in top_tlds: 
                return tld
            else: 
                return None 
            
        row = {k:False for k in top_tlds}
        if tld in row:
            row[tld] = True 
        return row 

    tlds = [
        url.split('.')[-1].split('/')[0].split(':')[0].upper()
        for url in urls 
    ]

    return [make_row(tld) for tld in tlds]

def get_tld_headers(top_k=100):
    fname = os.path.join(
        os.path.dirname(__file__),
        'helper_files',
        'ranked_tlds.csv'
    )

    tlds = []
    with open(fname, 'r') as f:
        f.readline() # Skip header

        for _ in range(top_k):
            line = f.readline()
            tlds.append(line.split(',')[0])

    return tlds 

def order_df_cols(df, col_order): 
    return df.reindex(columns = col_order)