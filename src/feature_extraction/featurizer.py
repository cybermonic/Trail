import glob
import json

from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

from . import ip, url, domain

class Featurizer():
    '''
    Converts raw feature dictionaries (such as those returned by the OTX
    enrichment module) into dataframes which can be easilly turned to vectors
    '''
    def get_features(self, samples, dense=False):
        ips, domains, urls = [],[],[]

        for sample in samples:
            ioc_type = sample['type']

            if ioc_type.upper() == 'IP':
                ips.append(sample)
            elif ioc_type.upper() in ['DOMAIN', 'HOSTNAME']:
                domains.append(sample)
            elif ioc_type.upper() == 'URL':
                urls.append(sample)
            else:
                print("I don't know how to parse IOC type %s" % ioc_type)

        del samples # Save memory

        # Parse in parallel bc why not
        jobmap = {0:ip.extract, 1:domain.extract, 2:url.extract}
        def thread_job(i,samples,dense): return jobmap[i](samples,dense=dense)

        return Parallel(n_jobs=1, prefer='threads')(
            delayed(thread_job)(i,samples,dense)
            for i,samples in enumerate([ips, domains, urls])
        )

    def get_ip_features(self, samples, dense=False):
        return ip.extract(
            [s for s in samples if s['type'] == 'IP'],
            dense=dense
        )
    def get_domain_features(self, samples, dense=False):
        return domain.extract(
            [s for s in samples if s['type'] in ['domain', 'hostname']],
            dense=dense
        )
    def get_url_features(self, samples, dense=False):
        return url.extract(
            [s for s in samples if s['type'] == 'URL'],
            dense=dense
        )

    def get_feat_dict(self, samples):
        '''
        For compatibility with old featurizer
        '''
        return [df.to_dict() for df in self.get_features(samples)]



def build_otx_features(config):
    '''
    Parses all OTX files output by
    greyhound_graph.injest.alienvault_otx.pull:build_dataset()
    '''
    apts = glob.glob(config['OTX_DIR'] + '*/')
    out_dir = config['OTX_FEATS']
    parser = Featurizer()

    parsed = set()
    idx = dict(ips=dict(), domains=dict(), urls=dict())
    ip_idx=0; domain_idx=0; url_idx=0

    # Clear old files so function is idempotent
    open(out_dir + 'ips.csv', 'w+').close()
    open(out_dir + 'ips_map.csv', 'w+').close()
    open(out_dir + 'domains.csv', 'w+').close()
    open(out_dir + 'domains_map.csv', 'w+').close()
    open(out_dir + 'urls.csv', 'w+').close()
    open(out_dir + 'urls_map.csv', 'w+').close()

    def flush(iocs, idx, apt, first_write=False):
        ips, domains, urls = parser.get_features(iocs)
        args = [(ips,'ips'), (domains,'domains'), (urls,'urls')]

        for df,fname in args:
            df['apt'] = apt
            with open(out_dir + fname + '.csv', 'a') as f:
                f.write(df.to_csv(sep='\t', header=first_write, index=False))
            with open(out_dir + fname + '_map.csv', 'a') as f:
                for k,v in idx[fname].items():
                    f.write(f'{k},{v}\n')

            idx[fname] = dict()

    first_write = True
    for apt in apts:
        events = glob.glob(apt+'*.json')
        iocs = []
        apt_str = apt.split('/')[-2]

        # Known to be dirty/full of missing data
        if apt_str == 'PAT BEAR':
            continue

        for event in tqdm(events, desc=apt_str):
            with open(event, 'r') as f:
                file_iocs = json.load(f)['iocs']

            to_parse = []
            for ioc in file_iocs:
                uuid = ioc['ioc']+'-'+ioc['type']

                if uuid not in parsed:
                    to_parse.append(ioc)
                    parsed.add(uuid)

                    # Keep track of which row each IOC appears on
                    if ioc['type'] == 'IP':
                        idx['ips'][ioc['ioc']] = ip_idx
                        ip_idx += 1
                    elif ioc['type'] == 'URL':
                        idx['urls'][ioc['ioc']] = url_idx
                        url_idx += 1
                    else:
                        idx['domains'][ioc['ioc']] = domain_idx
                        domain_idx += 1

            iocs += to_parse
            # Avoid eating up a bunch of memory
            if len(iocs) > 1028:
                flush(iocs, idx, apt_str, first_write=first_write)
                iocs = []
                first_write = False

        flush(iocs, idx, apt_str)
        first_write = False


    with open(out_dir + 'ioc_map.json', 'w+') as f:
        f.write(json.dumps(idx))