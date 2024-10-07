from collections import defaultdict
import json
import os
import time
from random import shuffle

from joblib import Parallel, delayed
from OTXv2 import OTXv2, RetryError
from tqdm import tqdm

from build_dataset.enrich import enrich
from build_dataset.label_mapper.apt_label_mapper import build_ta_map

WORKERS_PER_KEY = 5

# Most IOCs one thread will pull. Otherwise, use the
# method multithread_event_pull (TODO)
MAX_JOB_SIZE = 1000

# Valid API IOC types ignoring hashes
iocs_we_want = ['IPv4', 'IPv6', 'domain', 'hostname', 'URL']

file_dir = os.path.dirname(os.path.realpath(__file__))
API_KEY = os.environ.get('OTX_KEY')
if not API_KEY:
    raise KeyError("Please export an OTX API key to your environment with `export OTX_KEY=[your key]`")

def get_otx(api=API_KEY):
    return OTXv2(api)

def get_otx_pulse_ids():
    otx = get_otx()
    ta_map = build_ta_map()

    saved = defaultdict(set)
    apts = ta_map.keys()
    for apt in apts:
        # It would be nice if we could use 'adversary:Sofacy' or something
        # but unfortunately otx is weird and even though it claims you can
        # in the docs, throws a 500 response for that
        resp = otx.search_pulses('tag:"%s"' % apt, max_results=1)

        # Just search 1 the first time, then get all of them next pass
        # not efficient, but neither is OTX
        hits = min(resp['count'], 1000) # Max is 1,250 I think
        if hits:
            resp = otx.search_pulses('tag:"%s"' % apt, max_results=hits)
            nresults = len(resp['results'])

            saved[ta_map[apt]].update(
                [resp['results'][i]['id'] for i in range(nresults)]
            )
            print(
                "Found %d pulses tagged %s (aka %s)" %
                (nresults, apt, ta_map[apt])
            )

        with open(file_dir + '/pulse_ids.json', 'w') as f:
            # Sets arent json serializable
            f.write(json.dumps({k:list(v) for k,v in saved.items()},indent=1))

    return saved

def get_top_apts(db, topk):
    lens = [(k,len(v)) for k,v in db.items()]
    lens.sort(key=lambda x : x[1], reverse=True)

    return lens[:topk]

def get_overlapping_pulses(apt_to_ids):
    '''
    If any two APTs are mapped to the same pulse, assume it's just a massive
    dump of IOCs without organization
    '''
    apt_to_ids = {k:set(v) for k,v in apt_to_ids.items()}
    overlaps = defaultdict(set)

    # I'm sure there's a more efficient way to do this but...
    for k1,v1 in apt_to_ids.items():
        for k2,v2 in apt_to_ids.items():
            if k1 == k2:
                continue

            if (overlap := v1.intersection(v2)):
                [overlaps[uuid].update([k1,k2]) for uuid in overlap]

    return list(overlaps.keys())

def build_list_of_pulse_ids(topk=25):
    with open(file_dir + '/pulse_ids.json', 'r') as f:
        apt_to_ids = json.load(f)

    ignore = get_overlapping_pulses(apt_to_ids)

    filtered = dict()
    for k,v in apt_to_ids.items():
        filtered[k] = [v_ for v_ in v if v_ not in ignore]

    # Just pull out keys, ignore counts
    topk = [x[0] for x in get_top_apts(filtered, topk)]
    pulse_dict = {k:filtered[k] for k in topk}
    return pulse_dict


def get_iocs(otx, iocs, pulse_id, apt, message=''):
    ret = dict(event_id=pulse_id, label=apt)

    enriched_iocs = []
    for ioc in tqdm(iocs, desc=message):
        st = time.time()
        enriched_iocs.append(enrich(otx, *ioc))
        elapsed = time.time()-st

        # Ensure we aren't going faster than API allows
        # 10k r / per hour -> 2.7 r/s -> 0.370370... s/r
        # Otherwise things get very slow
        time.sleep(max((0.3704*WORKERS_PER_KEY)-elapsed, 0))


    ret['iocs'] = enriched_iocs
    return ret

def get_ioc_job(otx, ioc, message, idx, tot_iocs):
    st = time.time()
    enriched = enrich(otx, *ioc)
    print(f'\r{message}: {idx}/{tot_iocs}', end='')

    # Ensure we aren't going faster than API allows
    # 10k r / per hour -> 2.7 r/s -> 0.370370... s/r
    # Otherwise things get very slow
    elapsed = time.time()-st
    time.sleep(max((0.3704*WORKERS_PER_KEY)-elapsed, 0))
    return enriched

def thread_job(otx, event, apt, message, out_dir):
    out_f = os.path.join(out_dir, event)+'.json'

    # Don't re-call API for files we already have
    if os.path.exists(out_f):
        return

    st = time.time()
    try:
        iocs = otx.get_pulse_indicators(event, include_inactive=True)
    except RetryError:
        time.sleep(5)
        try:
            iocs = otx.get_pulse_indicators(event, include_inactive=True)
        except RetryError:
            return

    # Avoid thrashing API
    elapsed = time.time()-st
    time.sleep(max((0.3704*WORKERS_PER_KEY)-elapsed, 0))

    # Get human language details
    st = time.time()
    try:
        deets = otx.get_pulse_details(event)
    except RetryError:
        time.sleep(5)
        try:
            deets = otx.get_pulse_details(event)
        except RetryError:
            deets = dict()

    # Only care about a few keys. The others are usually empty
    deets = {
        k:deets.get(k, '')
        for k in ['name', 'description', 'tags']
    }

    to_fetch = []
    for ioc in iocs:
        if (ioc_type:=ioc['type']) in iocs_we_want:
            to_fetch.append((ioc['indicator'], ioc_type))

    # We'll deal with these later (don't forget!)
    if len(to_fetch) > MAX_JOB_SIZE:
        return

    # Avoid thrashing API
    elapsed = time.time()-st
    time.sleep(max((0.3704*WORKERS_PER_KEY)-elapsed, 0))

    blob = get_iocs(otx, to_fetch, event, apt, message=message)
    blob['details'] = deets

    with open(out_f, 'w') as f:
        json.dump(blob, f)


def fmt_time(elapsed):
    min = int(elapsed / 60)
    sec = int(elapsed % 60)
    return f'{min}m.{sec}s'

def inter_thread_job(otxs, jobs):
    last_used = 0
    n_keys = len(otxs)
    tot_events = len(jobs)

    for i, (event, apt, out_dir) in enumerate(jobs):
        st = time.time()
        out_f = os.path.join(out_dir, event)+'.json'

        # Don't re-call API for files we already have
        if os.path.exists(out_f):
            continue

        try:
            iocs = otxs[last_used % n_keys].get_pulse_indicators(event, include_inactive=True)
            last_used += 1
        except RetryError:
            time.sleep(5)
            try:
                iocs = otxs[last_used % n_keys].get_pulse_indicators(event, include_inactive=True)
                last_used += 1
            except RetryError:
                continue

        # Get human language details
        st = time.time()
        try:
            deets = otxs[last_used % n_keys].get_pulse_details(event)
            last_used += 1
        except RetryError:
            try:
                deets = otxs[last_used % n_keys].get_pulse_details(event)
                last_used += 1
            except RetryError:
                deets = dict()

        # Only care about a few keys. The others are usually empty
        deets = {
            k:deets.get(k, '')
            for k in ['name', 'description', 'tags']
        }

        to_fetch = []
        for ioc in iocs:
            if (ioc_type:=ioc['type']) in iocs_we_want:
                to_fetch.append((ioc['indicator'], ioc_type))

        blob = dict(event_id=event, label=apt, deets=deets)
        iocs = Parallel(
            prefer='threads',
            n_jobs=len(otxs)*WORKERS_PER_KEY,
            batch_size=len(otxs)
        )(
            delayed(get_ioc_job)(
                otxs[(last_used+j) % n_keys],
                ioc, f'({i}/{tot_events})',
                j+1, len(to_fetch)
            ) for j, ioc in enumerate(to_fetch)
        )

        last_used += len(to_fetch)
        last_used %= n_keys

        print(f' elapsed: {fmt_time(time.time()-st)}')

        blob['iocs'] = iocs
        with open(out_f, 'w') as f:
            json.dump(blob, f)

def build_dataset(location, inter_thread=False):
    ids = build_list_of_pulse_ids()

    otxs = [
        get_otx()
    ]

    i = 1
    jobs = []
    for apt,events in ids.items():
        out_dir = os.path.join(location, apt)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # Check file doesn't already exist to better reflect how many
        # files left to parse in the loading bar (purely aesthetic)
        [
            jobs.append((event, apt, out_dir))
            for event in events
            if not os.path.exists( os.path.join(out_dir, event)+'.json' )
        ]

    # So we're not stuck waiting for one APT to finish
    shuffle(jobs)

    if not inter_thread:
        # I think this is still slow enough that we won't be rate limited?
        Parallel(n_jobs=len(otxs)*WORKERS_PER_KEY, prefer='threads')(
            delayed(thread_job)(otxs[i%len(otxs)], j[0], j[1], f'({i}/{len(jobs)})', j[2])
            for i,j in enumerate(jobs)
        )
    # For use on large events with > 1000 IOCs
    else:
        inter_thread_job(otxs, jobs)
