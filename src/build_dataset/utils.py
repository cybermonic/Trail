import os
import json
import pickle
import re
import xml

from joblib import Parallel, delayed
from xmltodict import parse

API = 'https://api.geoiplookup.net/?query='
def get_ip(ip):
    resp = os.popen('wget -qO- ' + API + ip).read()

    # There's a few special chars xml doesn't like
    # I only ran into amperstand, but in case there are
    # others, this try/catch block will find them
    try:
        resp = parse(resp.replace("&", "&#38;")).get('ip')
    except xml.parsers.expat.ExpatError:
        print("ERROR",resp)
        return {}

    # For some IPs there's no data (e.g. 0.0.0.0)
    if resp:
        resp = resp['results']['result']
        return resp
    else:
        return {}


# Stolen from stack overflow
IP_PATTERN = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$")
IP6_PATTERN = re.compile(r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))")

def is_ip(ioc): return re.match(IP_PATTERN, ioc) or re.match(IP6_PATTERN, ioc)

def infer_ioc_type(ioc):
    '''
    Very rough. Given IOC return 'IP', 'URL', 'DOMAIN' or 'UNK'
    '''
    # There will never be a space in a valid IOC
    if ' ' in ioc:
        return 'UNK'

    if is_ip(ioc):
        return 'IP'

    if ioc.startswith('http'):
        return 'URL'

    # Check for subnet mask input
    spl = ioc.split('/')
    if re.match(IP_PATTERN, spl[0]) and len(spl)==2 and spl[1].isdigit():
        return 'subnet'

    # Very sloppy. Anything with a period is a domain
    if ioc.count('.'):
        return 'domain'

    return 'UNK'