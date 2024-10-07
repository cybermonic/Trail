from urllib.parse import urlparse

import OTXv2
from IndicatorTypes import IPv4, DOMAIN, URL # Part of OTXv2

from build_dataset.utils import infer_ioc_type

# They just had to use .format instead of %.
# Some url IOCs break this so we have to sanitize
def sanitize(s): return s.replace('{','{{').replace('}', '}}')

def enrich_url(otx, url):
    base_dict = {
        'ioc': url,
        'type': 'URL',
        'hostname': urlparse(url).netloc
    }

    try:
        details = otx.get_indicator_details_full(URL, sanitize(url))
    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        return base_dict

    whois = {k:v for k,v in details['general'].items() if k in [
        'net_loc', 'city', 'region', 'latitude', 'longitude', 'country_code'
    ]}
    base_dict.update(whois)

    # The API is a bit sloppy so bear with me. This code is gonna
    # get really ugly really fast
    server_deets = details['url_list']['url_list']
    if server_deets:
        # I don't think it's ever more than one element
        server_deets = server_deets[0]['result']
    else:
        return base_dict

    # I think sometimes it returns a list with just [None] in it?
    # not sure, but it crashes without this line
    if server_deets is None:
        return base_dict

    # Info about webpage
    urlworker = server_deets.get('urlworker', dict())
    for k in ['ip', 'filetype', 'fileclass', 'http_code']:
        base_dict[k] = urlworker.get(k)

    # Server info
    if (resp := urlworker.get('http_response')):
        # Make keys case-insensitive
        new_resp = dict()
        for k in resp.keys():
            new_resp[k.upper()] = resp[k]
        resp = new_resp

        base_dict['server'] = resp.get('SERVER')
        base_dict['expires'] = resp.get('EXPIRES')
        base_dict['cache-control'] = resp.get('CACHE-CONTROL')
        base_dict['encoding'] = resp.get('CONTENT-ENCODING')
        base_dict['content-type'] = resp.get('CONTENT-TYPE')

    # Webpage info (not sure how often these keys are repeated, keep all?)
    if (resp := server_deets.get('extractor')):
        resp = {'extracted-'+k:v for k,v in resp.items()}
        base_dict.update(resp)

    return base_dict


def enrich_ip(otx, ip):
    '''
    Works for IPv4 and IPv6
    '''
    base_dict = {
        'ioc': ip,
        'type': 'IP',
        'resolves_to': []
    }

    try:
        details = otx.get_indicator_details_by_section(
            IPv4, ip, section='general'
        )

        # For some reason, only seems to pop up for IPv6
        whois = {k:v for k,v in details.items() if k in [
            'net_loc', 'city', 'region', 'latitude', 'longitude', 'country_code', 'asn'
        ]}
        base_dict.update(whois)

    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        pass

    try:
        # Passive dns has basically everything we need.
        # no need to waste bandwidth
        details = otx.get_indicator_details_by_section(
            IPv4, ip, section='passive_dns'
        )
    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        return base_dict

    details = details['passive_dns']
    if len(details) == 0:
        return base_dict

    base_dict['asn'] = details[0]['asn']

    for resolution in details:
        base_dict['resolves_to'].append({
            'host':resolution.get('hostname'),
            'record_type':resolution.get('record_type'),
            'first_seen':resolution.get('first'),
            'last_seen':resolution.get('last')
        })

    return base_dict


def enrich_domain(otx, domain):
    base_dict = {
        'ioc': domain,
        'type': 'domain',
        'dns_records': []
    }

    try:
        # Passive dns has basically everything we need.
        # no need to waste bandwidth
        details = otx.get_indicator_details_by_section(
            DOMAIN, domain, section='passive_dns'
        )
    except (OTXv2.NotFound, OTXv2.BadRequest, OTXv2.RetryError):
        return base_dict

    for record in details['passive_dns']:
        base_dict['dns_records'].append({
            k:record.get(k) for k in ['address', 'first', 'last', 'record_type', 'asn']
        })

    return base_dict


def enrich_host(otx, host):
    '''
    Same as above
    '''
    ret_dict = enrich_domain(otx, host)
    ret_dict['type'] = 'hostname'
    return ret_dict


def enrich(otx, ioc, ioc_type):
    if ioc_type.lower() == 'domain':
        return enrich_domain(otx, ioc)
    elif ioc_type in ['IPv4', 'IPv6', 'IP']:
        return enrich_ip(otx, ioc)
    elif ioc_type == 'URL':
        return enrich_url(otx, ioc)
    elif ioc_type.lower() == 'hostname':
        return enrich_host(otx, ioc)
    else:
        raise TypeError("I don't know how to enrich %s" % ioc_type)

class EnrichOTX():
    '''
    I'm more of a functional programming guy, so I wrote the
    code as functions above. This is just a wrapper if we ever
    need this as an object.
    '''
    def __init__(self, api_key):
        self.otx = OTXv2.OTXv2(api_key)

    def enrich_ip(self, ip):
        return enrich_ip(self.otx, ip)
    def enrich_domain(self, domain):
        return enrich_domain(self.otx, domain)
    def enrich_host(self, host):
        return enrich_host(self.otx, host)
    def enrich_url(self, url):
        return enrich_url(self.otx, url)

    def enrich(self, ioc, ioc_type=None):
        if ioc_type is None:
            # Sort of dangerous, will raise TypeError if
            # it can't infer the type. Better to provide
            # type if known.
            ioc_type = infer_ioc_type(ioc)

        return enrich(self.otx, ioc, ioc_type)