import json
import os
from pathlib import Path

file_dir = Path(os.path.dirname(os.path.realpath(__file__)))


class AptLabelMapper():
    def __init__(self):
        self.ta_map = build_ta_map()

    def get_label(self, misp_event):
        # First, check the tags
        for tag in misp_event['Event'].get('Tag', []):
            name = tag['name']

            if 'threat-actor' in name:
                has_apt = True
                k = name.replace(
                    'misp-galaxy:threat-actor=', ''
                ).replace('"', '').upper()

                if k in self.ta_map:
                    return self.ta_map[k]

        # No threat-actor tag, try to parse it from the event title
        title = misp_event['Event']['info']
        for apt in self.ta_map.keys():
            if apt in title.upper():
                return self.ta_map[apt]

        # If no APT inferable, return None
        return None

    def get_label_from_str(self, label):
        return self.ta_map.get(label, label)

    def get_all_labels(self):
        return list(set(list(self.ta_map.values())))


def build_ta_map():
    '''
    Builds a map of APT name -> Official name 
    (Note: official name usually matches ^APT[0-9]+$ )
    '''
    outf = file_dir / 'label_map.json'
    ta_map = dict()

    # If precalculated, return the one on disk
    if os.path.exists(outf):
        with open(outf, 'r') as f:
            lmap = json.load(f)
        return lmap

    # First use source of every known alias for APTs
    with open(file_dir / 'apt_names.csv', 'r') as f:
        ta_names = f.read()

    # Threat Actor Official Name,Confidence,Type,Country,[synonyms, ...]
    for line in ta_names.split('\n')[1:]:
        tokens = line.split(',')
        value = tokens[0].upper()

        ta_map[value] = value
        for synonym in tokens[4:]:
            if synonym:  # Often just ''
                ta_map[synonym.upper()] = value

    # Now use galaxy labels. Often there is overlap
    with open(file_dir / 'galaxy-labels.json', 'r', encoding='utf-8') as f:
        tas = json.load(f)

    # Build out list of synonyms
    for ta in tas['values']:
        k = ta['value'].upper()
        aliases = ta.get('meta', dict()).get('synonyms', [])

        official = k

        # Check if appears in synonym map
        if k in ta_map:
            official = ta_map[k]
        # Check if any synonym is in synonym map
        # (Establish official name both sources agree on)
        else:
            for alias in aliases:
                alias = alias.upper()
                if alias in ta_map:
                    official = ta_map[alias]
                    break

        for ta_name in aliases + [k]:
            ta_map[ta_name.upper()] = official

    # Save so above only happens once
    with open(outf, 'w') as f:
        json.dump(ta_map, f)

    return ta_map


def get_label(misp_event):
    ta_map = build_ta_map()

    # First, check the tags
    for tag in misp_event['Event'].get('Tag', []):
        name = tag['name']

        if 'threat-actor' in name:
            has_apt = True
            k = name.replace(
                'misp-galaxy:threat-actor=', ''
            ).replace('"', '').upper()

            if k in ta_map:
                return ta_map[k]

    # No threat-actor tag, try to parse it from the event title
    title = misp_event['Event']['info']
    for apt in ta_map.keys():
        if apt in title.upper():
            return ta_map[apt]

    # If no APT inferable, return None
    return None
