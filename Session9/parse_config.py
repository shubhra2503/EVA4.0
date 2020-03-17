# -*- coding: utf-8 -*-
#!/usr/bin/env python

import yaml
from pdb import set_trace as bp

def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def iterdict(d):
    for k,v in d.items():        
        if isinstance(v, dict):
            iterdict(v)

        elif isinstance(v, str) and ("," in v):
            d[k] = [int(f) if isint(f)
                     else float(f) if isfloat(f)
                     else f.strip() if isinstance(f, str)
                     else f for f in v.split(",")]
    return d

def load_hparam(filename):
    stream = open(filename, 'r')
    docs = yaml.load_all(stream)
    hparam_dict = dict()
    for doc in docs:
        doc = iterdict(doc)
        for k, v in doc.items():
            
            hparam_dict[k] = v
    return hparam_dict


def merge_dict(user, default):
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = merge_dict(user[k], v)
    return user


class Dotdict(dict):
    """
    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None):
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = Dotdict(value)
            self[key] = value


class Hparam(Dotdict):

    def __init__(self, file='config.yaml'):
        super(Dotdict, self).__init__()
        hp_dict = load_hparam(file)
        hp_dotdict = Dotdict(hp_dict)
        for k, v in hp_dotdict.items():
            setattr(self, k, v)
            
    __getattr__ = Dotdict.__getitem__
    __setattr__ = Dotdict.__setitem__
    __delattr__ = Dotdict.__delitem__

        
hparam = Hparam()