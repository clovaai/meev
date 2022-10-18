"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import os
import json

def load_json(filename, default = None):
    if not os.path.isfile(filename):
        return default
    with open(filename) as f:
        data = json.load(f)
    return data

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)