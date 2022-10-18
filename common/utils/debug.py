"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

from config import cfg
from utils.cfg_utils import getBooleanFromCfg

def printDebug(*args, **kwargs):
    if isDebug():
        print(*args, **kwargs)    

def isDebug():
    return getBooleanFromCfg(cfg, 'debug', False)