"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

def getBooleanFromCfg(cfg, key:str, default=False) -> bool:
    try:
        if hasattr(cfg, key):
            return bool(getattr(cfg, key))
    except:
        pass
    return default

def getIntFromCfg(cfg, key:str, default=0) -> int:
    try:
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    except:
        pass
    return default

def getFloatFromCfg(cfg, key:str, default=0.0) -> float:
    try:
        if hasattr(cfg, key):
            return float(getattr(cfg, key))
    except:
        pass
    return default

def getStringFromCfg(cfg, key:str, default='') -> str:
    try:
        if hasattr(cfg, key):
            return str(getattr(cfg, key))
    except:
        pass
    return default

def getAnyFromCfg(cfg, key:str, default={}):
    try:
        if hasattr(cfg, key):
            return getattr(cfg, key)
    except:
        pass
    return default