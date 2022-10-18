"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import sys
import numpy as np
#

def print_eval_result(eval_result: dict):
    s = ''
    if 'mpjpe' in eval_result:
        s += 'MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']) +'\n'
    if 'pa_mpjpe' in eval_result:
        s += 'PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']) +'\n'
    if 'mpvpe' in eval_result:
        s += 'MPVPE: %.2f mm' % np.mean(eval_result['mpvpe']) +'\n'
    if 'pa_mpvpe' in eval_result:
        s += 'PA MPVPE: %.2f mm' % np.mean(eval_result['pa_mpvpe']) +'\n'
        
    print (s)
    return s

