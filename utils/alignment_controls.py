"""Backward-compatible independent alignment auxiliary weights."""
import math


def alignment_aux_weights(train):
    shared=float(train.get('lambda_align_aux',.3))
    global_weight=float(train.get('lambda_align_global_aux',shared))
    local_weight=float(train.get('lambda_align_local_aux',shared))
    if any(not math.isfinite(x) or x<0 for x in (shared,global_weight,local_weight)):
        raise ValueError('Alignment auxiliary weights must be finite and nonnegative')
    return global_weight,local_weight
