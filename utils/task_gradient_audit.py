"""Validate observational probes separately from completed training evidence.

Nonfinite/unavailable observations are retained and reported, never fabricated
or used as evidence for gradient conflict. Malformed records still fail closed.
"""
import math

from utils.task_gradient_probe import GROUPS


def audit_probe_records(records, variant):
    if variant not in {'aprobe', 'aregionword'}:
        raise ValueError('Unknown gradient audit variant')
    if [(x['epoch'], x['batch']) for x in records] != [
            (e, b) for e in (1, 10, 30, 60) for b in (1, 2)]:
        raise ValueError('Missing or repeated gradient diagnostics')
    counts = {name: {'valid': 0, 'unavailable': 0} for name in GROUPS}
    unavailable = []
    for row in records:
        if (row['version'] != 'same_forward_v1' or row['model_mode'] != 'train'
                or set(row['groups']) != set(GROUPS)):
            raise ValueError('Wrong gradient probe semantics')
        for name, group in row['groups'].items():
            status = group.get('status')
            if status not in {'ok', 'nonfinite', 'no_trainable_parameters'}:
                raise ValueError(f'Unknown gradient diagnostic status: {status!r}')
            if status != 'ok':
                # The original writer deliberately discarded all scalar values
                # in a nonfinite group. Do not invent finite replacements.
                if any(group.get(key) is not None for key in (
                        'quality_norm', 'align_norm', 'cosine', 'align_over_quality')):
                    raise ValueError('Unavailable gradient diagnostics must not claim numeric values')
                counts[name]['unavailable'] += 1
                unavailable.append(dict(epoch=row['epoch'], batch=row['batch'],
                                        group=name, status=status))
            else:
                for key in ('quality_norm', 'align_norm'):
                    value = group.get(key)
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError('Invalid gradient norm')
                cosine = group.get('cosine')
                if cosine is not None and (type(cosine) not in (int, float)
                        or not math.isfinite(cosine) or not -1 <= cosine <= 1):
                    raise ValueError('Invalid gradient cosine')
                if group['quality_norm'] and group['align_norm'] and cosine is None:
                    raise ValueError('Missing cosine for nonzero finite gradients')
                counts[name]['valid'] += 1
            if variant == 'aregionword' and name == 'resnet_local':
                if status == 'ok' and group['align_norm'] != 0:
                    raise ValueError('Region-word alignment unexpectedly depends on ResNet')
                # Even when norms are unavailable, recorded connectivity must
                # not contradict the intended ResNet-free alignment path.
                if any(group.get(key, 0) != 0 for key in ('shared_tensors', 'align_only_tensors')):
                    raise ValueError('Region-word alignment unexpectedly depends on ResNet')
    valid = sum(x['valid'] for x in counts.values())
    invalid = sum(x['unavailable'] for x in counts.values())
    return dict(version='task_gradient_audit_v1',
                diagnostics_status='complete' if not invalid else ('partial' if valid else 'unavailable'),
                valid_group_records=valid, unavailable_group_records=invalid,
                by_group=counts, unavailable=unavailable,
                note='Training completion is verified separately. Unavailable gradients cannot support conflict claims.')
