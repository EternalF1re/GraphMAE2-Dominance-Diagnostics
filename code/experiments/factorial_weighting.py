"""Source-exact scientific routing/clipping primitives; requires a compatible native model.

No data loader, training CLI or experiment manager is included.
"""

from contextlib import contextmanager

import math

import torch

def branch_scale(z, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("positive finite branch scale required")
    return z.detach() + scale * (z - z.detach())

def parameter_group(name):
    if name.startswith(("encoder_ema.", "projector_ema.")):
        return "teacher"
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith(("decoder.", "encoder_to_decoder.")) or name == "dec_mask_token":
        return "reconstruction"
    if name.startswith(("projector.", "predictor.")):
        return "latent"
    if name == "enc_mask_token":
        return "other"
    raise ValueError("unclassified parameter: " + name)

def ownership(model):
    groups = {key: [] for key in ('encoder', 'reconstruction', 'latent')}
    for name, parameter in model.named_parameters():
        group = parameter_group(name)
        if not parameter.requires_grad:
            if group != 'teacher':
                raise ValueError('unexpected frozen parameter: ' + name)
            continue
        if group == 'other' and name == 'enc_mask_token':
            group = 'encoder'
        if group not in groups:
            raise ValueError('unowned trainable parameter: ' + name)
        groups[group].append((name, parameter))
    ids = [id(p) for pairs in groups.values() for _, p in pairs]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('parameter ownership must be disjoint and exhaustive')
    return groups

@contextmanager
def branch_context(model, scale):
    handle = model.projector.register_forward_pre_hook(
        lambda module, values: (branch_scale(values[0], scale),))
    try:
        yield
    finally:
        handle.remove()

def norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum())
                         for p in parameters if p.grad is not None))

def clip_gradients(model, protocol):
    if protocol not in ('A', 'B'):
        raise ValueError('clipping protocol must be A or B')
    groups = ownership(model)
    trainable = [p for pairs in groups.values() for _, p in pairs]
    if not all(bool(torch.isfinite(p.grad).all()) for p in trainable if p.grad is not None):
        raise FloatingPointError('nonfinite gradient before clipping')
    pre = {group: norm(p for _, p in pairs) for group, pairs in groups.items()}
    shared = [p for n, p in model.named_parameters() if n == 'enc_mask_token']
    shared_pre = norm(shared)
    total_pre = norm(trainable)
    if protocol == 'A':
        returned = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True))
        coefficient = min(1., 3. / (returned + 1e-6))
    else:
        returned = {group: float(torch.nn.utils.clip_grad_norm_(
            [p for _, p in pairs], 3., error_if_nonfinite=True)) for group, pairs in groups.items()}
        coefficient = {group: min(1., 3. / (value + 1e-6)) for group, value in returned.items()}
    post = {group: norm(p for _, p in pairs) for group, pairs in groups.items()}
    return {'protocol': protocol, 'raw_total_gradient_norm': total_pre,
            'clip_grad_norm_returned_preclip': returned,
            'post_clip_total_gradient_norm': norm(trainable),
            'clipping_coefficient': coefficient,
            **{group + '_preclip_norm': value for group, value in pre.items()},
            **{group + '_postclip_norm': value for group, value in post.items()},
            'shared_upstream_preclip_norm': shared_pre,
            'shared_upstream_postclip_norm': norm(shared)}

def maybe_ema(model, epoch):
    active = epoch >= int(model._delayed_ema_epoch)
    if active:
        model.ema_update()
    return active

def maybe_scheduler(scheduler, completed_updates, updates_per_epoch):
    active = completed_updates % updates_per_epoch == 0
    if active:
        scheduler.step()
    return active
