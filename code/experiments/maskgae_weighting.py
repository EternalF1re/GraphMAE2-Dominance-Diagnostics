"""Release-authored inspection math, not a training runner.

Corresponds to MaskGAE's train_step at upstream commit
343ced7028681f052b7e9ea47945be7fc5166213 (EdisonLeeeee/MaskGAE).
No upstream source is vendored. Losses and gradients must come from the same
forward. Alpha applies to ALL degree-objective parameter gradients, including
the shared encoder and degree head; there is no encoder-only gradient hook.
"""
import math


def joint_loss(edge_loss, degree_loss, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Finite nonnegative joint alpha required")
    # A fresh scalar avoids mutating the separately observed edge objective.
    return edge_loss + alpha * degree_loss


def joint_gradients(edge_gradients, degree_gradients, alpha):
    if len(edge_gradients) != len(degree_gradients):
        raise ValueError("Matching complete parameter coordinates required")
    values = [joint_loss(edge, degree, alpha) for edge, degree in zip(edge_gradients, degree_gradients)]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("Finite joint gradient required")
    return values


def clip_triggered(preclip_global_norm):
    if not math.isfinite(preclip_global_norm) or preclip_global_norm < 0:
        raise ValueError("Finite nonnegative norm required")
    return preclip_global_norm > 1.


def clip_coefficient(preclip_global_norm):
    clip_triggered(preclip_global_norm)
    return min(1., 1. / (preclip_global_norm + 1e-6))
