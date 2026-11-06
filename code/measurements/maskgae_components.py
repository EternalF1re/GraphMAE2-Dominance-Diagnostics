"""Release-authored full-shared-Z component math; no model or Torch imports.

Definition: ||d(alpha*L_degree)/dZ|| / ||dL_edge/dZ||, same forward,
all node rows, before clipping. Not a parameter-gradient or target-row ratio.
Scalar fixtures here use Python floating point; recorded measurements use
FP32 autograd followed by FP64 norm accumulation and are not recomputed here.
"""
import math


def component_norms(edge_gradient, raw_degree_gradient, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Finite nonnegative alpha required")
    if not edge_gradient or len(edge_gradient) != len(raw_degree_gradient):
        raise ValueError("Matching full-Z rows required")
    columns = len(edge_gradient[0])
    if not columns or any(len(a) != columns or len(b) != columns for a, b in zip(edge_gradient, raw_degree_gradient)):
        raise ValueError("Matching rectangular full-Z gradients required")
    edge = [v for row in edge_gradient for v in row]
    degree = [v for row in raw_degree_gradient for v in row]
    if any(not math.isfinite(v) for v in edge + degree):
        raise ValueError("Finite full-Z components required")
    rec = math.hypot(*edge)
    weighted = math.hypot(*(alpha * v for v in degree))
    return dict(rec_Z_norm=rec, weighted_degree_Z_norm=weighted,
                R=weighted / rec if rec > 1e-12 else None,
                status="DEFINED" if rec > 1e-12 else "UNDEFINED")


def same_forward_vjps(z, edge_loss, degree_loss, alpha, autograd_grad):
    """Inspectable two-VJP boundary, with the caller supplying autograd.grad.

    Does not call backward(), clip gradients, construct a model or step Adam.
    The caller owns isolation of model/BN/gradient/RNG state around its forward.
    """
    if not z.requires_grad or len(z.shape) != 2 or not math.isfinite(alpha) or alpha < 0:
        raise ValueError("Differentiable full shared Z and finite alpha required")
    edge, = autograd_grad(edge_loss, (z,), retain_graph=True)
    weighted, = autograd_grad(alpha * degree_loss, (z,), retain_graph=False)
    if edge.shape != z.shape or weighted.shape != z.shape:
        raise ValueError("Both VJPs must cover full Z")
    return edge, weighted
