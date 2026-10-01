"""Public integration adapter derived from the frozen formal implementation.

The historical schedule and complete runner are private. Bind a compatible
GraphMAE2 runner explicitly; this module is not a one-command historical replay.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

frozen = None
RUN_ID = "public_sota_weighting_adapter"
PROTOCOL = "PUBLIC_SOTA_WEIGHTING_ADAPTER"
METHODS = ("ALIGNED_MTL", "NASH_MTL", "FAMO")
EXTRA_FIELDS = (
    "coefficient_update_index",
    "coefficient_semantics",
    "shared_parameter_manifest_sha256",
    "applied_reconstruction_multiplier",
    "applied_latent_multiplier",
    "post_state_reconstruction_multiplier",
    "post_state_latent_multiplier",
    "famo_post_state_softmax_json",
    "total_loss_semantics",
    "aligned_rank",
    "aligned_transform_matrix_json",
    "aligned_direct_coefficients_json",
    "nash_solver_status",
    "nash_solver_iterations",
    "famo_post_step_reconstruction_loss",
    "famo_post_step_latent_loss",
    "famo_rng_restored",
)

_original_forward = None
_forward_context: dict[str, Any] = {}


def bind_runner(runner_module: Any) -> None:
    """Bind a user-supplied compatible runner without altering method mathematics."""
    global frozen, _original_forward
    frozen = runner_module
    _original_forward = runner_module.component_forward
    install_adapter()


def shared_encoder_parameters(model: Any) -> tuple[list[Any], dict[str, Any]]:
    entries = [(name, p) for name, p in model.encoder.named_parameters() if p.requires_grad]
    if not entries:
        raise RuntimeError("shared GNN encoder gradient set is empty")
    manifest = [
        {"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
        for name, p in entries
    ]
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return [p for _, p in entries], {"parameters": manifest, "sha256": digest}


def tracked_forward(model: Any, batch: Any, device: Any, modules: dict[str, Any]):
    base = modules["base"]
    _forward_context.clear()
    _forward_context["pre_forward_rng"] = base.capture_rng_states()
    _forward_context["batch"] = batch
    _forward_context["device"] = device
    result = _original_forward(model, batch, device, modules)
    _forward_context["metrics"] = result[3]
    return result


def init_method(method: str, optimizer: Any, device: Any, modules: dict[str, Any]):
    torch = modules["torch"]
    state: dict[str, Any] = {
        "completed_updates": 0,
        "last_applied": (1.0, 1.0),
        "shared_manifest": None,
    }
    if method == "NASH_MTL":
        import cvxpy as cp

        if "ECOS" not in cp.installed_solvers():
            raise RuntimeError("Nash-MTL requires the preregistered ECOS solver")
        state.update(previous_alpha=[1.0, 1.0], solver_steps=0)
    elif method == "FAMO":
        logits = torch.nn.Parameter(torch.zeros(2, device=device))
        state.update(
            logits=logits,
            weight_optimizer=torch.optim.Adam([logits], lr=0.025, weight_decay=0.01),
            min_losses=torch.zeros(2, device=device),
        )
    elif method != "ALIGNED_MTL":
        raise ValueError(method)
    return state


def _nash_alpha(gram: Any, previous: list[float]) -> tuple[list[float], int, str]:
    """Author CVXPY formulation, with solver failure surfaced instead of hidden."""
    import cvxpy as cp
    import numpy as np

    norm = float(np.linalg.norm(gram))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Nash-MTL Gram normalization is nonpositive/nonfinite")
    scaled = np.asarray(gram, dtype=np.float64) / norm
    alpha = cp.Variable(shape=(2,), nonneg=True)
    prior = cp.Parameter(shape=(2,), value=np.asarray(previous, dtype=np.float64))
    matrix = cp.Parameter(shape=(2, 2), value=scaled)
    scale = cp.Parameter(shape=(1,), value=np.asarray([norm]))
    previous_product = matrix @ prior
    if np.any(scaled @ np.asarray(previous) <= 0):
        raise RuntimeError("Nash-MTL previous alpha has nonpositive Gram product")
    phi_tag = 1 / prior + (1 / previous_product) @ matrix
    phi = phi_tag @ (alpha - prior)
    product = matrix @ alpha
    constraints = [
        -cp.log(alpha[i] * scale) - cp.log(product[i]) <= 0
        for i in range(2)
    ]
    problem = cp.Problem(cp.Minimize(cp.sum(product) + phi / scale), constraints)
    current = np.asarray(previous, dtype=np.float64)
    for iteration in range(1, 21):
        prior.value = current
        problem.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or alpha.value is None:
            raise RuntimeError(f"Nash-MTL ECOS failed: {problem.status}")
        candidate = np.asarray(alpha.value, dtype=np.float64)
        if not np.isfinite(candidate).all() or np.any(candidate <= 0):
            raise RuntimeError("Nash-MTL produced invalid alpha")
        residual = float(np.linalg.norm(scaled @ current - 1 / (current + 1e-10)))
        change = float(np.linalg.norm(candidate - current))
        if residual < 1e-3 or change < 1e-6:
            return candidate.tolist(), iteration, str(problem.status)
        current = candidate
    return current.tolist(), 20, str(problem.status)


def _aligned_shared_gradients(rec: Any, lat: Any, model: Any, shared: list[Any], torch: Any):
    """Mirror author amtl: accumulate branch grads; overwrite only shared grads."""
    rec.backward(retain_graph=True)
    rec_parts = [p.grad.detach().clone() if p.grad is not None else None for p in shared]
    for p in shared:
        p.grad = None
    lat.backward()
    lat_parts = [p.grad.detach().clone() if p.grad is not None else None for p in shared]
    if any(a is None or b is None for a, b in zip(rec_parts, lat_parts)):
        raise RuntimeError("Aligned-MTL has missing shared GNN encoder task gradient")
    gradients = torch.stack(
        (torch.cat([g.flatten() for g in rec_parts]),
         torch.cat([g.flatten() for g in lat_parts])), dim=0
    )
    gram = gradients @ gradients.T
    eigenvalues, basis = torch.linalg.eigh(gram)
    tolerance = eigenvalues.max() * 2 * torch.finfo(gram.dtype).eps
    keep = eigenvalues > tolerance
    if not bool(keep.any()) or not bool(torch.isfinite(eigenvalues).all()):
        raise RuntimeError("Aligned-MTL shared gradient matrix has invalid rank")
    eigenvalues = eigenvalues[keep].flip(0)
    basis = basis[:, keep].flip(1)
    scale = eigenvalues[-1].sqrt()
    transform = ((basis * scale) / eigenvalues.sqrt()) @ basis.T
    coefficients = transform @ torch.ones(2, device=gram.device, dtype=gram.dtype)
    merged = gradients.T @ coefficients
    offset = 0
    for p in shared:
        p.grad = merged[offset:offset + p.numel()].view_as(p).detach().clone()
        offset += p.numel()
    if offset != merged.numel():
        raise RuntimeError("Aligned-MTL shared gradient ordering mismatch")
    return int(keep.sum().item()), transform.detach().cpu().tolist(), coefficients.detach().cpu().tolist()


def _famo_applied(rec: Any, lat: Any, state: dict[str, Any], torch: Any):
    losses = torch.stack((rec, lat))
    denominator = losses - state["min_losses"] + 1e-8
    if not bool(torch.isfinite(denominator).all()) or not bool((denominator > 0).all()):
        raise RuntimeError("FAMO nonpositive/nonfinite loss denominator")
    z = torch.softmax(state["logits"], dim=0)
    normalizer = (z / denominator).sum().detach()
    if not bool(torch.isfinite(normalizer)) or float(normalizer) <= 0:
        raise RuntimeError("FAMO invalid normalization")
    objective = (denominator.log() * z / normalizer).sum()
    effective = (z / (normalizer * denominator)).detach()
    return objective, [float(value) for value in effective.tolist()], denominator.detach()


def _famo_poststep_replay(model: Any, state: dict[str, Any], modules: dict[str, Any]):
    torch = modules["torch"]
    base = modules["base"]
    if "batch" not in _forward_context:
        raise RuntimeError("FAMO same-batch forward context missing")
    after_update_rng = base.capture_rng_states()
    after_update_rng_sha = modules["b128"].serialized_sha(after_update_rng)
    try:
        base.restore_rng_states(_forward_context["pre_forward_rng"])
        with torch.no_grad():
            rec, lat, _shared, replay_metrics = _original_forward(
                model, _forward_context["batch"], _forward_context["device"], modules
            )
            for key in ("root_ids_sha256", "mask_indices_sha256", "remask_indices_sha256"):
                if replay_metrics[key] != _forward_context["metrics"][key]:
                    raise RuntimeError(f"FAMO same-batch replay changed {key}")
            post = torch.stack((rec, lat)) - state["min_losses"] + 1e-8
            if not bool(torch.isfinite(post).all()) or not bool((post > 0).all()):
                raise RuntimeError("FAMO post-step loss denominator invalid")
            return post.detach(), float(rec.item()), float(lat.item())
    finally:
        base.restore_rng_states(after_update_rng)
        _forward_context.clear()
        if modules["b128"].serialized_sha(base.capture_rng_states()) != after_update_rng_sha:
            raise RuntimeError("FAMO replay did not restore training RNG state")


def method_step(method: str, rec: Any, lat: Any, shared_representation: Any,
                model: Any, optimizer: Any, state: dict[str, Any], modules: dict[str, Any]):
    torch = modules["torch"]
    t2 = modules["t2"]
    shared, manifest = shared_encoder_parameters(model)
    if state["shared_manifest"] is None:
        state["shared_manifest"] = manifest
    elif state["shared_manifest"]["sha256"] != manifest["sha256"]:
        raise RuntimeError("shared GNN parameter order changed")
    optimizer.zero_grad(set_to_none=True)
    extra: dict[str, Any] = {
        "aligned_rank": None,
        "aligned_transform_matrix_json": None,
        "aligned_direct_coefficients_json": None,
        "nash_solver_status": None,
        "nash_solver_iterations": None,
        "famo_post_step_reconstruction_loss": None,
        "famo_post_step_latent_loss": None,
        "famo_rng_restored": None,
    }
    if method == "ALIGNED_MTL":
        rank, transform, coefficients = _aligned_shared_gradients(rec, lat, model, shared, torch)
        applied = (1.0, 1.0)
        total = rec + lat  # accounting only: the shared gradient is transformed.
        extra.update(
            aligned_rank=rank,
            aligned_transform_matrix_json=json.dumps(transform),
            aligned_direct_coefficients_json=json.dumps(coefficients),
        )
        semantics = "pre_transformation_representation_ratio"
    elif method == "NASH_MTL":
        parts = []
        for loss in (rec, lat):
            gradients = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
            if any(gradient is None for gradient in gradients):
                raise RuntimeError("Nash-MTL has missing shared GNN encoder task gradient")
            parts.append(torch.cat([gradient.detach().flatten() for gradient in gradients]))
        gram = (torch.stack(parts) @ torch.stack(parts).T).double().cpu().numpy()
        alpha, iterations, solver_status = _nash_alpha(gram, state["previous_alpha"])
        applied = tuple(float(value) for value in alpha)
        state["previous_alpha"] = list(applied)
        state["solver_steps"] += 1
        total = applied[0] * rec + applied[1] * lat
        total.backward()
        extra.update(nash_solver_status=solver_status, nash_solver_iterations=iterations)
        semantics = "solver_applied_alpha"
    elif method == "FAMO":
        total, applied_list, previous_denominator = _famo_applied(rec, lat, state, torch)
        applied = tuple(applied_list)
        total.backward()
        semantics = "applied_positive_loss_gradient_multiplier"
    else:
        raise ValueError(method)
    if any(not math.isfinite(value) or value <= 0 for value in applied):
        raise RuntimeError(f"{method} has invalid applied coefficient")
    raw_norm = float(t2.gradient_norm(model.parameters()))
    clip_return = torch.nn.utils.clip_grad_norm_(model.parameters(), frozen.GRAD_CLIP)
    post_clip = float(t2.gradient_norm(model.parameters()))
    optimizer.step()
    if method == "FAMO":
        current_denominator, post_rec, post_lat = _famo_poststep_replay(model, state, modules)
        delta = previous_denominator.log() - current_denominator.log()
        logits = state["logits"]
        gradient = torch.autograd.grad(
            torch.softmax(logits, dim=0), logits, grad_outputs=delta.detach()
        )[0]
        auxiliary_optimizer = state["weight_optimizer"]
        auxiliary_optimizer.zero_grad(set_to_none=True)
        logits.grad = gradient
        auxiliary_optimizer.step()
        extra.update(
            famo_post_step_reconstruction_loss=post_rec,
            famo_post_step_latent_loss=post_lat,
            famo_rng_restored=True,
        )
    state["completed_updates"] += 1
    state["last_applied"] = applied
    if method == "FAMO":
        post_state = (None, None)
        post_softmax = json.dumps(
            [float(x) for x in torch.softmax(state["logits"].detach(), 0).tolist()]
        )
    elif method == "NASH_MTL":
        post_state = applied
        post_softmax = None
    else:
        post_state = (1.0, 1.0)
        post_softmax = None
    if method != "FAMO":
        _forward_context.clear()
    return {
        "reconstruction_weight": applied[0],
        "latent_weight": applied[1],
        "latent_reconstruction_relative_weight": applied[1] / applied[0],
        "weighted_latent_reconstruction_gradient_ratio": None,
        "total_loss": float(total.detach().item()),
        "raw_gradient_norm": raw_norm,
        "clip_grad_norm_returned_preclip": float(clip_return.detach().item()),
        "post_clip_gradient_norm": post_clip,
        "clipping_active": raw_norm > frozen.GRAD_CLIP,
        "extreme_clipping": raw_norm > 10 * frozen.GRAD_CLIP,
        "coefficient_update_index": state["completed_updates"],
        "coefficient_semantics": semantics,
        "shared_parameter_manifest_sha256": manifest["sha256"],
        "applied_reconstruction_multiplier": applied[0],
        "applied_latent_multiplier": applied[1],
        "post_state_reconstruction_multiplier": post_state[0],
        "post_state_latent_multiplier": post_state[1],
        "famo_post_state_softmax_json": post_softmax,
        "total_loss_semantics": (
            "unweighted_accounting_only_shared_gradient_transformed"
            if method == "ALIGNED_MTL" else "differentiated_model_objective"
        ),
        **extra,
    }


def method_state_payload(method: str, state: dict[str, Any], modules: dict[str, Any]):
    payload = {
        "completed_updates": state["completed_updates"],
        "last_applied": list(state["last_applied"]),
        "shared_parameter_manifest": state["shared_manifest"],
    }
    if method == "NASH_MTL":
        payload.update(previous_alpha=state["previous_alpha"], solver_steps=state["solver_steps"])
    elif method == "FAMO":
        payload.update(
            logits=state["logits"].detach().cpu(),
            weight_optimizer=state["weight_optimizer"].state_dict(),
            min_losses=state["min_losses"].detach().cpu(),
        )
    return payload


def effective_method_state(method: str, state: dict[str, Any], modules: dict[str, Any]):
    rec, lat = state["last_applied"]
    return {"reconstruction_weight": rec, "latent_weight": lat}


def install_adapter() -> None:
    if frozen is None:
        raise RuntimeError("bind_runner requires a compatible GraphMAE2 runner")
    frozen.RUN_ID = RUN_ID
    frozen.PROTOCOL = PROTOCOL
    frozen.NEW_METHODS = METHODS
    frozen.TRAIN_FIELDS = tuple(dict.fromkeys((*frozen.TRAIN_FIELDS, *EXTRA_FIELDS)))
    frozen.component_forward = tracked_forward
    frozen.initialize_method_state = init_method
    frozen.method_step = method_step
    frozen.method_state_payload = method_state_payload
    frozen.effective_method_state = effective_method_state
