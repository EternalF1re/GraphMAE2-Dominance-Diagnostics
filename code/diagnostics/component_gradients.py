"""Component-gradient dominance measurement on a compatible GraphMAE2 batch.

This module contains no optimizer step and does not update model parameters. It
expects the GraphMAE2 model and LC batch semantics included under
``code/graphmae2``. Checkpoints and LC artifacts are intentionally not shipped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DominanceMeasurement:
    reconstruction_gradient_norm: float
    raw_latent_gradient_norm: float
    weighted_latent_gradient_norm: float
    dominance_ratio: float
    gradient_cosine: float
    target_root_count: int
    sampled_subgraph_node_count: int
    sampled_subgraph_edge_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unpack_batch(batch: Any, device: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    graph, targets, labels, node_ids = batch[:4]
    drop_graph_a = batch[4] if len(batch) > 4 else None
    drop_graph_b = batch[5] if len(batch) > 5 else None
    graph = graph.to(device)
    targets = targets.to(device)
    node_ids = node_ids.to(device)
    features = graph.ndata["feat"].to(device)
    if drop_graph_a is not None:
        drop_graph_a = drop_graph_a.to(device)
    if drop_graph_b is not None:
        drop_graph_b = drop_graph_b.to(device)
    return graph, targets, node_ids, drop_graph_a, drop_graph_b, features


def measure_component_gradients(
    model: Any,
    batch: Any,
    latent_weight: float,
    device: Any,
) -> DominanceMeasurement:
    """Measure both loss-component gradients on the same target representation.

    The function mirrors the reported shared-target diagnostic: both component
    gradients are differentiated with respect to the online encoder output and
    then restricted to ``targets`` before their Euclidean norms are compared.
    It performs forward and autograd operations, but never calls backward on
    parameters, never calls an optimizer, and never mutates a checkpoint.
    """

    import torch

    graph, targets, _node_ids, drop_a, drop_b, features = _unpack_batch(batch, device)
    online_graph = drop_a if drop_a is not None else graph
    teacher_graph = drop_b if drop_b is not None else graph
    decoder_graph, masked_features, (mask_nodes, _keep_nodes) = model.encoding_mask_noise(
        graph, features, model._mask_rate
    )
    representation = model.encoder(online_graph, masked_features)
    with torch.no_grad():
        teacher_representation = model.encoder_ema(teacher_graph, features)
        latent_target = model.projector_ema(teacher_representation[targets]).detach()
    latent_prediction = model.predictor(model.projector(representation[targets]))

    from models.loss_func import sce_loss

    latent_loss = sce_loss(latent_prediction, latent_target, 1)
    decoder_representation = model.encoder_to_decoder(representation)
    reconstruction_losses = []
    for _ in range(int(model._num_remasking)):
        remasked, _nodes, _ = model.random_remask(
            online_graph, decoder_representation.clone(), model._remask_rate
        )
        reconstruction = model.decoder(decoder_graph, remasked)
        reconstruction_losses.append(model.criterion(features[mask_nodes], reconstruction[mask_nodes]))
    reconstruction_loss = sum(reconstruction_losses)

    reconstruction_gradient = torch.autograd.grad(
        reconstruction_loss, representation, retain_graph=True
    )[0][targets]
    raw_latent_gradient = torch.autograd.grad(
        latent_loss, representation, retain_graph=True
    )[0][targets]
    weighted_latent_gradient = raw_latent_gradient * float(latent_weight)

    rec_norm = float(torch.linalg.vector_norm(reconstruction_gradient.double()).item())
    raw_norm = float(torch.linalg.vector_norm(raw_latent_gradient.double()).item())
    weighted_norm = float(torch.linalg.vector_norm(weighted_latent_gradient.double()).item())
    denominator = rec_norm * weighted_norm
    cosine = (
        float(
            torch.sum(reconstruction_gradient.double() * weighted_latent_gradient.double()).item()
            / denominator
        )
        if denominator
        else 0.0
    )
    return DominanceMeasurement(
        reconstruction_gradient_norm=rec_norm,
        raw_latent_gradient_norm=raw_norm,
        weighted_latent_gradient_norm=weighted_norm,
        dominance_ratio=weighted_norm / max(rec_norm, 1e-30),
        gradient_cosine=cosine,
        target_root_count=int(targets.numel()),
        sampled_subgraph_node_count=int(graph.num_nodes()),
        sampled_subgraph_edge_count=int(graph.num_edges()),
    )
