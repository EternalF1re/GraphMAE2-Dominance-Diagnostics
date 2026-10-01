"""Public direct target-representation diagnostic entry point."""

from __future__ import annotations

import math
from typing import Any

from diagnostics.component_gradients import measure_component_gradients


def measure_with_applied_weights(
    model: Any, batch: Any, reconstruction_weight: float, latent_weight: float, device: Any
):
    """Use the completed update's applied weights, not post-step auxiliary state."""
    if not all(math.isfinite(value) and value > 0 for value in (reconstruction_weight, latent_weight)):
        raise ValueError("applied weights must be finite and positive")
    return measure_component_gradients(model, batch, latent_weight / reconstruction_weight, device)
