"""Gate metrics.

The gate scores exactly two per-image metrics against a shared base reference
set: perceptual distance (LPIPS) and CLIP-embedding distance.

Import safety is part of the contract: importing this module must not load a
model and must not touch the network. LPIPS pulls AlexNet weights and CLIP
pulls a transformer checkpoint; both loaders therefore live inside function
bodies with lazy imports, and only the GPU rendering/scoring path calls them.

API pinned:
    from lorafactory.gate.metrics import METRIC_NAMES, embedding_distance
"""

from __future__ import annotations

import torch

# Order matters: gate/report.py and the report CSV columns follow it.
METRIC_NAMES: tuple[str, str] = ("lpips", "clip_distance")


def embedding_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine distance between two embedding vectors, in [0, 2].

    Zero for identical (non-degenerate) vectors, symmetric, non-negative, and
    monotone in dissimilarity. Pure torch — no model, no network.
    """
    x = torch.as_tensor(a, dtype=torch.float32).reshape(-1)
    y = torch.as_tensor(b, dtype=torch.float32).reshape(-1)
    if x.numel() != y.numel():
        raise ValueError(f"embedding size mismatch: {x.numel()} vs {y.numel()}")

    cos = torch.nn.functional.cosine_similarity(x, y, dim=0, eps=1e-8)
    dist = float(1.0 - cos.item())
    # Guard float noise so identical vectors compare exactly non-negative.
    return max(0.0, dist)


def load_lpips(net: str = "alex", device: str = "cuda"):
    """Load the LPIPS network. Downloads weights — GPU/scoring path only."""
    import lpips as _lpips  # noqa: PLC0415 - lazy on purpose: importing pulls weights

    return _lpips.LPIPS(net=net).to(device)


def load_clip(model_id: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
    """Load the CLIP model + processor. Downloads weights — GPU path only."""
    # noqa on the import: lazy on purpose, importing transformers is heavy.
    from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor
