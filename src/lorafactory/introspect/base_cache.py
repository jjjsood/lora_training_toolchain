"""Base-model top-k subspaces, computed once and cached on disk.

API pinned:
    from lorafactory.introspect.base_cache import BaseSubspaceCache, load_base_weight
    cache = BaseSubspaceCache(cache_dir, model_revision, base_checkpoint=path)
    u0 = cache.top_k_subspace("transformer.transformer_blocks.0.attn.to_q", k=10)
    # -> (out_dim, k) orthonormal columns, or None when this module has no
    #    pinned base key / no base checkpoint was supplied.

The intruder statistic needs the *base* weight's dominant left subspace, which
costs one SVD of a 1536x1536-ish matrix per module. Doing that on every run of
`lorafactory introspect` would dominate the runtime, so each subspace is written
to `{cache_dir}/.cache/introspect/{revision}/{module}.k{k}.safetensors` and read
back on the next run. The revision is part of the path because a different base
checkpoint has a different subspace: a stale cache would silently answer for the
wrong model.

Base-key mapping — SD3 first, FLUX deferred
-------------------------------------------
An adapter module name says nothing about where its base weight lives; that is a
per-architecture convention. `SD3_BASE_KEY_MAP` pins the mapping for the whole
SD3 vocabulary (both the diffusers and the kohya spelling of each module) onto
the Stability single-file layout, `model.diffusion_model.joint_blocks.{N}....`.

The SAI checkpoint keeps q/k/v fused in one `attn.qkv.weight` of shape
(3*d, d), stacked q, k, v along the output dimension. A diffusers-side module
such as `attn.to_q` therefore maps to a *slice* of that tensor, which is what
`BaseKeyRef.part`/`parts` encode. The kohya spelling keeps the fused module
whole (`..._attn_qkv`), so it maps to the whole tensor with `parts=1`.

This dict is written from the documented SD3 single-file layout, not read off a
downloaded checkpoint — nothing here downloads anything. `load_base_weight`
returns None for a key the given checkpoint does not contain, so a mismatch
degrades to "no intruder columns" rather than to a wrong number.

FLUX is deliberately out of scope: BFL's fused `double_blocks.N.img_attn.qkv.
weight` needs its own slice convention, and shipping it half-checked would put
unverifiable numbers in the thesis. FLUX checkpoints still get frob_norm,
effective_rank and top_sigma; their intruder columns come out empty.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from lorafactory.constants import SD3_BLOCK23_ABSENT, SD3_NUM_BLOCKS

#: Tensor name used inside a cached subspace file.
SUBSPACE_TENSOR_KEY = "u_top"

#: Where the SD3 single-file checkpoint keeps its MM-DiT blocks.
SD3_SINGLE_FILE_PREFIX = "model.diffusion_model.joint_blocks"


@dataclass(frozen=True)
class BaseKeyRef:
    """A base-checkpoint tensor, optionally one slice of a fused projection.

    ``part``/``parts`` select rows [part*d : (part+1)*d] of a (parts*d, in_dim)
    fused weight; ``parts == 1`` means "the whole tensor".
    """

    key: str
    part: int = 0
    parts: int = 1


# leaf (diffusers spelling) -> (suffix under the block, slice index, slice count)
_SD3_DIFFUSERS_LEAF = {
    "attn.to_q": ("x_block.attn.qkv.weight", 0, 3),
    "attn.to_k": ("x_block.attn.qkv.weight", 1, 3),
    "attn.to_v": ("x_block.attn.qkv.weight", 2, 3),
    "attn.to_out.0": ("x_block.attn.proj.weight", 0, 1),
    "attn.add_q_proj": ("context_block.attn.qkv.weight", 0, 3),
    "attn.add_k_proj": ("context_block.attn.qkv.weight", 1, 3),
    "attn.add_v_proj": ("context_block.attn.qkv.weight", 2, 3),
    "attn.to_add_out": ("context_block.attn.proj.weight", 0, 1),
    "ff.net.0.proj": ("x_block.mlp.fc1.weight", 0, 1),
    "ff.net.2": ("x_block.mlp.fc2.weight", 0, 1),
    "ff_context.net.0.proj": ("context_block.mlp.fc1.weight", 0, 1),
    "ff_context.net.2": ("context_block.mlp.fc2.weight", 0, 1),
}

# leaf (kohya spelling) -> suffix under the block; kohya never splits qkv, so
# every one of these takes the whole tensor.
_SD3_KOHYA_LEAF = {
    "attn_qkv": "attn.qkv.weight",
    "attn_proj": "attn.proj.weight",
    "mlp_fc1": "mlp.fc1.weight",
    "mlp_fc2": "mlp.fc2.weight",
}

# Context-stream leaves that the context-pre-only last block does not have.
_KOHYA_CONTEXT_ABSENT = ("attn_proj", "mlp_fc1", "mlp_fc2")


def _build_sd3_base_key_map() -> dict[str, BaseKeyRef]:
    mapping: dict[str, BaseKeyRef] = {}
    for block in range(SD3_NUM_BLOCKS):
        block_prefix = f"{SD3_SINGLE_FILE_PREFIX}.{block}"
        last = block == SD3_NUM_BLOCKS - 1

        for leaf, (suffix, part, parts) in _SD3_DIFFUSERS_LEAF.items():
            if last and leaf in SD3_BLOCK23_ABSENT:
                continue
            mapping[f"transformer.transformer_blocks.{block}.{leaf}"] = BaseKeyRef(
                key=f"{block_prefix}.{suffix}", part=part, parts=parts
            )

        for stream in ("x", "context"):
            for leaf, suffix in _SD3_KOHYA_LEAF.items():
                if last and stream == "context" and leaf in _KOHYA_CONTEXT_ABSENT:
                    continue
                name = f"lora_unet_joint_blocks_{block}_{stream}_block_{leaf}"
                mapping[name] = BaseKeyRef(key=f"{block_prefix}.{stream}_block.{suffix}")
    return mapping


#: Adapter module name (diffusers *and* kohya spelling) -> base checkpoint tensor.
SD3_BASE_KEY_MAP: dict[str, BaseKeyRef] = _build_sd3_base_key_map()


def base_key_for(module_name: str) -> BaseKeyRef | None:
    """The pinned base tensor for an adapter module, or None if unmapped."""
    return SD3_BASE_KEY_MAP.get(module_name)


def load_base_weight(checkpoint: Path | str, ref: BaseKeyRef | str) -> torch.Tensor | None:
    """Stream one base weight (or fused slice) off a safetensors checkpoint.

    Uses `safe_open` so only the requested tensor is ever read — a 4 GB base
    checkpoint costs one 1536x1536 matrix here, on CPU. Returns None when the
    checkpoint has no such key, which is how an architecture mismatch stays a
    missing column instead of a crash.
    """
    if isinstance(ref, str):
        ref = BaseKeyRef(key=ref)
    with safe_open(str(checkpoint), framework="pt") as f:
        if ref.key not in set(f.keys()):
            return None
        weight = f.get_tensor(ref.key)
    weight = weight.float()
    if ref.parts <= 1:
        return weight
    rows = weight.shape[0] // ref.parts
    return weight[ref.part * rows : (ref.part + 1) * rows, :]


def top_left_subspace(weight: torch.Tensor, k: int) -> torch.Tensor:
    """The k dominant left singular vectors of ``weight``, as columns."""
    u, _, _ = torch.linalg.svd(weight.float(), full_matrices=False)
    return u[:, :k].contiguous()


class StaleSubspaceCacheError(ValueError):
    """The cache under this revision was built from a different base file."""


#: Marker recording which base file a revision's cached subspaces came from.
BASE_MARKER_FILENAME = "base.json"


def base_fingerprint(checkpoint: Path | str) -> str:
    """A cheap identity for a base checkpoint: resolved path, size, mtime.

    Not a content hash — hashing a multi-GB base checkpoint on every run is
    exactly the cost this cache exists to avoid. It is enough to tell two
    different files apart, which is all the staleness guard needs.
    """
    path = Path(checkpoint).resolve()
    stat = path.stat()
    payload = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class BaseSubspaceCache:
    """Disk-cached top-k left subspaces of the base model's weights.

    ``model_revision`` names the base checkpoint the subspaces belong to and is
    part of the cache path, so two revisions never share entries.
    ``base_checkpoint`` may be omitted to run purely off an already-populated
    cache (the offline case: the subspaces were computed on the machine that
    has the weights, and the cache directory travelled with the results).

    Staleness guard. The revision string is a *claim* about which weights these
    subspaces describe, and a wrong claim is worse than no cache: it yields a
    plausible intruder count computed against the wrong model. So the first
    write under a revision drops a `base.json` marker recording the base file's
    fingerprint, and every later use with a base checkpoint checks it. A second
    run pointing `--base` at a different file under the same revision raises
    `StaleSubspaceCacheError` instead of silently reusing the first one's
    subspaces. `lorafactory introspect` additionally refuses `--base` without an
    explicit `--base-revision`, so the revision is never a placeholder.
    """

    def __init__(
        self,
        cache_dir: Path | str,
        model_revision: str,
        base_checkpoint: Path | str | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.model_revision = str(model_revision)
        self.base_checkpoint = Path(base_checkpoint) if base_checkpoint is not None else None
        self.hits = 0
        self.misses = 0
        self._identity_checked = False

    @property
    def root(self) -> Path:
        return self.cache_dir / ".cache" / "introspect" / self.model_revision

    @property
    def marker_path(self) -> Path:
        return self.root / BASE_MARKER_FILENAME

    def subspace_path(self, module_name: str, k: int) -> Path:
        """`{cache_dir}/.cache/introspect/{revision}/{module}.k{k}.safetensors`."""
        safe_name = module_name.replace("/", "_")
        return self.root / f"{safe_name}.k{k}.safetensors"

    def check_base_identity(self) -> None:
        """Bind this revision's cache to one base file, or refuse to use it.

        A no-op when no base checkpoint was supplied: a cache-only reader has
        nothing to contradict, and the recorded marker is what the offline
        results carry as provenance.
        """
        if self._identity_checked or self.base_checkpoint is None:
            return

        fingerprint = base_fingerprint(self.base_checkpoint)
        marker = self.marker_path
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded.get("fingerprint") != fingerprint:
                raise StaleSubspaceCacheError(
                    f"cache {self.root} was built from {recorded.get('checkpoint')!r} "
                    f"(fingerprint {recorded.get('fingerprint')}), but --base is "
                    f"{str(self.base_checkpoint)!r} (fingerprint {fingerprint}). "
                    "Use a different --base-revision or clear the cache directory."
                )
        else:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                json.dumps(
                    {
                        "checkpoint": str(Path(self.base_checkpoint).resolve()),
                        "fingerprint": fingerprint,
                        "revision": self.model_revision,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        self._identity_checked = True

    def top_k_subspace(self, module_name: str, k: int) -> torch.Tensor | None:
        """Top-k base subspace for one adapter module, computing it at most once.

        Returns None when the module has no pinned base key (e.g. FLUX, text
        encoders), when no base checkpoint is available to compute from, or
        when the base checkpoint does not carry the mapped tensor.
        """
        self.check_base_identity()
        path = self.subspace_path(module_name, k)
        if path.exists():
            self.hits += 1
            with safe_open(str(path), framework="pt") as f:
                return f.get_tensor(SUBSPACE_TENSOR_KEY)

        ref = base_key_for(module_name)
        if ref is None or self.base_checkpoint is None:
            return None
        weight = load_base_weight(self.base_checkpoint, ref)
        if weight is None:
            return None

        subspace = top_left_subspace(weight, k)
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({SUBSPACE_TENSOR_KEY: subspace}, str(path))
        return subspace
