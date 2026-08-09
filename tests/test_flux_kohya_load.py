"""FLUX load path, real weights: the GPU-only confirmation of the CPU decision
in test_flux_load_smoke.py. Loads a real kohya-trained FLUX LoRA adapter into
the real `black-forest-labs/FLUX.1-dev` pipeline through diffusers' own
`load_lora_weights` (the same `_convert_kohya_flux_lora_to_diffusers` path the
CPU test exercises against a tiny model) and runs one denoise step.

WRITTEN, NOT RUN. Requires a CUDA GPU and ~24 GB of local FLUX.1-dev weights
(README §Requirements) — running it is a standing user decision, not something
this suite does on its own. Collection-safe (`pytest --collect-only`) since it
imports lazily and only touches the network/GPU inside the test body.

Point `LORAFACTORY_TEST_FLUX_ADAPTER` at a real kohya-layout FLUX LoRA
checkpoint (e.g. an `F-F` run's `adapter/<ID>.safetensors`, README's run
layout) before running this by hand:

    LORAFACTORY_TEST_FLUX_ADAPTER=runs/F-F-a1/adapter/F-F.safetensors \\
        uv run pytest -m gpu tests/test_flux_kohya_load.py
"""

import os

import pytest

from test_model_pins import VERIFIED_REVISIONS

FLUX_REPO = "black-forest-labs/FLUX.1-dev"
FLUX_REVISION = VERIFIED_REVISIONS[FLUX_REPO]

ADAPTER_ENV = "LORAFACTORY_TEST_FLUX_ADAPTER"


@pytest.mark.gpu
def test_kohya_flux_lora_loads_and_denoises_one_step():
    adapter_path = os.environ.get(ADAPTER_ENV)
    if not adapter_path:
        pytest.skip(f"set {ADAPTER_ENV} to a real kohya-layout FLUX LoRA checkpoint")

    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        FLUX_REPO, revision=FLUX_REVISION, torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    # Real diffusers load path: `is_kohya` detection + `_convert_kohya_flux_lora_to_diffusers`
    # inside `FluxLoraLoaderMixin.lora_state_dict`, exactly what test_flux_load_smoke.py
    # exercises against a tiny model. No conversion CLI step needed (T4: waiver).
    pipe.load_lora_weights(adapter_path)

    peft_layers = [n for n, _ in pipe.transformer.named_modules() if n.endswith("lora_A")]
    assert peft_layers, "no LoRA layers attached to the real FluxTransformer2DModel"

    generator = torch.Generator(device="cuda").manual_seed(0)
    image = pipe(
        prompt="a photo of sks_lantern on a table",
        num_inference_steps=1,
        guidance_scale=3.5,
        height=512,
        width=512,
        generator=generator,
    ).images[0]

    assert image.size == (512, 512)
