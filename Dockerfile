# lorafactory toolchain image.
#
# Static definition only — the test suite never builds this image (it is
# CPU-only, with no GPU and no model downloads). Build it by hand:
#   docker build -t lorafactory .
#
# Two Python environments are provisioned:
#   /opt/venv-tool   - lorafactory itself (this repo, `uv sync --frozen`)
#   /opt/venv-kohya  - kohya-ss/sd-scripts training deps
#                       (docker/kohya-requirements.lock.txt)
# plus the sd-scripts source checkout at /opt/sd-scripts, pinned to v0.11.1
# and asserted against that exact commit so a moved tag fails the build
# loudly instead of silently training against different code.

# CUDA 12.8, not 12.4: the target GPU is Blackwell (sm_120), which needs
# CUDA 12.8+ runtime libraries to match the cu128 torch wheels pinned in
# docker/kohya-requirements.lock.txt. 05_docker_static.sh asserts the two
# stay in step, because a mismatch here only surfaces as a kernel launch
# failure on the GPU, long after the image builds cleanly.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG SDSCRIPTS_REF=v0.11.1
ARG SDSCRIPTS_COMMIT=6721028c79ee85a78b3a06dfd8954dae310a1cce

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    # uv's default HTTP timeout is 30s per request. The CUDA wheels torch
    # pulls in are far too large for that (nvidia-cudnn-cu12 alone is ~693 MB),
    # so the build fails at a random wheel roughly whenever the mirror is slow
    # — "Failed to download distribution due to network timeout". Raising it
    # makes the build reproducible rather than luck-dependent.
    UV_HTTP_TIMEOUT=600

# --- OS packages + Python 3.11 ------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
        ca-certificates \
        git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        # opencv-python (pinned by sd-scripts v0.11.1) links against libGL and
        # glib. sd-scripts' library.train_util imports cv2 at module scope, so
        # without these every training run dies on `import cv2` before it ever
        # reaches the GPU.
        libgl1 \
        libglib2.0-0 \
        # triton (pulled in by the cu128 torch wheels) compiles its launcher
        # stub with a host C compiler at runtime; without one, sd-scripts dies
        # on `import diffusers` with "Failed to find C compiler".
        gcc \
    && rm -rf /var/lib/apt/lists/*

# --- uv -------------------------------------------------------------------
# Pinned, not :latest — an unpinned build tool silently changes resolution
# behaviour between two builds of the same "pinned" image.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

# --- kohya-ss/sd-scripts, pinned to v0.11.1 --------------------------------
RUN git clone --branch "${SDSCRIPTS_REF}" --depth 1 \
        https://github.com/kohya-ss/sd-scripts.git /opt/sd-scripts \
    && cd /opt/sd-scripts \
    && actual="$(git rev-parse HEAD)" \
    && test "${actual}" = "${SDSCRIPTS_COMMIT}" || \
        (echo "sd-scripts v0.11.1 tag resolved to unexpected commit ${actual}, expected ${SDSCRIPTS_COMMIT}" >&2; exit 1)

# --- /opt/venv-kohya: sd-scripts training environment ----------------------
COPY docker/kohya-requirements.lock.txt /tmp/kohya-requirements.lock.txt
RUN uv venv --python 3.11 /opt/venv-kohya \
    && uv pip install --python /opt/venv-kohya/bin/python \
        -r /tmp/kohya-requirements.lock.txt \
    && uv pip install --python /opt/venv-kohya/bin/python -e /opt/sd-scripts

# --- /opt/venv-tool: lorafactory itself --------------------------------
WORKDIR /opt/lorafactory
COPY LICENSE README.md pyproject.toml uv.lock ./
COPY src ./src
RUN uv venv --python 3.11 /opt/venv-tool \
    && UV_PROJECT_ENVIRONMENT=/opt/venv-tool uv sync --frozen --no-dev

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Path roots, matching docker-compose.yaml's mount points. RUNS_DIR in
# particular is not optional: without it the CLI falls back to `cwd/runs`,
# which is WORKDIR (/opt/lorafactory) — so a finished training run would write
# its adapter inside the container and vanish with `docker run --rm`.
ENV LORAFACTORY_KOHYA_PYTHON=/opt/venv-kohya/bin/python \
    LORAFACTORY_SDSCRIPTS_DIR=/opt/sd-scripts \
    LORAFACTORY_MODELS_DIR=/workspace/models \
    LORAFACTORY_DATASETS_DIR=/workspace/datasets \
    LORAFACTORY_RUNS_DIR=/workspace/runs

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
