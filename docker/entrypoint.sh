#!/usr/bin/env bash
# Entrypoint for the lorafactory toolchain image.
#
# Activates the tool venv (/opt/venv-tool) that lorafactory itself is
# installed into, then execs the lorafactory CLI with whatever arguments the
# container was invoked with. The training venv (/opt/venv-kohya) and the
# sd-scripts checkout (/opt/sd-scripts) are reached indirectly via
# LORAFACTORY_KOHYA_PYTHON / LORAFACTORY_SDSCRIPTS_DIR, consumed by
# lorafactory.kohya.runner — this entrypoint never invokes kohya directly.
set -euo pipefail

export PATH="/opt/venv-tool/bin:${PATH}"
export LORAFACTORY_KOHYA_PYTHON="${LORAFACTORY_KOHYA_PYTHON:-/opt/venv-kohya/bin/python}"
export LORAFACTORY_SDSCRIPTS_DIR="${LORAFACTORY_SDSCRIPTS_DIR:-/opt/sd-scripts}"

exec lorafactory "$@"
