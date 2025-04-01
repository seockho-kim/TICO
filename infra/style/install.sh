#!/bin/bash

GIT_ROOT="$(git rev-parse --show-toplevel)"
STYLE_DIR="${GIT_ROOT}/infra/style"

REQ_PKG="requirements.txt"
python3 -m pip install -r "${STYLE_DIR}/${REQ_PKG}"

lintrunner init --config ${GIT_ROOT}/.lintrunner.toml
