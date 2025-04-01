#!/bin/bash

# Immediately exit if any command has a non-zero exit status
set -e

APPLY_PATCH_OPTION="-a"
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --no-apply-patches) APPLY_PATCH_OPTION="";;
    *) echo "[ERROR] Unknown parameter passed: $1"; exit 255 ;;
  esac
  shift
done

GIT_ROOT="$(git rev-parse --show-toplevel)"
STYLE_DIR="${GIT_ROOT}/infra/style"

lintrunner --force-color --all-files $APPLY_PATCH_OPTION --config "${GIT_ROOT}/.lintrunner.toml"
