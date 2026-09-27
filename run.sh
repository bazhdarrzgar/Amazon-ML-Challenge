#!/usr/bin/env bash
# Quick entry point to run the end-to-end pipeline with tqdm progress tracking
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/code/business_entity_resolution/run_all.sh" "$@"
