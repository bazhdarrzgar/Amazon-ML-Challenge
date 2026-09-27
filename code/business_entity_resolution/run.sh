#!/usr/bin/env bash
# Wrapper to execute run_all.sh with tqdm progress tracking
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/run_all.sh" "$@"
