#!/usr/bin/env bash
# End-to-end reproduction: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv
#
# Usage (from this folder):   bash run_all.sh /path/to/student_resource/dataset /path/to/workdir /path/to/output
# Defaults assume the challenge layout: ../../dataset, ../../work, ../../output
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

# Detect default dataset, work, and output directories
if [ -d "$HERE/../../../dataset" ]; then
    DEFAULT_DATA="$HERE/../../../dataset"
    DEFAULT_WORK="$HERE/../../../work"
    DEFAULT_OUT="$HERE/../../../output"
    DEFAULT_VENV="$HERE/../../../.venv/bin/python"
else
    DEFAULT_DATA="$HERE/../../dataset"
    DEFAULT_WORK="$HERE/../../work"
    DEFAULT_OUT="$HERE/../../output"
    DEFAULT_VENV="$HERE/../../.venv/bin/python"
fi

# Auto-detect Python executable if not explicitly set
if [ -z "${PYTHON:-}" ]; then
    if [ -x "$DEFAULT_VENV" ]; then
        PY="$DEFAULT_VENV"
    elif [ -x "$HERE/../../.venv/bin/python" ]; then
        PY="$HERE/../../.venv/bin/python"
    elif [ -x "$HERE/.venv/bin/python" ]; then
        PY="$HERE/.venv/bin/python"
    elif [ -x "${VIRTUAL_ENV:-}/bin/python" ]; then
        PY="${VIRTUAL_ENV}/bin/python"
    else
        PY="python3"
    fi
else
    PY="$PYTHON"
fi

export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    $PY -m ber.pipeline_runner --help
    exit 0
fi

DATA="${1:-$DEFAULT_DATA}"
WORK="${2:-$DEFAULT_WORK}"
OUT="${3:-$DEFAULT_OUT}"
TAG="${TAG:-final}"

NB="${NAME_BLOCK:---name-block}"   # set NAME_BLOCK="--no-name-block" to disable the name-block channel

# Run end-to-end pipeline with master and granular tqdm progress bars
$PY -m ber.pipeline_runner \
    --data-dir "$DATA" \
    --work-dir "$WORK" \
    --out-dir "$OUT" \
    --tag "$TAG" \
    $NB
