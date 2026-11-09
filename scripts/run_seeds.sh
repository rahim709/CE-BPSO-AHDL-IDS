#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_seeds.sh configs/gas_pipeline.yaml results/comparison/gas
python -m src.cli compare --config "${1:-configs/gas_pipeline.yaml}" --output "${2:-results/comparison/gas}"
