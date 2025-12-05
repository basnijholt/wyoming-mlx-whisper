#!/bin/bash
set -e
cd "$(dirname "$0")/.."
uv run wyoming-mlx-whisper --uri tcp://0.0.0.0:7891 "$@"
