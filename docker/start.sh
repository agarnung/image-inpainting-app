#!/usr/bin/env bash
# Launcher for the dockerised Image Inpainting app.
# Run from anywhere: ./start.sh (args are forwarded to the app binary).
set -euo pipefail

cd "$(dirname "$0")"

# Allow local Docker containers to reach the X server (no-op on WSLg).
if command -v xhost >/dev/null 2>&1; then
    xhost +local:docker >/dev/null 2>&1 || true
fi

mkdir -p data

DOCKER_BUILDKIT=1 docker compose build
exec docker compose run --rm app "$@"
