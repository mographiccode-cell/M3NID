#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python desktop_app/app.py "$@"
