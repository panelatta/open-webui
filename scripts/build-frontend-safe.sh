#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MAX_OLD_SPACE_MB="${BUILD_MAX_OLD_SPACE_MB:-768}"
SKIP_PYODIDE_FETCH="${SAFE_BUILD_SKIP_PYODIDE_FETCH:-0}"

echo "Starting safer frontend build in $ROOT_DIR"
echo "NODE_OPTIONS=--max-old-space-size=${MAX_OLD_SPACE_MB}"
echo "This entry is intended for low-resource hosts. It uses lower CPU/IO priority."

export NODE_OPTIONS="--max-old-space-size=${MAX_OLD_SPACE_MB} ${NODE_OPTIONS:-}"

run_low_priority() {
	if command -v ionice >/dev/null 2>&1; then
		ionice -c2 -n7 nice -n 15 "$@"
	else
		nice -n 15 "$@"
	fi
}

if [ "$SKIP_PYODIDE_FETCH" != "1" ]; then
	run_low_priority npm run pyodide:fetch
fi

run_low_priority vite build
