#!/bin/sh
# MeshCompute installer: create a venv and install the `mesh` CLI + all
# Phase-1 packages. Idempotent (safe to re-run), no sudo, no network access
# beyond package installation (pip/uv talking to PyPI).
#
# Usage:
#   ./scripts/install.sh
#   mesh node start
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
REQUIRED_MAJOR=3
REQUIRED_MINOR=12

# --- find a python3.12+ interpreter -----------------------------------------
PYTHON=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo "0.0")
        major=$(printf '%s' "$version" | cut -d. -f1)
        minor=$(printf '%s' "$version" | cut -d. -f2)
        if [ "$major" -gt "$REQUIRED_MAJOR" ] || { [ "$major" -eq "$REQUIRED_MAJOR" ] && [ "$minor" -ge "$REQUIRED_MINOR" ]; }; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "error: need python${REQUIRED_MAJOR}.${REQUIRED_MINOR}+ on PATH (found none that qualifies)." >&2
    echo "       install Python ${REQUIRED_MAJOR}.${REQUIRED_MINOR} and re-run this script." >&2
    exit 1
fi
echo "using $($PYTHON --version) ($PYTHON)"

# --- create the venv (idempotent: skip if it already looks valid) ----------
if [ -x "$VENV_DIR/bin/python" ]; then
    echo "venv already exists at $VENV_DIR — reusing it"
else
    echo "creating venv at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

# --- install the package (editable) -----------------------------------------
# Prefer uv if the user already has it (faster); fall back to the venv's own
# pip otherwise. Either way, nothing is installed outside the venv.
if command -v uv >/dev/null 2>&1; then
    echo "installing with uv..."
    uv pip install --python "$VENV_PY" -e "$REPO_ROOT"
else
    echo "installing with pip..."
    "$VENV_PY" -m pip install --upgrade pip >/dev/null
    "$VENV_PY" -m pip install -e "$REPO_ROOT"
fi

echo
echo "done. activate the venv and start a node:"
echo
echo "    source $VENV_DIR/bin/activate"
echo "    mesh node start"
echo
echo "(or run it without activating: $VENV_DIR/bin/mesh node start)"
echo
echo "see docs/QUICKSTART.md for contribution flags and the full-mesh flow."
