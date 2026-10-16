#!/bin/sh
# Local/PR entry point to the unchanged shared Rust builder. Never publishes.
set -eu
cd "$(dirname "$0")/.."
target=$1
case "$target" in
  aarch64-apple-darwin) export MACOSX_DEPLOYMENT_TARGET=14.0 ;;
  *-pc-windows-msvc) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+crt-static" ;;
esac
exec "${PYTHON:-python3}" -I scripts/distribution.py build "$target" "${2:-dist}"
