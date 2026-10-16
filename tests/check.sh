#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo test --locked --no-fail-fast
sh tests/licenses.sh
sh tests/distribution.sh
