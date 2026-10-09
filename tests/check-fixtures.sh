#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
fixture_build=$(mktemp -d)
trap 'rm -rf "$fixture_build"' EXIT HUP INT TERM
rustfmt --edition 2024 --check tests/fixture_inputs.rs
rustc --edition=2024 --deny warnings --test tests/fixture_inputs.rs -o "$fixture_build/check-fixtures"
"$fixture_build/check-fixtures"
