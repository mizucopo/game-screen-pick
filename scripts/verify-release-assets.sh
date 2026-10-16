#!/bin/sh
# Product provenance/evidence guard around the unchanged common archive reader.
set -eu
exec "${PYTHON:-python3}" -I "$(dirname "$0")/distribution.py" verify "$@"
