#!/usr/bin/env bash
set -euo pipefail

echo "=== Running CME 296 Unit Test Suite ==="
python3 -m pytest

ARCHIVE="submission_cme296.tar.gz"
echo "=== Creating submission archive: ${ARCHIVE} ==="
tar -czf "${ARCHIVE}" \
  cme296_assignments \
  tests/adapters.py \
  $( [ -f writeup.pdf ] && echo "writeup.pdf" ) \
  $( [ -f writeup.md ] && echo "writeup.md" )

echo "Created ${ARCHIVE}"
