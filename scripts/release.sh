#!/usr/bin/env bash
# Build and check distribution artifacts for a release.
#
# Before running: bump vlmscope/__about__.py, update CHANGELOG.md, then
#   git tag vX.Y.Z && git push --tags
# The release workflow publishes tagged builds; this script is for local checks.
set -euo pipefail

rm -rf dist
python -m build
python -m twine check dist/*

echo
echo "Built artifacts:"
ls -1 dist
echo "Upload manually with: python -m twine upload dist/*"
