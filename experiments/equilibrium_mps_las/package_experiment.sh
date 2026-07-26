#!/usr/bin/env bash
# Create a transferable bundle containing only the isolated experiment.

set -euo pipefail

# Prevent macOS tar from adding AppleDouble `._*` metadata entries.
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION="${1:-$SCRIPT_DIR/dist}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="equilibrium_mps_las_${STAMP}"
STAGING="$(mktemp -d)"

trap 'rm -rf "$STAGING"' EXIT
FLAT="$STAGING/$NAME"
mkdir -p "$DESTINATION" "$FLAT"
for pattern in "*.py" "*.sh" "*.md"; do
    for source in "$SCRIPT_DIR"/$pattern; do
        [[ -f "$source" ]] && cp "$source" "$FLAT/"
    done
done
if [[ -d "$SCRIPT_DIR/tests" ]]; then
    mkdir -p "$FLAT/tests"
    find "$SCRIPT_DIR/tests" -type f ! -name '*.pyc' -exec cp {} "$FLAT/tests/" \;
fi
(
    cd "$FLAT"
    find . -type f ! -name SHA256SUMS -print0 \
        | sort -z \
        | xargs -0 shasum -a 256 \
        > SHA256SUMS
)

ARCHIVE="$DESTINATION/$NAME.tar.gz"
tar -C "$STAGING" -czf "$ARCHIVE" "$NAME"
echo "$ARCHIVE"
