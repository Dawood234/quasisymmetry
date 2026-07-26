#!/usr/bin/env bash
# Package local parent DMRG artifacts without duplicating experiment source.

set -euo pipefail

# Prevent macOS tar from adding AppleDouble `._*` metadata entries.
export COPYFILE_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SCRIPT_DIR/parent_references"
DESTINATION="${1:-$SCRIPT_DIR/dist}"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="equilibrium_mps_parent_references_${STAMP}"
STAGING="$(mktemp -d)"

trap 'rm -rf "$STAGING"' EXIT

if [[ ! -d "$SOURCE/h2o/job_50658575" ]]; then
    echo "Missing H2O parent references: $SOURCE/h2o/job_50658575" >&2
    exit 2
fi
if [[ ! -d "$SOURCE/n2/job_50805667" ]]; then
    echo "Missing N2 parent references: $SOURCE/n2/job_50805667" >&2
    exit 2
fi

FLAT="$STAGING/$NAME"
mkdir -p "$DESTINATION" "$FLAT/parent_references"
while IFS= read -r -d '' source; do
    relative="${source#"$SOURCE/"}"
    mkdir -p "$FLAT/parent_references/$(dirname "$relative")"
    cp "$source" "$FLAT/parent_references/$relative"
done < <(find "$SOURCE" -type f ! -name '.DS_Store' -print0)

(
    cd "$FLAT"
    find parent_references -type f -print0 \
        | sort -z \
        | xargs -0 shasum -a 256 \
        > SHA256SUMS
)

ARCHIVE="$DESTINATION/$NAME.tar.gz"
tar -C "$STAGING" -czf "$ARCHIVE" "$NAME"
echo "$ARCHIVE"
