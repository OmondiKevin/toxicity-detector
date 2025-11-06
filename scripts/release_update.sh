#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/release_update.sh v1.0.0 "models/*.pth data/processed/*.csv toxicity-detector.zip" "Optional notes to append"
#
# Requires: gh CLI authenticated (gh auth login)

TAG="${1:?Specify tag e.g. v1.0.0}"
FILES="${2:?Specify files/globs to upload}"
BODY="${3:-}"

# Verify gh CLI is installed and authenticated
if ! command -v gh &> /dev/null; then
  echo "Error: GitHub CLI (gh) is not installed. Install from https://cli.github.com/"
  exit 1
fi

# Verify release exists
if ! gh release view "$TAG" &> /dev/null; then
  echo "Error: Release with tag '$TAG' does not exist."
  echo "Verify with: gh release view $TAG"
  exit 1
fi

# Optional: append to release notes
if [[ -n "$BODY" ]]; then
  # Fetch existing notes and append new body
  EXISTING_NOTES=$(gh release view "$TAG" --json body -q .body || echo "")
  if [[ -n "$EXISTING_NOTES" ]]; then
    NEW_NOTES="${EXISTING_NOTES}

${BODY}"
  else
    NEW_NOTES="$BODY"
  fi
  gh release edit "$TAG" --notes "$NEW_NOTES"
fi

# Upload/replace assets. --clobber replaces existing assets with the same filename.
UPLOADED_COUNT=0
for pattern in $FILES; do
  # Use shell glob expansion; if no matches, skip with nullglob behavior
  shopt -s nullglob
  for f in $pattern; do
    if [[ -f "$f" ]]; then
      echo "Uploading $f to release $TAG"
      gh release upload "$TAG" "$f" --clobber
      ((UPLOADED_COUNT++)) || true
    fi
  done
  shopt -u nullglob
done

if [[ $UPLOADED_COUNT -eq 0 ]]; then
  echo "Error: No files matched the provided patterns: $FILES"
  echo "Verify file paths and globs. To update notes only, use: gh release edit $TAG --notes \"<notes>\""
  exit 1
else
  echo "Successfully uploaded $UPLOADED_COUNT file(s) to release $TAG"
fi

