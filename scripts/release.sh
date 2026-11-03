#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z ${1:-} ]]; then
    printf 'Usage: %s TAG\n' "$0" >&2
    exit 1
fi

release_tag=$1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

if ! command -v gh >/dev/null; then
    printf 'Install the GitHub CLI (gh) first.\n' >&2
    exit 1
fi

git check-ref-format "refs/tags/$release_tag"
if git show-ref --verify --quiet "refs/tags/$release_tag"; then
    printf 'Tag already exists: %s\n' "$release_tag" >&2
    exit 1
fi
if ! release_branch=$(git symbolic-ref --quiet --short HEAD); then
    printf 'Check out a branch before releasing.\n' >&2
    exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
    printf 'Commit your changes before releasing.\n' >&2
    exit 1
fi

release_remote_head=$(git rev-parse --verify --quiet "refs/remotes/origin/$release_branch" || true)
if [[ -n $release_remote_head ]] && ! git merge-base --is-ancestor "$release_remote_head" HEAD; then
    printf 'The release branch does not contain the known origin commits. Update it before releasing.\n' >&2
    exit 1
fi

python3 - "${release_tag#v}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path("custom_components/glow_assistant/manifest.json")
data = json.loads(manifest.read_text(encoding="utf-8"))
data["version"] = sys.argv[1]
manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
git add -- custom_components/glow_assistant/manifest.json
git commit --amend --no-edit
git tag -a -m "$release_tag" -- "$release_tag"
git push --atomic --force-with-lease="refs/heads/$release_branch:$release_remote_head" origin "HEAD:refs/heads/$release_branch" "refs/tags/$release_tag"
gh release create "$release_tag" --verify-tag --generate-notes --title "$release_tag"
