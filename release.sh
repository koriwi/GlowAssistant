#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z ${1:-} ]]; then
    printf 'Usage: %s TAG\n' "$0" >&2
    exit 1
fi

release_tag=$1
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if ! command -v gh >/dev/null; then
    printf 'Install the GitHub CLI (gh) first.\n' >&2
    exit 1
fi

git check-ref-format "refs/tags/$release_tag"
if ! release_branch=$(git symbolic-ref --quiet --short HEAD); then
    printf 'Check out a branch before releasing.\n' >&2
    exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
    printf 'Commit your changes before releasing.\n' >&2
    exit 1
fi

git tag -a -m "$release_tag" -- "$release_tag"
git push --atomic origin "HEAD:refs/heads/$release_branch" "refs/tags/$release_tag"
gh release create "$release_tag" --verify-tag --generate-notes --title "$release_tag"
