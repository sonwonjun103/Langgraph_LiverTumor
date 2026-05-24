#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS="${1:-30}"
BRANCH="$(git branch --show-current)"

if [[ -z "${BRANCH}" ]]; then
  echo "Could not detect the current git branch."
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Run this script inside a git repository."
  exit 1
fi

echo "Watching for changes every ${INTERVAL_SECONDS}s on branch '${BRANCH}'."
echo "Press Ctrl+C to stop."

while true; do
  if [[ -n "$(git status --porcelain)" ]]; then
    TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${TIMESTAMP}] Changes detected. Committing and pushing..."

    git add -A

    if git diff --cached --quiet; then
      echo "[${TIMESTAMP}] No staged changes after git add."
    else
      git commit -m "Auto update: ${TIMESTAMP}"
      git pull --rebase --autostash origin "${BRANCH}"
      git push origin "${BRANCH}"
      echo "[${TIMESTAMP}] Synced to origin/${BRANCH}."
    fi
  fi

  sleep "${INTERVAL_SECONDS}"
done
