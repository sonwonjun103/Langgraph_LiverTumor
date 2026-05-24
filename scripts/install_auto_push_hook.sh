#!/usr/bin/env bash
set -euo pipefail

HOOK_PATH=".git/hooks/post-commit"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Run this script inside a git repository."
  exit 1
fi

cat > "${HOOK_PATH}" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail

BRANCH="$(git branch --show-current)"
if [[ -z "${BRANCH}" ]]; then
  exit 0
fi

git push origin "${BRANCH}"
HOOK

chmod +x "${HOOK_PATH}"
echo "Installed post-commit auto-push hook at ${HOOK_PATH}."
