SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLUMBER_DIR="${SCRIPT_DIR}/plumber"

# Plumber Config
REPO_URL="${PLUMBER_REPO_URL:-https://github.com/orkg/plumber.git}"
REPO_BRANCH="${PLUMBER_BRANCH:-main}"

echo "Cloning ${REPO_URL} (branch ${REPO_BRANCH})"
git clone "${REPO_BRANCH}" "${REPO_URL}" "${PLUMBER_DIR}"


# Need to Figure out Docker For Now