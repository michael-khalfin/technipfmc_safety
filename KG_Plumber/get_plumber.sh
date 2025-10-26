SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLUMBER_DIR="${SCRIPT_DIR}/plumber"

# Plumber Config
REPO_URL="${PLUMBER_REPO_URL:-https://github.com/YaserJaradeh/ThePlumber}"
REPO_BRANCH="${PLUMBER_BRANCH:-master}"

echo "Cloning ${REPO_URL} (branch ${REPO_BRANCH})"
git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${PLUMBER_DIR}"

# For reference:
# ./KG_Plumber/get_plumber.sh


# Docker Setup Later (Currently, Plumber works on Local but we are finding ways to utilize it within NOTS)