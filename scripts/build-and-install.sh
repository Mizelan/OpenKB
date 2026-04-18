#!/usr/bin/env bash
# build-and-install.sh — Install openkb fork globally via pip editable + wrapper
#
# Usage:
#   ./scripts/build-and-install.sh           # full install
#   ./scripts/build-and-install.sh --rebuild  # force reinstall
#   ./scripts/build-and-install.sh --uninstall # remove
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INSTALL_BIN="${HOME}/.local/bin"
WRAPPER="${INSTALL_BIN}/openkb"

echo "=== openkb fork install ==="
echo "  Source: ${REPO_ROOT}"
echo "  Wrapper: ${WRAPPER}"
echo ""

install() {
    echo ">>> Installing openkb fork in editable mode..."
    pip3 install -e "${REPO_ROOT}" --quiet 2>&1 || {
        echo "ERROR: pip install failed. Try: pip3 install -e ${REPO_ROOT}"
        exit 1
    }

    echo ">>> Creating wrapper script at ${WRAPPER}..."
    mkdir -p "${INSTALL_BIN}"
    cat > "${WRAPPER}" <<'WRAPPER'
#!/usr/bin/env bash
# openkb wrapper — runs forked OpenKB with subprocess executor
# Installed by: OpenKB/scripts/build-and-install.sh
exec python3 -c "from openkb.cli import cli; cli()" "$@"
WRAPPER
    chmod +x "${WRAPPER}"

    echo ">>> Verifying installation..."
    local version
    version=$(python3 -c "import openkb; print(openkb.__version__)" 2>/dev/null || echo "unknown")
    echo "  Version: ${version}"
    echo "  Wrapper: ${WRAPPER}"
    echo "  Command: $(which openkb 2>/dev/null || echo 'not found')"
    echo ""
    echo "  Usage: openkb add <file>"
    echo "         openkb watch"
    echo "         openkb status"
    echo "         openkb list"
}

uninstall() {
    echo ">>> Uninstalling openkb fork..."
    pip3 uninstall openkb -y 2>/dev/null || true
    rm -f "${WRAPPER}"
    echo "  Removed: ${WRAPPER}"
    echo "  Removed: pip package 'openkb'"
}

rebuild() {
    echo ">>> Force reinstall..."
    pip3 uninstall openkb -y 2>/dev/null || true
    install
}

case "${1:-}" in
    --rebuild)
        rebuild
        ;;
    --uninstall)
        uninstall
        ;;
    "")
        install
        ;;
    *)
        echo "Usage: $0 [--rebuild|--uninstall]"
        exit 1
        ;;
esac