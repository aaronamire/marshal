#!/bin/sh
# Install Marshal systemd user units.
# Usage: ./install.sh
set -e

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

mkdir -p "$UNIT_DIR"

for unit in \
    marshal-inference.service \
    marshal-agentd.service \
    marshal-api.service \
    marshal-compositor.service \
    marshal-notifyd.service \
    marshal-session.target; do
    cp "$SCRIPT_DIR/$unit" "$UNIT_DIR/$unit"
    echo "installed $unit"
done

systemctl --user daemon-reload
echo "systemd user units reloaded"
echo ""
echo "Enable the session target:"
echo "  systemctl --user enable marshal-session.target"
echo ""
echo "Or start individual services:"
echo "  systemctl --user start marshal-inference"
echo "  systemctl --user start marshal-agentd"
echo "  systemctl --user start marshal-api"
