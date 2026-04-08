#!/bin/sh
# Install Leaves OS systemd user units.
# Usage: ./install.sh
set -e

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

mkdir -p "$UNIT_DIR"

for unit in \
    leaves-inference.service \
    leaves-agentd.service \
    leaves-api.service \
    leaves-compositor.service \
    leaves-notifyd.service \
    leaves-session.target; do
    cp "$SCRIPT_DIR/$unit" "$UNIT_DIR/$unit"
    echo "installed $unit"
done

systemctl --user daemon-reload
echo "systemd user units reloaded"
echo ""
echo "Enable the session target:"
echo "  systemctl --user enable leaves-session.target"
echo ""
echo "Or start individual services:"
echo "  systemctl --user start leaves-inference"
echo "  systemctl --user start leaves-agentd"
echo "  systemctl --user start leaves-api"
