#!/bin/sh
# Install Marshal systemd --user units.
# The shipped unit files use the placeholder __MARSHAL_ROOT__ instead of a
# hardcoded path; install.sh substitutes the absolute path of this checkout
# so the units work regardless of where the user cloned the repo.
set -e

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
MARSHAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$UNIT_DIR"

for unit in \
    marshal-inference.service \
    marshal-agentd.service \
    marshal-api.service \
    marshal-compositor.service \
    marshal-notifyd.service \
    marshal-session.target; do
    sed "s|__MARSHAL_ROOT__|$MARSHAL_ROOT|g" \
        "$SCRIPT_DIR/$unit" > "$UNIT_DIR/$unit"
    echo "installed $unit (root: $MARSHAL_ROOT)"
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
