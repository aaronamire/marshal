#!/bin/sh
# Install Leaves OS XDG configuration.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Install leaves-open
sudo install -m 755 "$SCRIPT_DIR/leaves-open" /usr/local/bin/leaves-open
echo "installed leaves-open to /usr/local/bin/"

# Install default mimeapps.list (user can override in ~/.config/mimeapps.list)
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}"
if [ ! -f "${XDG_CONFIG_HOME:-$HOME/.config}/mimeapps.list" ]; then
    cp "$SCRIPT_DIR/mimeapps.list" "${XDG_CONFIG_HOME:-$HOME/.config}/mimeapps.list"
    echo "installed default mimeapps.list"
else
    echo "mimeapps.list already exists — skipping (your overrides preserved)"
fi

# Install portal config
PORTAL_DIR="$SCRIPT_DIR/../portal"
if [ -f "$PORTAL_DIR/leaves.portal" ]; then
    sudo install -m 644 "$PORTAL_DIR/leaves.portal" /usr/share/xdg-desktop-portal/portals/
    echo "installed leaves.portal"
fi
if [ -f "$PORTAL_DIR/leaves-portals.conf" ]; then
    sudo install -m 644 "$PORTAL_DIR/leaves-portals.conf" /usr/share/xdg-desktop-portal/
    echo "installed leaves-portals.conf"
fi

# Install session desktop entry
SESSION_DIR="$SCRIPT_DIR/../session"
if [ -f "$SESSION_DIR/leaves-os.desktop" ]; then
    sudo install -m 644 "$SESSION_DIR/leaves-os.desktop" /usr/share/wayland-sessions/
    echo "installed leaves-os.desktop to /usr/share/wayland-sessions/"
fi

echo "done"
