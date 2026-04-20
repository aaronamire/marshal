#!/bin/sh
# Install Marshal XDG configuration.
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Install marshal-open
sudo install -m 755 "$SCRIPT_DIR/marshal-open" /usr/local/bin/marshal-open
echo "installed marshal-open to /usr/local/bin/"

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
if [ -f "$PORTAL_DIR/marshal.portal" ]; then
    sudo install -m 644 "$PORTAL_DIR/marshal.portal" /usr/share/xdg-desktop-portal/portals/
    echo "installed marshal.portal"
fi
if [ -f "$PORTAL_DIR/marshal-portals.conf" ]; then
    sudo install -m 644 "$PORTAL_DIR/marshal-portals.conf" /usr/share/xdg-desktop-portal/
    echo "installed marshal-portals.conf"
fi

# Install session desktop entry
SESSION_DIR="$SCRIPT_DIR/../session"
if [ -f "$SESSION_DIR/marshal.desktop" ]; then
    sudo install -m 644 "$SESSION_DIR/marshal.desktop" /usr/share/wayland-sessions/
    echo "installed marshal.desktop to /usr/share/wayland-sessions/"
fi

echo "done"
