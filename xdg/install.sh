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

# Install session desktop entry. The .desktop file points at
# /usr/local/bin/marshal-session so it's portable across users; we generate
# that wrapper now so it invokes the start-session.sh from this checkout.
SESSION_DIR="$SCRIPT_DIR/../session"
MARSHAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ -f "$SESSION_DIR/marshal.desktop" ]; then
    sudo tee /usr/local/bin/marshal-session >/dev/null <<EOF
#!/bin/sh
exec "$MARSHAL_ROOT/scripts/start-session.sh" "\$@"
EOF
    sudo chmod 755 /usr/local/bin/marshal-session
    echo "installed marshal-session wrapper to /usr/local/bin/ (root: $MARSHAL_ROOT)"
    sudo install -m 644 "$SESSION_DIR/marshal.desktop" /usr/share/wayland-sessions/
    echo "installed marshal.desktop to /usr/share/wayland-sessions/"
fi

echo "done"
