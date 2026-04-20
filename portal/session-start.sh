#!/bin/sh
# Marshal session startup — ensures portals and daemons are running
# before the compositor launches user applications.
#
# Called by: marshal-compositor (via wl_event_loop callback after first output)
# Or:        ~/.config/marshal/autostart

set -e

export XDG_CURRENT_DESKTOP=marshal
export XDG_SESSION_TYPE=wayland
export XDG_SESSION_DESKTOP=marshal

# ── XDG Desktop Portal ──
if command -v xdg-desktop-portal >/dev/null 2>&1; then
    /usr/lib/xdg-desktop-portal -r &
fi

# wlr portal backend (screencopy, screenshot)
if command -v xdg-desktop-portal-wlr >/dev/null 2>&1; then
    xdg-desktop-portal-wlr &
fi

# ── Marshal daemons ──
if command -v marshal-notifyd >/dev/null 2>&1; then
    marshal-notifyd &
fi

if command -v python3 >/dev/null 2>&1; then
    python3 -m agentd &
fi

# ── XDG Autostart ──
# Launch .desktop entries from standard autostart directories.
# Respects OnlyShowIn/NotShowIn for the "marshal" desktop.
launch_autostart_entry() {
    local desktop_file="$1"
    [ -f "$desktop_file" ] || return

    # Skip hidden entries
    if grep -qi '^Hidden=true' "$desktop_file" 2>/dev/null; then
        return
    fi

    # Check OnlyShowIn (if present, must include "marshal")
    only=$(grep -i '^OnlyShowIn=' "$desktop_file" 2>/dev/null | cut -d= -f2)
    if [ -n "$only" ]; then
        echo "$only" | tr ';' '\n' | grep -qi 'marshal' || return
    fi

    # Check NotShowIn (if present, must not include "marshal")
    notin=$(grep -i '^NotShowIn=' "$desktop_file" 2>/dev/null | cut -d= -f2)
    if [ -n "$notin" ]; then
        echo "$notin" | tr ';' '\n' | grep -qi 'marshal' && return
    fi

    # Extract Exec line and launch
    exec_line=$(grep -m1 '^Exec=' "$desktop_file" 2>/dev/null | cut -d= -f2-)
    if [ -n "$exec_line" ]; then
        # Strip desktop-entry field codes (%f %u %F %U etc.)
        exec_line=$(echo "$exec_line" | sed 's/%[fFuUdDnNickvm]//g')
        eval "$exec_line" &
    fi
}

autostart_dirs="${XDG_CONFIG_HOME:-$HOME/.config}/autostart /etc/xdg/autostart"

for dir in $autostart_dirs; do
    [ -d "$dir" ] || continue
    for entry in "$dir"/*.desktop; do
        [ -e "$entry" ] || continue
        launch_autostart_entry "$entry"
    done
done
