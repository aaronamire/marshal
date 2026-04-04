#!/bin/sh
# Leaves OS session startup — ensures portals and daemons are running
# before the compositor launches user applications.
#
# Called by: leaves-compositor (via wl_event_loop callback after first output)
# Or:        ~/.config/leaves/autostart

set -e

export XDG_CURRENT_DESKTOP=leaves
export XDG_SESSION_TYPE=wayland

# Start XDG Desktop Portal (needed for Firefox file dialogs, screen sharing, etc.)
if command -v xdg-desktop-portal >/dev/null 2>&1; then
    /usr/lib/xdg-desktop-portal -r &
fi

# Start wlr portal backend (screencopy, screenshot)
if command -v xdg-desktop-portal-wlr >/dev/null 2>&1; then
    xdg-desktop-portal-wlr &
fi

# Start notification daemon
if command -v leaves-notifyd >/dev/null 2>&1; then
    leaves-notifyd &
fi

# Start agent daemon
if command -v python3 >/dev/null 2>&1; then
    python3 -m agentd &
fi
