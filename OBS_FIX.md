# OBS Screen Capture Fix

## Diagnosis

OBS launched but `Screen Capture (PipeWire)` failed to construct, the
screen-picker dialog rendered garbled, and the log showed:

- `info: [pipewire] No capture sources available`
- `error: Source ID 'pipewire-screen-capture-source' not found`
- `error: Failed to create source 'Screen Capture (PipeWire)'!`
- `warning: Can't create subsurface, not supported by the compositor`
- `warning: Failed to register with host portal QDBusError(... "Connection
  already associated with an application ID")`

### Root causes

**A. Broken portal routing (the headline bug).**
`/usr/share/xdg-desktop-portal/marshal-portals.conf` declares
`org.freedesktop.impl.portal.ScreenCast=marshal`, which routes screencast
to the portal whose backend is `org.freedesktop.impl.portal.desktop.marshal`.
**No such backend is installed** — there is no `.service` file and no
`xdg-desktop-portal-marshal` binary. xdg-desktop-portal silently fails to
activate it, so `linux-pipewire.so` registers no source. The
`xdg-desktop-portal-wlr` backend *is* installed and activatable on D-Bus —
we just never told the portal to use it.

**B. Missing `wl_subcompositor` global.**
`compositor.c` called `wlr_compositor_create()` but never
`wlr_subcompositor_create()`. wlroots does not auto-create the
subcompositor. Without it, Qt-Wayland (OBS, all Qt apps) can't compose
popups/tooltips — hence the "Can't create subsurface" warning and the
garbled screen-picker dialog.

**C. Wrong `UseIn` in `portal/marshal.portal`.**
It said `UseIn=marshal-compositor` but `start-session.sh` exports
`XDG_CURRENT_DESKTOP=marshal`, so the auto-matcher would never have
picked it. Masked previously by the explicit override in
`marshal-portals.conf`, but still wrong.

**D. `XDG_CURRENT_DESKTOP` missing the `wlroots` token.**
sway uses `sway:wlroots`, Hyprland uses `Hyprland:wlroots`. Some apps
and portal versions key off `wlroots` to enable wlr-specific paths.

**E. Non-bug noise** that resolves once A is fixed:

- `Failed to register with host portal … Connection already associated
  with an application ID` — OBS retries portal init after the first
  ScreenCast attempt fails.
- `=== env ===` empty in the diagnostic log — pre-existing
  terminal-paste bug already fixed (the regex was line-wrapped
  mid-quote, so grep matched nothing).

## Changes (committed to repo)

1. **`compositor/compositor.c`**
   - Added `#include <wlr/types/wlr_subcompositor.h>`.
   - Added `wlr_subcompositor_create(server.display);` next to
     `wlr_compositor_create()`. Fixes Qt-Wayland popup rendering.

2. **`portal/marshal.portal`**
   - `DBusName`: `…desktop.marshal` → `…desktop.wlr` (route to the
     actually-installed backend).
   - `UseIn`: `marshal-compositor` → `marshal;wlroots;` (matches
     `XDG_CURRENT_DESKTOP`).

3. **`scripts/start-session.sh`**, **`systemd/marshal-compositor.service`**,
   **`portal/session-start.sh`**
   - `XDG_CURRENT_DESKTOP=marshal` → `marshal:wlroots`
     (sway/Hyprland convention).

`marshal-portals.conf` was left as-is — it references the portal-file
basename `marshal`, which now resolves to the wlr backend via the
`marshal.portal` change.

Compositor rebuilt cleanly (`ninja -C compositor/builddir`).

## Deployment (requires sudo + session restart)

```bash
# 1. Deploy the fixed portal file (only marshal.portal changed;
#    marshal-portals.conf is already in sync with the repo).
sudo cp ~/dev/marshal/portal/marshal.portal /usr/share/xdg-desktop-portal/portals/

# 2. Stop any running portal backends so they respawn under the new
#    config and the updated dbus activation environment.
systemctl --user stop \
    xdg-desktop-portal.service \
    xdg-desktop-portal-wlr.service \
    xdg-desktop-portal-gtk.service 2>/dev/null
pkill -u "$USER" -f xdg-desktop-portal 2>/dev/null

# 3. Log out of Marshal and back in. This picks up
#    XDG_CURRENT_DESKTOP=marshal:wlroots and the rebuilt compositor
#    with wl_subcompositor.

# 4. Verify.
echo "$XDG_CURRENT_DESKTOP"           # → marshal:wlroots

busctl --user --no-pager call \
    org.freedesktop.portal.Desktop \
    /org/freedesktop/portal/desktop \
    org.freedesktop.DBus.Properties Get \
    ss org.freedesktop.portal.ScreenCast version
# Should return a version uint, not an error.

pgrep -fa xdg-desktop-portal-wlr      # Appears after first screencast call.

# 5. Re-run the OBS diagnostic (one-liner form avoids the paste line-wrap
#    issue that produced the empty "=== env ===" section last time).
mkdir -p ~/.marshal/logs && rm -f ~/.marshal/logs/obs-debug.log && \
    obs --verbose 2>&1 | tee ~/.marshal/logs/obs-debug.log
```

## Expected result

After the redeploy and relogin:

- `Screen Capture (PipeWire)` source constructs successfully — the
  `Source ID 'pipewire-screen-capture-source' not found` error is gone.
- The screen-picker dialog renders correctly — the `Can't create
  subsurface` warning is gone.
- The `Failed to register with host portal` retry warning is gone (it
  was a downstream symptom of the screencast failure).
