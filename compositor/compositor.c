/*
 * Marshal compositor — wlr_scene-based Wayland compositor.
 *
 * Layout: Marshal AI panel (left, Cairo-rendered) + app windows (right, xdg_shell).
 * The panel is a wlr_scene_buffer; app windows are wlr_scene_xdg_surface nodes.
 * wlr_scene handles compositing, damage tracking, and z-ordering.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <errno.h>
#include <math.h>
#include <linux/input-event-codes.h>

#include <cairo.h>

#include <wayland-server-core.h>
#include <wlr/backend.h>
#include <wlr/render/allocator.h>
#include <wlr/render/wlr_renderer.h>
#include <wlr/types/wlr_output.h>
#include <wlr/types/wlr_output_layout.h>
#include <wlr/types/wlr_scene.h>
#include <wlr/types/wlr_xdg_shell.h>
#include <wlr/types/wlr_cursor.h>
#include <wlr/types/wlr_xcursor_manager.h>
#include <wlr/types/wlr_compositor.h>
#include <wlr/types/wlr_subcompositor.h>
#include <wlr/types/wlr_data_device.h>
#include <wlr/types/wlr_keyboard.h>
#include <wlr/types/wlr_seat.h>
#include <wlr/types/wlr_input_device.h>
#include <wlr/types/wlr_pointer.h>
#include <wlr/types/wlr_buffer.h>
#include <wlr/interfaces/wlr_buffer.h>
#include <wlr/types/wlr_xdg_decoration_v1.h>
#include <wlr/types/wlr_presentation_time.h>
#include <wlr/types/wlr_screencopy_v1.h>
#include <wlr/types/wlr_xdg_output_v1.h>
#include <wlr/types/wlr_fractional_scale_v1.h>
#include <wlr/types/wlr_layer_shell_v1.h>
#include <wlr/types/wlr_text_input_v3.h>
#include <wlr/types/wlr_input_method_v2.h>
#include <wlr/types/wlr_primary_selection_v1.h>
#include <wlr/types/wlr_session_lock_v1.h>
#include <wlr/xwayland/xwayland.h>
#include <wlr/backend/session.h>
#include <wlr/util/log.h>
#include <xkbcommon/xkbcommon.h>
#include <drm_fourcc.h>

#include <cjson/cJSON.h>
#include <systemd/sd-bus.h>

#include "renderer.h"
#include "feed.h"
#include "input.h"
#include "geometry.h"
#include "status.h"

/* ── Layout constants ── */

#define PANEL_WIDTH 420  /* fixed width for the AI panel (legacy) */
#define TITLEBAR_H   32  /* server-side title bar height */

/* ── Modifier bit-flags ── */

#define MOD_CTRL  (1 << 0)
#define MOD_ALT   (1 << 1)
#define MOD_SUPER (1 << 2)
#define MOD_SHIFT (1 << 3)

/* ── Workspaces ── */

#define NUM_WORKSPACES 4

/* ── Focus mode ── */

enum marshal_focus_mode {
	FOCUS_NONE,   /* keyboard input is idle (no target) */
	FOCUS_PANEL,  /* keyboard goes to the Marshal input bar */
	FOCUS_APP,    /* keyboard goes to the focused toplevel */
};

enum marshal_cursor_mode {
	CURSOR_PASSTHROUGH,
	CURSOR_MOVE,        /* dragging a window by its title bar */
};

/* ── Panel buffer (custom wlr_buffer wrapping Cairo pixel data) ── */

struct marshal_panel_buffer {
	struct wlr_buffer base;
	void *data;
	size_t stride;
	int width, height;
};

static void panel_buffer_destroy(struct wlr_buffer *wlr_buf) {
	struct marshal_panel_buffer *buf = wl_container_of(wlr_buf, buf, base);
	/* data is owned by Cairo — do not free */
	free(buf);
}

static bool panel_buffer_begin_data_ptr_access(struct wlr_buffer *wlr_buf,
		uint32_t flags, void **data, uint32_t *format, size_t *stride) {
	struct marshal_panel_buffer *buf = wl_container_of(wlr_buf, buf, base);
	if (flags & WLR_BUFFER_DATA_PTR_ACCESS_WRITE) return false;
	*data = buf->data;
	*format = DRM_FORMAT_ARGB8888;
	*stride = buf->stride;
	return true;
}

static void panel_buffer_end_data_ptr_access(struct wlr_buffer *wlr_buf) {
	/* no-op */
}

static const struct wlr_buffer_impl panel_buffer_impl = {
	.destroy = panel_buffer_destroy,
	.begin_data_ptr_access = panel_buffer_begin_data_ptr_access,
	.end_data_ptr_access = panel_buffer_end_data_ptr_access,
};

static struct marshal_panel_buffer *panel_buffer_create(void *data,
		int width, int height, size_t stride) {
	struct marshal_panel_buffer *buf = calloc(1, sizeof(*buf));
	if (!buf) return NULL;
	wlr_buffer_init(&buf->base, &panel_buffer_impl, width, height);
	buf->data = data;
	buf->stride = stride;
	buf->width = width;
	buf->height = height;
	return buf;
}

/* ── Window control glyph buffer (PlayStation-style: ◯ ▢ ✕) ──
 * Owns a cairo_surface_t that is released when the buffer is dropped. */

struct marshal_glyph_buffer {
	struct wlr_buffer base;
	cairo_surface_t *surface;
};

static void glyph_buffer_destroy(struct wlr_buffer *wlr_buf) {
	struct marshal_glyph_buffer *b = wl_container_of(wlr_buf, b, base);
	cairo_surface_destroy(b->surface);
	free(b);
}

static bool glyph_buffer_begin_data_ptr_access(struct wlr_buffer *wlr_buf,
		uint32_t flags, void **data, uint32_t *format, size_t *stride) {
	struct marshal_glyph_buffer *b = wl_container_of(wlr_buf, b, base);
	if (flags & WLR_BUFFER_DATA_PTR_ACCESS_WRITE) return false;
	*data = cairo_image_surface_get_data(b->surface);
	*format = DRM_FORMAT_ARGB8888;
	*stride = (size_t)cairo_image_surface_get_stride(b->surface);
	return true;
}

static void glyph_buffer_end_data_ptr_access(struct wlr_buffer *wlr_buf) {
	(void)wlr_buf;
}

static const struct wlr_buffer_impl glyph_buffer_impl = {
	.destroy = glyph_buffer_destroy,
	.begin_data_ptr_access = glyph_buffer_begin_data_ptr_access,
	.end_data_ptr_access = glyph_buffer_end_data_ptr_access,
};

enum marshal_ps_glyph {
	MARSHAL_PS_CIRCLE,    /* close    — black ring   (◯) */
	MARSHAL_PS_TRIANGLE,  /* maximize — black square (▢) — drawn as a
	                       * rectangle inscribed in the same bounding
	                       * box as the circle so the row reads as a
	                       * symmetric ◯ ▢ ✕ trio. The enum name is
	                       * historical (PlayStation-shape lineage). */
	MARSHAL_PS_CROSS,     /* minimize — black cross  (✕) */
};

#define MARSHAL_BTN_PX 20

static struct wlr_buffer *make_ps_glyph_buffer(enum marshal_ps_glyph g) {
	const int sz = MARSHAL_BTN_PX;
	cairo_surface_t *surf = cairo_image_surface_create(
		CAIRO_FORMAT_ARGB32, sz, sz);
	if (!surf || cairo_surface_status(surf) != CAIRO_STATUS_SUCCESS) {
		if (surf) cairo_surface_destroy(surf);
		return NULL;
	}
	cairo_t *cr = cairo_create(surf);
	cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
	cairo_set_source_rgba(cr, 0, 0, 0, 0);
	cairo_paint(cr);
	cairo_set_operator(cr, CAIRO_OPERATOR_OVER);
	cairo_set_antialias(cr, CAIRO_ANTIALIAS_BEST);

	const double cx = sz / 2.0;
	const double cy = sz / 2.0;
	const double r  = sz * 0.40;

	/* Monochrome black on transparent — matches the rest of the
	 * compositor chrome. No filled disc background. */
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 1.0);
	cairo_set_line_cap(cr, CAIRO_LINE_CAP_ROUND);
	cairo_set_line_join(cr, CAIRO_LINE_JOIN_ROUND);

	switch (g) {
	case MARSHAL_PS_CIRCLE: {
		cairo_set_line_width(cr, 2.0);
		cairo_arc(cr, cx, cy, r, 0, 2 * M_PI);
		cairo_stroke(cr);
		break;
	}
	case MARSHAL_PS_TRIANGLE: {
		/* Square inscribed in the same bounding box as the circle
		 * (side = 2r), so the middle button reads at the same
		 * visual weight as the circle on either side. */
		cairo_set_line_width(cr, 2.0);
		cairo_rectangle(cr, cx - r, cy - r, 2 * r, 2 * r);
		cairo_stroke(cr);
		break;
	}
	case MARSHAL_PS_CROSS: {
		cairo_set_line_width(cr, 2.2);
		double a = r * 0.85;
		cairo_move_to(cr, cx - a, cy - a);
		cairo_line_to(cr, cx + a, cy + a);
		cairo_move_to(cr, cx + a, cy - a);
		cairo_line_to(cr, cx - a, cy + a);
		cairo_stroke(cr);
		break;
	}
	}

	cairo_destroy(cr);
	cairo_surface_flush(surf);

	struct marshal_glyph_buffer *buf = calloc(1, sizeof(*buf));
	if (!buf) {
		cairo_surface_destroy(surf);
		return NULL;
	}
	wlr_buffer_init(&buf->base, &glyph_buffer_impl, sz, sz);
	buf->surface = surf;
	return &buf->base;
}

static struct wlr_scene_buffer *create_ps_button(struct wlr_scene_tree *parent,
		enum marshal_ps_glyph g) {
	struct wlr_buffer *buf = make_ps_glyph_buffer(g);
	if (!buf) return NULL;
	struct wlr_scene_buffer *sbuf = wlr_scene_buffer_create(parent, buf);
	wlr_buffer_drop(buf);
	return sbuf;
}

/* ── Scene-tree owners ──
 *
 * Three different struct types embed a wlr_scene_tree and stash a back-pointer
 * to themselves in `node.data` (so cursor hit-testing can recover the owner).
 * They are NOT layout-compatible. To distinguish them safely, every such
 * struct begins with a `kind` tag; readers of `node.data` MUST check the tag
 * before casting.
 *
 * Why this exists: prior to the tag, `toplevel_at()` blindly returned the
 * `node.data` pointer cast to `marshal_toplevel*`. Clicking a layer-shell
 * surface (mako notification, slurp region picker, lock surface) handed
 * `focus_toplevel()` a `marshal_layer_surface*`; reading the struct at
 * offsets that exist in `marshal_toplevel` but not in `marshal_layer_surface`
 * crashed in `wlr_scene_node_place_above`, taking the whole session down.
 */
enum marshal_scene_kind {
	MARSHAL_SCENE_KIND_NONE = 0,
	MARSHAL_SCENE_KIND_TOPLEVEL,   /* marshal_toplevel        — XDG */
	MARSHAL_SCENE_KIND_XWAYLAND,   /* marshal_xwayland_surface */
	MARSHAL_SCENE_KIND_LAYER,      /* marshal_layer_surface    */
};

/* ── Toplevel (managed app window) ── */

struct marshal_toplevel {
	enum marshal_scene_kind kind;  /* must be first; MARSHAL_SCENE_KIND_TOPLEVEL */
	struct wl_list link;  /* marshal_server.toplevels */
	struct marshal_server *server;
	struct wlr_xdg_toplevel *xdg_toplevel;

	/* Scene hierarchy:
	 *   frame_tree (positioned at window x,y in app_tree)
	 *     ├── titlebar_bg   (rect: title bar background)
	 *     ├── btn_close      (◯ — close)
	 *     ├── btn_max        (▢ — maximize)
	 *     ├── btn_min        (✕ — minimize)
	 *     └── scene_tree     (xdg surface, at y=TITLEBAR_H)
	 */
	struct wlr_scene_tree *frame_tree;
	struct wlr_scene_tree *scene_tree;
	struct wlr_scene_rect *titlebar_bg;
	struct wlr_scene_buffer *btn_close;
	struct wlr_scene_buffer *btn_max;
	struct wlr_scene_buffer *btn_min;

	/* Floating geometry (output coords) */
	int x, y;
	int width, height;     /* client surface size, excl. title bar */
	bool maximized;
	bool fullscreen;
	bool minimized;
	int workspace;         /* 0..NUM_WORKSPACES-1 */
	int saved_x, saved_y, saved_w, saved_h;

	struct wl_listener map;
	struct wl_listener unmap;
	struct wl_listener commit;
	struct wl_listener destroy;
	struct wl_listener request_maximize;
	struct wl_listener request_fullscreen;
};

/* ── Layer surface (layer shell) ── */

struct marshal_layer_surface {
	enum marshal_scene_kind kind;  /* must be first; MARSHAL_SCENE_KIND_LAYER */
	struct marshal_server *server;
	struct wlr_layer_surface_v1 *layer_surface;
	struct wlr_scene_layer_surface_v1 *scene;

	struct wl_listener map;
	struct wl_listener unmap;
	struct wl_listener commit;
	struct wl_listener destroy;
};

/* ── XWayland surface ── */

struct marshal_xwayland_surface {
	enum marshal_scene_kind kind;  /* must be first; MARSHAL_SCENE_KIND_XWAYLAND */
	struct wl_list link;  /* marshal_server.toplevels — shares list with xdg */
	struct marshal_server *server;
	struct wlr_xwayland_surface *xsurface;

	/* SSD frame tree (same pattern as marshal_toplevel) */
	struct wlr_scene_tree *frame_tree;
	struct wlr_scene_tree *scene_tree;
	struct wlr_scene_rect *titlebar_bg;
	struct wlr_scene_buffer *btn_close;
	struct wlr_scene_buffer *btn_max;
	struct wlr_scene_buffer *btn_min;

	int x, y;
	int width, height;
	bool maximized;
	bool fullscreen;
	bool minimized;
	int workspace;         /* 0..NUM_WORKSPACES-1 */
	int saved_x, saved_y, saved_w, saved_h;

	struct wl_listener map;
	struct wl_listener unmap;
	struct wl_listener destroy;
	struct wl_listener request_configure;
	struct wl_listener request_maximize;
	struct wl_listener request_fullscreen;
	struct wl_listener set_geometry;
};

/* ── Output ── */

struct marshal_output {
	struct wl_list link;
	struct marshal_server *server;
	struct wlr_output *wlr_output;
	struct wlr_scene_output *scene_output;
	struct wl_listener frame;
	struct wl_listener request_state;
	struct wl_listener destroy;
};

/* ── Keyboard ── */

struct marshal_keyboard {
	struct marshal_server *server;
	struct wlr_keyboard *wlr_keyboard;
	struct wl_listener key;
	struct wl_listener modifiers;
	struct wl_listener destroy;
};

/* ── Server ── */

struct marshal_server {
	struct wl_display *display;
	struct wl_event_loop *event_loop;
	struct wlr_backend *backend;
	struct wlr_session *session;
	struct wlr_renderer *renderer;
	struct wlr_allocator *allocator;
	struct wlr_compositor *compositor;
	struct wlr_output_layout *output_layout;

	/* Scene graph */
	struct wlr_scene *scene;
	struct wlr_scene_output_layout *scene_layout;
	struct wlr_scene_tree *panel_tree;  /* parent for panel buffer */
	struct wlr_scene_tree *app_tree;    /* parent for app windows */
	struct wlr_scene_buffer *panel_scene_buf;
	struct wlr_scene_rect *panel_bg;    /* background behind panel */

	struct wl_list outputs;  /* marshal_output */
	struct wlr_seat *seat;

	/* Wayland protocols */
	struct wlr_xdg_shell *xdg_shell;
	struct wlr_xdg_decoration_manager_v1 *decoration_mgr;
	struct wlr_presentation *presentation;
	struct wlr_screencopy_manager_v1 *screencopy_mgr;
	struct wlr_xdg_output_manager_v1 *xdg_output_mgr;
	struct wlr_fractional_scale_manager_v1 *fractional_scale_mgr;

	/* Layer shell */
	struct wlr_layer_shell_v1 *layer_shell;
	struct wlr_scene_tree *layer_trees[4]; /* bg, bottom, top, overlay */

	/* Text input / IME */
	struct wlr_text_input_manager_v3 *text_input_mgr;
	struct wlr_input_method_manager_v2 *input_method_mgr;
	struct wlr_input_method_v2 *input_method;
	struct wlr_text_input_v3 *active_text_input;

	/* XWayland */
	struct wlr_xwayland *xwayland;

	/* Listeners (decoration) */
	struct wl_listener new_decoration;

	/* Listeners (layer shell) */
	struct wl_listener new_layer_surface;

	/* Listeners (text input / IME) */
	struct wl_listener new_text_input;
	struct wl_listener new_input_method;

	/* Listeners (XWayland) */
	struct wl_listener xwayland_ready;
	struct wl_listener xwayland_new_surface;

	/* Cursor */
	struct wlr_cursor *cursor;
	struct wlr_xcursor_manager *cursor_mgr;

	/* App windows */
	struct wl_list toplevels;  /* marshal_toplevel */

	/* Marshal subsystems */
	struct marshal_feed *feed;
	struct marshal_input input;
	struct marshal_renderer *lrenderer;

	/* Focus */
	enum marshal_focus_mode focus_mode;

	/* Modifier state */
	uint32_t modifiers;

	/* Timers */
	struct wl_event_source *cursor_timer;
	struct wl_event_source *anim_timer;
	struct wl_event_source *briefing_retry_timer;
	struct wl_event_source *status_timer;

	/* System status bar */
	struct marshal_status *status;

	/* Session lock (ext-session-lock-v1) */
	struct wlr_session_lock_manager_v1 *session_lock_mgr;
	struct wlr_session_lock_v1 *active_session_lock;
	struct wlr_scene_tree *lock_tree; /* above everything when locked */
	bool locked;
	struct wl_listener new_session_lock;
	struct wl_listener session_lock_new_surface;
	struct wl_listener session_lock_unlock;
	struct wl_listener session_lock_destroy;

	/* Compositor → agentd event broadcast */
	int                    event_srv_fd;
	struct wl_event_source *event_srv_src;
	int                    event_clients[8];
	int                    event_client_count;

	/* agentd → compositor proactive-intent push */
	int                    proactive_srv_fd;
	struct wl_event_source *proactive_srv_src;
	struct {
		int fd;
		struct wl_event_source *src;
		char buf[8192];
		size_t buf_len;
	}                      proactive_clients[4];
	int                    proactive_client_count;

	/* PID → app_id mapping for child process tracking */
	struct { pid_t pid; char app_id[64]; } child_pids[64];
	int child_pid_count;

	/* logind PrepareForSleep → auto-lock */
	sd_bus *logind_bus;
	sd_bus_slot *sleep_slot;
	struct wl_event_source *logind_event;

	/* Taskbar overlay (above app windows, always visible) */
	struct wlr_scene_tree *taskbar_tree;
	struct wlr_scene_rect *taskbar_bg;
	struct wlr_scene_buffer *taskbar_scene_buf;

	/* Workspaces */
	int active_workspace;  /* 0..NUM_WORKSPACES-1 */

	/* Floating window management */
	enum marshal_cursor_mode cursor_mode;
	struct marshal_toplevel *grabbed_toplevel;
	int grab_x, grab_y;    /* cursor offset from frame origin at grab start */

	/* Panel dirty flag */
	bool panel_dirty;

	/* Mouse text-selection state */
	bool     input_drag;          /* true while LMB is held in the input bar */
	bool     card_text_drag;      /* true while LMB is held in card text */
	uint32_t last_click_ms;       /* timestamp of previous click (ms) */
	int      last_click_offset;   /* byte offset of previous click */

	/* Listeners */
	struct wl_listener new_output;
	struct wl_listener new_input;
	struct wl_listener backend_destroy;
	struct wl_listener new_xdg_toplevel;
	struct wl_listener new_xdg_popup;
	struct wl_listener cursor_motion;
	struct wl_listener cursor_motion_absolute;
	struct wl_listener cursor_button;
	struct wl_listener cursor_axis;
	struct wl_listener cursor_frame;
	struct wl_listener request_set_cursor;
	struct wl_listener request_set_selection;
};

/* ── Forward declarations ── */

static void schedule_panel_redraw(struct marshal_server *server);
static void update_panel_buffer(struct marshal_server *server);
static void focus_toplevel(struct marshal_server *server,
	struct marshal_toplevel *toplevel);
static void relayout_toplevels(struct marshal_server *server);
static int output_width(struct marshal_server *server);
static int output_height(struct marshal_server *server);
static void update_titlebar_decorations(struct marshal_toplevel *toplevel);
static void toggle_maximize(struct marshal_toplevel *toplevel);
static void emit_window_event(struct marshal_server *server,
	const char *type, const char *app_id, const char *title,
	pid_t pid, int workspace);
static pid_t launch_subprocess_tracked(struct marshal_server *server,
	const char *app_id, const char *path, char *const argv[]);

/* ── Workspace helpers ── */

/* Show/hide toplevels based on workspace. */
static void workspace_update_visibility(struct marshal_server *server) {
	struct marshal_toplevel *toplevel;
	wl_list_for_each(toplevel, &server->toplevels, link) {
		bool visible = (toplevel->workspace == server->active_workspace)
			&& !toplevel->minimized;
		wlr_scene_node_set_enabled(&toplevel->frame_tree->node, visible);
	}
}

static void switch_workspace(struct marshal_server *server, int ws) {
	if (ws < 0 || ws >= NUM_WORKSPACES) return;
	if (ws == server->active_workspace) return;

	server->active_workspace = ws;
	workspace_update_visibility(server);

	/* Focus the top window on the new workspace, or clear focus */
	struct marshal_toplevel *toplevel;
	wl_list_for_each(toplevel, &server->toplevels, link) {
		if (toplevel->workspace == ws && !toplevel->minimized) {
			server->focus_mode = FOCUS_APP;
			focus_toplevel(server, toplevel);
			schedule_panel_redraw(server);
			return;
		}
	}
	server->focus_mode = FOCUS_NONE;
	wlr_seat_keyboard_clear_focus(server->seat);
	schedule_panel_redraw(server);
}

static void move_focused_to_workspace(struct marshal_server *server, int ws) {
	if (ws < 0 || ws >= NUM_WORKSPACES) return;
	if (wl_list_empty(&server->toplevels)) return;

	/* Find the focused toplevel on the current workspace */
	struct marshal_toplevel *toplevel;
	wl_list_for_each(toplevel, &server->toplevels, link) {
		if (toplevel->workspace == server->active_workspace &&
				!toplevel->minimized)
			break;
	}
	if (&toplevel->link == &server->toplevels) return;

	toplevel->workspace = ws;

	/* Hide it (it's now on a different workspace) */
	wlr_scene_node_set_enabled(&toplevel->frame_tree->node, false);

	/* Focus next window on current workspace */
	struct marshal_toplevel *next;
	wl_list_for_each(next, &server->toplevels, link) {
		if (next->workspace == server->active_workspace &&
				!next->minimized) {
			focus_toplevel(server, next);
			schedule_panel_redraw(server);
			return;
		}
	}
	server->focus_mode = FOCUS_NONE;
	wlr_seat_keyboard_clear_focus(server->seat);
	schedule_panel_redraw(server);
}

/* ── Window snap/fullscreen helpers ── */

static void snap_left(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;
	int ow = output_width(server);
	int oh = output_height(server);
	int usable_h = oh - INPUT_HEIGHT;

	if (!toplevel->maximized && !toplevel->fullscreen) {
		toplevel->saved_x = toplevel->x;
		toplevel->saved_y = toplevel->y;
		toplevel->saved_w = toplevel->width;
		toplevel->saved_h = toplevel->height;
	}
	toplevel->maximized = false;
	toplevel->fullscreen = false;

	toplevel->x = 0;
	toplevel->y = 0;
	toplevel->width  = ow / 2;
	toplevel->height = usable_h - TITLEBAR_H;

	wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
		toplevel->width, toplevel->height);
	wlr_scene_node_set_position(&toplevel->frame_tree->node,
		toplevel->x, toplevel->y);
	update_titlebar_decorations(toplevel);
	wlr_scene_node_set_enabled(&toplevel->titlebar_bg->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_close->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_max->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_min->node, true);
	wlr_scene_node_set_position(&toplevel->scene_tree->node, 0, TITLEBAR_H);
	schedule_panel_redraw(server);
}

static void snap_right(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;
	int ow = output_width(server);
	int oh = output_height(server);
	int usable_h = oh - INPUT_HEIGHT;

	if (!toplevel->maximized && !toplevel->fullscreen) {
		toplevel->saved_x = toplevel->x;
		toplevel->saved_y = toplevel->y;
		toplevel->saved_w = toplevel->width;
		toplevel->saved_h = toplevel->height;
	}
	toplevel->maximized = false;
	toplevel->fullscreen = false;

	toplevel->x = ow / 2;
	toplevel->y = 0;
	toplevel->width  = ow - ow / 2;
	toplevel->height = usable_h - TITLEBAR_H;

	wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
		toplevel->width, toplevel->height);
	wlr_scene_node_set_position(&toplevel->frame_tree->node,
		toplevel->x, toplevel->y);
	update_titlebar_decorations(toplevel);
	wlr_scene_node_set_enabled(&toplevel->titlebar_bg->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_close->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_max->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_min->node, true);
	wlr_scene_node_set_position(&toplevel->scene_tree->node, 0, TITLEBAR_H);
	schedule_panel_redraw(server);
}

static void enter_fullscreen(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;
	int ow = output_width(server);
	int oh = output_height(server);

	if (!toplevel->maximized && !toplevel->fullscreen) {
		toplevel->saved_x = toplevel->x;
		toplevel->saved_y = toplevel->y;
		toplevel->saved_w = toplevel->width;
		toplevel->saved_h = toplevel->height;
	}

	toplevel->fullscreen = true;
	toplevel->maximized = false;
	toplevel->x = 0;
	toplevel->y = 0;
	toplevel->width  = ow;
	toplevel->height = oh;  /* full screen, covers taskbar */

	/* Hide SSD decorations */
	wlr_scene_node_set_enabled(&toplevel->titlebar_bg->node, false);
	wlr_scene_node_set_enabled(&toplevel->btn_close->node, false);
	wlr_scene_node_set_enabled(&toplevel->btn_max->node, false);
	wlr_scene_node_set_enabled(&toplevel->btn_min->node, false);

	/* Surface starts at y=0 (no titlebar) */
	wlr_scene_node_set_position(&toplevel->scene_tree->node, 0, 0);

	wlr_xdg_toplevel_set_fullscreen(toplevel->xdg_toplevel, true);
	wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
		toplevel->width, toplevel->height);
	wlr_scene_node_set_position(&toplevel->frame_tree->node, 0, 0);

	/* Raise above taskbar */
	wlr_scene_node_raise_to_top(&toplevel->frame_tree->node);
	schedule_panel_redraw(server);
}

static void leave_fullscreen(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;

	toplevel->fullscreen = false;
	toplevel->x = toplevel->saved_x;
	toplevel->y = toplevel->saved_y;
	toplevel->width  = toplevel->saved_w;
	toplevel->height = toplevel->saved_h;

	/* Restore SSD decorations */
	wlr_scene_node_set_enabled(&toplevel->titlebar_bg->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_close->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_max->node, true);
	wlr_scene_node_set_enabled(&toplevel->btn_min->node, true);
	wlr_scene_node_set_position(&toplevel->scene_tree->node, 0, TITLEBAR_H);

	wlr_xdg_toplevel_set_fullscreen(toplevel->xdg_toplevel, false);
	wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
		toplevel->width, toplevel->height);
	wlr_scene_node_set_position(&toplevel->frame_tree->node,
		toplevel->x, toplevel->y);
	update_titlebar_decorations(toplevel);
	schedule_panel_redraw(server);
}

static void restore_window(struct marshal_toplevel *toplevel) {
	if (toplevel->fullscreen) {
		leave_fullscreen(toplevel);
	} else if (toplevel->maximized) {
		toggle_maximize(toplevel);
	}
	/* else already in floating state */
}

static void minimize_window(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;

	toplevel->minimized = true;
	wlr_scene_node_set_enabled(&toplevel->frame_tree->node, false);

	/* Focus next visible window on this workspace */
	struct marshal_toplevel *next;
	wl_list_for_each(next, &server->toplevels, link) {
		if (next != toplevel &&
				next->workspace == server->active_workspace &&
				!next->minimized) {
			focus_toplevel(server, next);
			schedule_panel_redraw(server);
			return;
		}
	}
	server->focus_mode = FOCUS_NONE;
	wlr_seat_keyboard_clear_focus(server->seat);
	schedule_panel_redraw(server);
}

static void unminimize_window(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;

	toplevel->minimized = false;
	wlr_scene_node_set_enabled(&toplevel->frame_tree->node, true);
	server->focus_mode = FOCUS_APP;
	focus_toplevel(server, toplevel);
	schedule_panel_redraw(server);
}

/* Cycle focus to the next visible window on the current workspace. */
static void cycle_window(struct marshal_server *server) {
	if (wl_list_empty(&server->toplevels)) return;

	/* Find the first non-head toplevel on the current workspace */
	struct marshal_toplevel *toplevel;
	struct marshal_toplevel *target = NULL;
	wl_list_for_each_reverse(toplevel, &server->toplevels, link) {
		if (toplevel->workspace == server->active_workspace &&
				!toplevel->minimized) {
			target = toplevel;
			break;
		}
	}
	if (!target) return;

	server->focus_mode = FOCUS_APP;
	focus_toplevel(server, target);
	schedule_panel_redraw(server);
}

/* Launch a subprocess (fire-and-forget, close inherited FDs). */
static void launch_subprocess(const char *path, char *const argv[]) {
	pid_t pid = fork();
	if (pid == 0) {
		int maxfd = sysconf(_SC_OPEN_MAX);
		for (int fd = 3; fd < maxfd && fd < 1024; fd++)
			close(fd);
		execvp(path, argv);
		_exit(127);
	}
}

/* Take a screenshot via grim. */
static void take_screenshot(bool region) {
	/* Ensure ~/Pictures/screenshots/ exists */
	const char *home = getenv("HOME");
	if (!home) return;

	char dir[512];
	snprintf(dir, sizeof(dir), "%s/Pictures", home);
	mkdir(dir, 0755);
	snprintf(dir, sizeof(dir), "%s/Pictures/screenshots", home);
	mkdir(dir, 0755);

	char filename[768];
	time_t now = time(NULL);
	struct tm *tm = localtime(&now);
	snprintf(filename, sizeof(filename),
		"%s/Pictures/screenshots/%04d-%02d-%02d_%02d%02d%02d.png",
		home,
		tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
		tm->tm_hour, tm->tm_min, tm->tm_sec);

	if (region) {
		/* grim -g "$(slurp)" <filename> — needs a shell for command substitution */
		char cmd[1024];
		snprintf(cmd, sizeof(cmd),
			"grim -g \"$(slurp)\" '%s'", filename);
		char *argv[] = { "sh", "-c", cmd, NULL };
		launch_subprocess("sh", argv);
	} else {
		char *argv[] = { "grim", filename, NULL };
		launch_subprocess("grim", argv);
	}
}

/* ── Panel dimensions ── */

static int output_height(struct marshal_server *server) {
	struct marshal_output *out;
	wl_list_for_each(out, &server->outputs, link) {
		int w, h;
		wlr_output_effective_resolution(out->wlr_output, &w, &h);
		return h;
	}
	return 1080;
}

static int output_width(struct marshal_server *server) {
	struct marshal_output *out;
	wl_list_for_each(out, &server->outputs, link) {
		int w, h;
		wlr_output_effective_resolution(out->wlr_output, &w, &h);
		return w;
	}
	return 1920;
}

static int effective_panel_width(struct marshal_server *server) {
	/* Panel is always full-screen — it's the desktop background.
	 * App windows float on top. */
	return output_width(server);
}

/* ── Panel clipboard source ── */

struct marshal_clipboard_source {
	struct wlr_data_source base;
	char text[1024];
	int len;
};

static void clipboard_source_send(struct wlr_data_source *wlr_source,
		const char *mime_type, int32_t fd) {
	struct marshal_clipboard_source *src =
		wl_container_of(wlr_source, src, base);
	(void)mime_type;
	write(fd, src->text, src->len);
	close(fd);
}

static void clipboard_source_destroy(struct wlr_data_source *wlr_source) {
	struct marshal_clipboard_source *src =
		wl_container_of(wlr_source, src, base);
	free(src);
}

static const struct wlr_data_source_impl clipboard_source_impl = {
	.send    = clipboard_source_send,
	.destroy = clipboard_source_destroy,
};

/* Copy an arbitrary text range to the Wayland selection. */
static void panel_copy_range_to_clipboard(struct marshal_server *server,
		const char *text, int len) {
	if (len <= 0) return;
	if (len > (int)sizeof(((struct marshal_clipboard_source *)0)->text) - 1)
		len = (int)sizeof(((struct marshal_clipboard_source *)0)->text) - 1;

	struct marshal_clipboard_source *src = calloc(1, sizeof(*src));
	if (!src) return;

	src->len = len;
	memcpy(src->text, text, len);
	src->text[len] = '\0';

	wlr_data_source_init(&src->base, &clipboard_source_impl);

	char **t1 = wl_array_add(&src->base.mime_types, sizeof(char *));
	if (!t1) { free(src); return; }
	*t1 = strdup("text/plain;charset=utf-8");

	char **t2 = wl_array_add(&src->base.mime_types, sizeof(char *));
	if (!t2) { wlr_data_source_destroy(&src->base); return; }
	*t2 = strdup("text/plain");

	wlr_seat_set_selection(server->seat, &src->base,
		wl_display_next_serial(server->display));
}

/* Copy the full panel input buffer to the Wayland selection. */
static void panel_copy_to_clipboard(struct marshal_server *server) {
	panel_copy_range_to_clipboard(server,
		server->input.buf, server->input.len);
}

/* Paste the Wayland selection into the panel input bar (Ctrl+V).
 * Uses a pipe + 100 ms select() timeout.  For remote Wayland clients the
 * display must be flushed first so they receive the send request. */
static void panel_paste_from_clipboard(struct marshal_server *server) {
	struct wlr_data_source *sel = server->seat->selection_source;
	if (!sel) return;

	/* Pick the best available text MIME type */
	const char *mime = NULL;
	char **p;
	wl_array_for_each(p, &sel->mime_types) {
		if (strcmp(*p, "text/plain;charset=utf-8") == 0) {
			mime = "text/plain;charset=utf-8";
			break;
		}
		if (strcmp(*p, "text/plain") == 0 && !mime)
			mime = "text/plain";
	}
	if (!mime) return;

	int pipefd[2];
	if (pipe(pipefd) < 0) return;

	/* wlr_data_source_send() closes pipefd[1] after sending */
	wlr_data_source_send(sel, mime, pipefd[1]);

	/* Flush pending Wayland protocol messages so client-side sources
	 * receive the send request before we block on the read end */
	wl_display_flush_clients(server->display);

	char text[1024];
	fd_set rfds;
	FD_ZERO(&rfds);
	FD_SET(pipefd[0], &rfds);
	struct timeval tv = {.tv_sec = 0, .tv_usec = 100000}; /* 100 ms */
	if (select(pipefd[0] + 1, &rfds, NULL, NULL, &tv) > 0) {
		ssize_t n = read(pipefd[0], text, sizeof(text) - 1);
		if (n > 0) {
			text[n] = '\0';
			input_paste(&server->input, text, (int)n);
		}
	}
	close(pipefd[0]);
}

/* ── Cursor blink timer ── */

static int cursor_timer_cb(void *data) {
	struct marshal_server *server = data;
	if (server->focus_mode == FOCUS_PANEL) {
		if (input_tick_cursor(&server->input, 530)) {
			schedule_panel_redraw(server);
		}
	} else {
		/* Hide cursor when panel isn't focused */
		if (server->input.cursor_visible) {
			server->input.cursor_visible = false;
			schedule_panel_redraw(server);
		}
	}
	wl_event_source_timer_update(server->cursor_timer, 530);
	return 0;
}

/* ── Animation timer ── */

static int anim_timer_cb(void *data) {
	struct marshal_server *server = data;
	float dt = 1.0f / 60.0f;
	if (feed_animate(server->feed, dt)) {
		schedule_panel_redraw(server);
		wl_event_source_timer_update(server->anim_timer, 16);
	}
	return 0;
}

/* ── Briefing retry timer ── */

static int briefing_retry_cb(void *data) {
	struct marshal_server *server = data;
	pthread_mutex_lock(&server->feed->mutex);
	bool empty = !server->feed->briefing.loaded;
	pthread_mutex_unlock(&server->feed->mutex);
	if (empty) {
		feed_load_briefing(server->feed);
		feed_load_watchers(server->feed);
		schedule_panel_redraw(server);
	}
	return 0;
}

/* ── Status bar timer — fires every 1 s ── */

static int status_timer_cb(void *data) {
	struct marshal_server *server = data;
	static int tick = 0;

	status_update_clock(server->status);
	if (tick++ % 30 == 0)   /* full system poll every 30 s */
		status_poll(server->status);

	schedule_panel_redraw(server);
	wl_event_source_timer_update(server->status_timer, 1000);
	return 0;
}

/* ── logind PrepareForSleep: auto-lock before suspend ── */

static int prepare_for_sleep_cb(sd_bus_message *msg, void *userdata,
		sd_bus_error *ret_error) {
	(void)ret_error;
	struct marshal_server *server = userdata;
	int going_to_sleep = 0;
	if (sd_bus_message_read(msg, "b", &going_to_sleep) < 0)
		return 0;
	if (going_to_sleep) {
		/* Launch locker before the system actually suspends */
		pid_t pid = launch_subprocess_tracked(server,
			"marshal-locker", "marshal-locker",
			(char *const[]){"marshal-locker", NULL});
		/* Brief delay to let the locker grab input before sleep completes */
		if (pid > 0)
			usleep(200000); /* 200 ms */
	}
	return 0;
}

static int logind_bus_dispatch(int fd, uint32_t mask, void *data) {
	(void)fd;
	(void)mask;
	struct marshal_server *server = data;
	while (sd_bus_process(server->logind_bus, NULL) > 0)
		;
	return 0;
}

static void setup_logind_sleep_monitor(struct marshal_server *server) {
	if (sd_bus_default_system(&server->logind_bus) < 0) {
		fprintf(stderr, "logind: cannot connect to system bus\n");
		return;
	}

	int r = sd_bus_match_signal(server->logind_bus,
		&server->sleep_slot,
		"org.freedesktop.login1",
		"/org/freedesktop/login1",
		"org.freedesktop.login1.Manager",
		"PrepareForSleep",
		prepare_for_sleep_cb, server);
	if (r < 0) {
		fprintf(stderr, "logind: failed to subscribe to PrepareForSleep\n");
		sd_bus_unref(server->logind_bus);
		server->logind_bus = NULL;
		return;
	}

	int fd = sd_bus_get_fd(server->logind_bus);
	if (fd >= 0) {
		server->logind_event = wl_event_loop_add_fd(
			server->event_loop, fd,
			WL_EVENT_READABLE, logind_bus_dispatch, server);
	}
	fprintf(stderr, "logind: PrepareForSleep monitor active\n");
}

/* ── Wakeup pipe ── */

static int wakeup_handler(int fd, uint32_t mask __attribute__((unused)),
		void *data) {
	struct marshal_server *server = data;
	char byte;
	bool quit = false;
	bool open_history = false;
	/* Drain the pipe. 'q' from feed_request_exit() means the user typed
	 * "exit" / "quit" in the intent bar — terminate the compositor cleanly
	 * after the read loop so we don't leave the pipe in an odd state.
	 * 'h' = history was just (re)loaded; flip the feed pane open so the
	 * user actually sees the cards instead of the bare desktop. */
	while (read(fd, &byte, 1) == 1) {
		if (byte == 'q') quit = true;
		else if (byte == 'h') open_history = true;
	}
	if (quit) {
		wl_display_terminate(server->display);
		return 0;
	}
	if (open_history && server->status) {
		server->status->history_open = true;
	}
	schedule_panel_redraw(server);
	wl_event_source_timer_update(server->anim_timer, 16);
	return 0;
}

/* ── Panel rendering ── */

static void schedule_panel_redraw(struct marshal_server *server) {
	server->panel_dirty = true;
	/* Schedule frame on all outputs so the scene gets re-committed */
	struct marshal_output *output;
	wl_list_for_each(output, &server->outputs, link) {
		wlr_output_schedule_frame(output->wlr_output);
	}
}

static void update_panel_buffer(struct marshal_server *server) {
	if (!server->panel_dirty) return;
	server->panel_dirty = false;

	int pw = effective_panel_width(server);
	int ph = output_height(server);

	/* Ensure cairo surface matches panel size */
	if (server->lrenderer->width != pw ||
			server->lrenderer->height != ph) {
		renderer_resize(server->lrenderer, pw, ph);
	}

	/* Snapshot the running-toplevel list into the renderer so the bottom
	 * bar can paint a monogram per app. We resolve a display label and a
	 * one-character glyph here (UI-side) instead of in the renderer so
	 * Wayland surface details stay out of the rendering layer. */
	{
		struct marshal_renderer *lr = server->lrenderer;
		struct wlr_surface *focused =
			server->seat ? server->seat->keyboard_state.focused_surface
				: NULL;
		int n = 0;
		struct marshal_toplevel *t;
		wl_list_for_each(t, &server->toplevels, link) {
			if (n >= MARSHAL_MAX_BAR_APPS) break;
			if (t->workspace != server->active_workspace) continue;
			/* DON'T skip minimized — the bar's whole purpose is to give
			 * the user a way back to a hidden window. We dim the icon
			 * for minimized windows below (focused != true), so they
			 * still read as "running, just not visible right now". */

			const char *label = NULL;
			if (t->xdg_toplevel) {
				if (t->xdg_toplevel->app_id && t->xdg_toplevel->app_id[0])
					label = t->xdg_toplevel->app_id;
				else if (t->xdg_toplevel->title && t->xdg_toplevel->title[0])
					label = t->xdg_toplevel->title;
			}
			if (!label || !label[0]) label = "App";

			/* Last "."-separated component is the human-readable name
			 * for reverse-DNS app_ids ("org.mozilla.firefox" → "firefox") */
			const char *base = strrchr(label, '.');
			base = base ? base + 1 : label;

			snprintf(lr->apps[n].label, sizeof(lr->apps[n].label),
				"%s", base);
			/* Icon name: full app_id (some Icon=lines key off the
			 * reverse-DNS form like org.mozilla.firefox), with the
			 * shortened base preserved as a secondary search term
			 * via the resolver's lower-case fallback. */
			snprintf(lr->apps[n].icon_name,
				sizeof(lr->apps[n].icon_name), "%s", label);
			/* First UTF-8 codepoint, uppercased if ASCII. */
			unsigned char c = (unsigned char)base[0];
			if (c < 0x80) {
				if (c >= 'a' && c <= 'z') c -= ('a' - 'A');
				lr->apps[n].glyph[0] = (char)c;
				lr->apps[n].glyph[1] = '\0';
			} else {
				int len = 1;
				if      ((c & 0xE0) == 0xC0) len = 2;
				else if ((c & 0xF0) == 0xE0) len = 3;
				else if ((c & 0xF8) == 0xF0) len = 4;
				memcpy(lr->apps[n].glyph, base, len);
				lr->apps[n].glyph[len] = '\0';
			}
			lr->apps[n].focused = (focused != NULL &&
				t->xdg_toplevel != NULL &&
				t->xdg_toplevel->base->surface == focused);
			lr->apps[n].hit_x = lr->apps[n].hit_y = 0;
			lr->apps[n].hit_w = lr->apps[n].hit_h = 0;
			n++;
		}
		lr->app_count = n;
	}

	int stride;
	unsigned char *pixels = renderer_draw_frame(server->lrenderer,
		server->feed, &server->input, &stride);
	if (!pixels) return;

	/* Create new wlr_buffer wrapping the Cairo pixel data */
	struct marshal_panel_buffer *pbuf = panel_buffer_create(pixels,
		pw, ph, (size_t)stride);
	if (!pbuf) return;

	/* Update the scene buffer node */
	wlr_scene_buffer_set_buffer(server->panel_scene_buf, &pbuf->base);
	wlr_buffer_drop(&pbuf->base);

	/* Update the taskbar overlay with the bottom strip of the panel.
	 * This overlay sits above app windows so the taskbar is always visible. */
	if (server->taskbar_scene_buf && ph > INPUT_HEIGHT) {
		int tb_y = ph - INPUT_HEIGHT;
		unsigned char *tb_pixels = pixels + tb_y * stride;
		struct marshal_panel_buffer *tb_buf = panel_buffer_create(
			tb_pixels, pw, INPUT_HEIGHT, (size_t)stride);
		if (tb_buf) {
			wlr_scene_buffer_set_buffer(server->taskbar_scene_buf,
				&tb_buf->base);
			wlr_buffer_drop(&tb_buf->base);
		}
	}
}

/* ── Output frame handler ── */

static void output_frame(struct wl_listener *listener, void *data) {
	struct marshal_output *output = wl_container_of(listener, output, frame);
	struct marshal_server *server = output->server;

	/* Re-render panel if dirty */
	update_panel_buffer(server);

	struct wlr_scene_output *scene_output = output->scene_output;
	wlr_scene_output_commit(scene_output, NULL);

	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	wlr_scene_output_send_frame_done(scene_output, &now);
}

static void output_request_state(struct wl_listener *listener, void *data) {
	struct marshal_output *output =
		wl_container_of(listener, output, request_state);
	const struct wlr_output_event_request_state *event = data;
	wlr_output_commit_state(output->wlr_output, event->state);
	schedule_panel_redraw(output->server);
	relayout_toplevels(output->server);
}

static void output_destroy(struct wl_listener *listener, void *data) {
	struct marshal_output *output = wl_container_of(listener, output, destroy);
	wl_list_remove(&output->frame.link);
	wl_list_remove(&output->request_state.link);
	wl_list_remove(&output->destroy.link);
	wl_list_remove(&output->link);
	free(output);
}

/* ── Toplevel management ── */

/* Update title bar decoration rects to match window width. */
static void update_titlebar_decorations(struct marshal_toplevel *toplevel) {
	wlr_scene_rect_set_size(toplevel->titlebar_bg,
		toplevel->width, TITLEBAR_H);
	const int bsz = MARSHAL_BTN_PX;
	const int gap = 6;
	const int margin = 10;
	int btn_y = (TITLEBAR_H - bsz) / 2;
	wlr_scene_node_set_position(&toplevel->btn_close->node,
		toplevel->width - bsz - margin, btn_y);
	wlr_scene_node_set_position(&toplevel->btn_max->node,
		toplevel->width - 2 * bsz - margin - gap, btn_y);
	wlr_scene_node_set_position(&toplevel->btn_min->node,
		toplevel->width - 3 * bsz - margin - 2 * gap, btn_y);
}

static void toplevel_map(struct wl_listener *listener, void *data) {
	struct marshal_toplevel *toplevel =
		wl_container_of(listener, toplevel, map);
	struct marshal_server *server = toplevel->server;

	toplevel->workspace = server->active_workspace;
	wl_list_insert(&server->toplevels, &toplevel->link);

	/* Position the floating frame */
	wlr_scene_node_set_position(&toplevel->frame_tree->node,
		toplevel->x, toplevel->y);
	update_titlebar_decorations(toplevel);

	emit_window_event(server, "window_opened",
		toplevel->xdg_toplevel->app_id,
		toplevel->xdg_toplevel->title, 0,
		server->active_workspace);

	/* Focus the new window */
	server->focus_mode = FOCUS_APP;
	focus_toplevel(server, toplevel);
	schedule_panel_redraw(server);
}

static void toplevel_unmap(struct wl_listener *listener, void *data) {
	struct marshal_toplevel *toplevel =
		wl_container_of(listener, toplevel, unmap);
	struct marshal_server *server = toplevel->server;

	emit_window_event(server, "window_closed",
		toplevel->xdg_toplevel->app_id, "",
		0, toplevel->workspace);

	wl_list_remove(&toplevel->link);

	/* If this was the focused window, switch focus */
	if (!wl_list_empty(&server->toplevels)) {
		struct marshal_toplevel *next = wl_container_of(
			server->toplevels.next, next, link);
		focus_toplevel(server, next);
	} else {
		server->focus_mode = FOCUS_NONE;
		wlr_seat_keyboard_clear_focus(server->seat);
		schedule_panel_redraw(server);
	}
}

static void toplevel_commit(struct wl_listener *listener, void *data) {
	struct marshal_toplevel *toplevel =
		wl_container_of(listener, toplevel, commit);

	if (toplevel->xdg_toplevel->base->initial_commit) {
		struct marshal_server *server = toplevel->server;
		int ow = output_width(server);
		int oh = output_height(server);

		/* Default floating size: ~70% of screen, centered */
		int usable_h = oh - INPUT_HEIGHT;  /* above the taskbar */
		toplevel->width  = ow * 7 / 10;
		toplevel->height = usable_h * 7 / 10;
		toplevel->x = (ow - toplevel->width) / 2;
		toplevel->y = (usable_h - toplevel->height - TITLEBAR_H) / 2;

		wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
			toplevel->width, toplevel->height);
		wlr_xdg_toplevel_set_activated(toplevel->xdg_toplevel, true);
	}
}

static void toplevel_destroy(struct wl_listener *listener, void *data) {
	struct marshal_toplevel *toplevel =
		wl_container_of(listener, toplevel, destroy);

	wl_list_remove(&toplevel->map.link);
	wl_list_remove(&toplevel->unmap.link);
	wl_list_remove(&toplevel->commit.link);
	wl_list_remove(&toplevel->destroy.link);
	wl_list_remove(&toplevel->request_maximize.link);
	wl_list_remove(&toplevel->request_fullscreen.link);

	/* Reparent the xdg scene tree out of frame_tree so wlroots
	 * can destroy it independently via its own lifecycle, then
	 * destroy our decoration frame. */
	if (toplevel->scene_tree) {
		wlr_scene_node_reparent(&toplevel->scene_tree->node,
			toplevel->server->app_tree);
		wlr_scene_node_set_enabled(&toplevel->scene_tree->node, false);
	}
	if (toplevel->frame_tree)
		wlr_scene_node_destroy(&toplevel->frame_tree->node);

	/* Clear grab if this toplevel was being dragged */
	if (toplevel->server->grabbed_toplevel == toplevel) {
		toplevel->server->cursor_mode = CURSOR_PASSTHROUGH;
		toplevel->server->grabbed_toplevel = NULL;
	}

	free(toplevel);
}

static void toggle_maximize(struct marshal_toplevel *toplevel) {
	struct marshal_server *server = toplevel->server;
	int ow = output_width(server);
	int oh = output_height(server);

	if (!toplevel->maximized) {
		toplevel->saved_x = toplevel->x;
		toplevel->saved_y = toplevel->y;
		toplevel->saved_w = toplevel->width;
		toplevel->saved_h = toplevel->height;
		toplevel->maximized = true;
		toplevel->x = 0;
		toplevel->y = 0;
		toplevel->width  = ow;
		toplevel->height = oh - INPUT_HEIGHT - TITLEBAR_H;
	} else {
		toplevel->maximized = false;
		toplevel->x = toplevel->saved_x;
		toplevel->y = toplevel->saved_y;
		toplevel->width  = toplevel->saved_w;
		toplevel->height = toplevel->saved_h;
	}

	wlr_xdg_toplevel_set_maximized(toplevel->xdg_toplevel,
		toplevel->maximized);
	wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
		toplevel->width, toplevel->height);
	wlr_scene_node_set_position(&toplevel->frame_tree->node,
		toplevel->x, toplevel->y);
	update_titlebar_decorations(toplevel);
	schedule_panel_redraw(server);
}

static void toplevel_request_maximize(struct wl_listener *listener,
		void *data) {
	struct marshal_toplevel *toplevel =
		wl_container_of(listener, toplevel, request_maximize);
	toggle_maximize(toplevel);
}

static void toplevel_request_fullscreen(struct wl_listener *listener,
		void *data) {
	struct marshal_toplevel *toplevel =
		wl_container_of(listener, toplevel, request_fullscreen);
	if (toplevel->fullscreen)
		leave_fullscreen(toplevel);
	else
		enter_fullscreen(toplevel);
}

static void focus_toplevel(struct marshal_server *server,
		struct marshal_toplevel *toplevel) {
	if (!toplevel) return;

	/* Deactivate previously focused toplevel */
	struct wlr_surface *prev_surface =
		server->seat->keyboard_state.focused_surface;
	if (prev_surface) {
		struct wlr_xdg_toplevel *prev =
			wlr_xdg_toplevel_try_from_wlr_surface(prev_surface);
		if (prev) {
			wlr_xdg_toplevel_set_activated(prev, false);
		}
	}

	/* Activate and focus new toplevel */
	wl_list_remove(&toplevel->link);
	wl_list_insert(&server->toplevels, &toplevel->link);
	wlr_scene_node_raise_to_top(&toplevel->frame_tree->node);
	wlr_xdg_toplevel_set_activated(toplevel->xdg_toplevel, true);

	emit_window_event(server, "window_focused",
		toplevel->xdg_toplevel->app_id, "", 0,
		toplevel->workspace);

	/* Set keyboard on seat before notify_enter — wlroots uses the seat's
	 * current keyboard to send keymap + modifiers to the entering surface.
	 * Without this, the client may never receive a keymap event. */
	struct wlr_keyboard *keyboard = wlr_seat_get_keyboard(server->seat);
	if (keyboard) {
		wlr_seat_set_keyboard(server->seat, keyboard);
		wlr_seat_keyboard_notify_enter(server->seat,
			toplevel->xdg_toplevel->base->surface,
			keyboard->keycodes, keyboard->num_keycodes,
			&keyboard->modifiers);
		fprintf(stderr, "[focus] keyboard enter sent to surface=%p\n",
			(void *)toplevel->xdg_toplevel->base->surface);
	} else {
		fprintf(stderr, "[focus] WARNING: no keyboard on seat!\n");
	}
}

static void relayout_toplevels(struct marshal_server *server) {
	int ow = output_width(server);
	int oh = output_height(server);

	/* Panel (desktop) is always full screen */
	wlr_scene_rect_set_size(server->panel_bg, ow, oh);
	renderer_resize(server->lrenderer, ow, oh);

	/* App tree at (0,0) — windows float with individual positions */
	wlr_scene_node_set_position(&server->app_tree->node, 0, 0);

	/* Taskbar overlay at bottom, full width, above app windows */
	if (server->taskbar_tree) {
		wlr_scene_node_set_position(&server->taskbar_tree->node,
			0, oh - INPUT_HEIGHT);
		if (server->taskbar_bg)
			wlr_scene_rect_set_size(server->taskbar_bg, ow, INPUT_HEIGHT);
	}

	/* Update maximized windows to fill available area */
	struct marshal_toplevel *toplevel;
	wl_list_for_each(toplevel, &server->toplevels, link) {
		if (toplevel->maximized) {
			toplevel->x = 0;
			toplevel->y = 0;
			toplevel->width  = ow;
			toplevel->height = oh - INPUT_HEIGHT - TITLEBAR_H;
			wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel,
				toplevel->width, toplevel->height);
			wlr_scene_node_set_position(&toplevel->frame_tree->node,
				0, 0);
		}
		update_titlebar_decorations(toplevel);
	}

	schedule_panel_redraw(server);
}

/* ── xdg_shell handlers ── */

static void server_new_xdg_toplevel(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_xdg_toplevel);
	struct wlr_xdg_toplevel *xdg_toplevel = data;

	struct marshal_toplevel *toplevel = calloc(1, sizeof(*toplevel));
	toplevel->kind = MARSHAL_SCENE_KIND_TOPLEVEL;
	toplevel->server = server;
	toplevel->xdg_toplevel = xdg_toplevel;

	/* Create frame tree (outer container for decorations + surface) */
	toplevel->frame_tree = wlr_scene_tree_create(server->app_tree);
	toplevel->frame_tree->node.data = toplevel;

	/* Title bar background — light gray */
	float tb_color[4] = {0.898f, 0.898f, 0.898f, 1.0f};  /* #E5E5E5 */
	toplevel->titlebar_bg = wlr_scene_rect_create(
		toplevel->frame_tree, 800, TITLEBAR_H, tb_color);

	/* Window control buttons (monochrome glyphs, right-aligned) */
	toplevel->btn_close = create_ps_button(toplevel->frame_tree,
		MARSHAL_PS_CIRCLE);
	toplevel->btn_max = create_ps_button(toplevel->frame_tree,
		MARSHAL_PS_TRIANGLE);
	toplevel->btn_min = create_ps_button(toplevel->frame_tree,
		MARSHAL_PS_CROSS);

	/* XDG surface below the title bar */
	toplevel->scene_tree = wlr_scene_xdg_surface_create(
		toplevel->frame_tree, xdg_toplevel->base);
	wlr_scene_node_set_position(&toplevel->scene_tree->node, 0, TITLEBAR_H);
	toplevel->scene_tree->node.data = toplevel;

	struct wlr_xdg_surface *xdg_surface = xdg_toplevel->base;

	toplevel->map.notify = toplevel_map;
	wl_signal_add(&xdg_surface->surface->events.map, &toplevel->map);

	toplevel->unmap.notify = toplevel_unmap;
	wl_signal_add(&xdg_surface->surface->events.unmap, &toplevel->unmap);

	toplevel->commit.notify = toplevel_commit;
	wl_signal_add(&xdg_surface->surface->events.commit, &toplevel->commit);

	toplevel->destroy.notify = toplevel_destroy;
	wl_signal_add(&xdg_toplevel->events.destroy, &toplevel->destroy);

	toplevel->request_maximize.notify = toplevel_request_maximize;
	wl_signal_add(&xdg_toplevel->events.request_maximize,
		&toplevel->request_maximize);

	toplevel->request_fullscreen.notify = toplevel_request_fullscreen;
	wl_signal_add(&xdg_toplevel->events.request_fullscreen,
		&toplevel->request_fullscreen);
}

static void server_new_xdg_popup(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_xdg_popup);
	struct wlr_xdg_popup *popup = data;
	struct wlr_xdg_surface *parent =
		wlr_xdg_surface_try_from_wlr_surface(popup->parent);
	if (!parent) return;

	struct wlr_scene_tree *parent_tree = parent->data;
	if (!parent_tree) {
		/* Parent has no scene tree (shouldn't happen normally).
		 * Fall back to the app tree so the popup still renders. */
		parent_tree = server->app_tree;
	}
	popup->base->data =
		wlr_scene_xdg_surface_create(parent_tree, popup->base);
}

/* ── Keyboard handling ── */

static void keyboard_handle_key(struct wl_listener *listener, void *data) {
	struct marshal_keyboard *keyboard =
		wl_container_of(listener, keyboard, key);
	struct marshal_server *server = keyboard->server;
	struct wlr_keyboard_key_event *event = data;

	uint32_t keycode = event->keycode + 8;
	const xkb_keysym_t *syms;
	int nsyms = xkb_state_key_get_syms(
		keyboard->wlr_keyboard->xkb_state, keycode, &syms);

	/* Get the unmodified (layout-level-0) keysym for compositor bindings.
	 * xkb_state_key_get_syms() returns keysyms with ALL modifiers applied,
	 * which can transform keys when Super/Logo is held on some XKB configs.
	 * Compositor shortcuts need the base key, not the modified one. */
	xkb_keysym_t raw_sym = xkb_state_key_get_one_sym(
		keyboard->wlr_keyboard->xkb_state, keycode);
	/* Also get the layout-only sym (no modifiers) as a final fallback */
	const xkb_keysym_t *layout_syms;
	int layout_nsyms = xkb_keymap_key_get_syms_by_level(
		xkb_state_get_keymap(keyboard->wlr_keyboard->xkb_state),
		keycode,
		xkb_state_key_get_layout(keyboard->wlr_keyboard->xkb_state,
			keycode),
		0, &layout_syms);
	xkb_keysym_t base_sym = (layout_nsyms > 0) ? layout_syms[0] : raw_sym;

	bool handled = false;

	if (event->state == WL_KEYBOARD_KEY_STATE_PRESSED) {
		for (int i = 0; i < nsyms; i++) {
			/* ── VT switch: Ctrl+Alt+F1..F12 (always allowed, even locked) ── */
			if ((server->modifiers & MOD_CTRL) &&
					(server->modifiers & MOD_ALT)) {
				if (syms[i] >= XKB_KEY_XF86Switch_VT_1 &&
						syms[i] <= XKB_KEY_XF86Switch_VT_12) {
					if (server->session) {
						unsigned vt = syms[i] -
							XKB_KEY_XF86Switch_VT_1 + 1;
						wlr_session_change_vt(server->session, vt);
					}
					handled = true;
					break;
				}
				/* Ctrl+Alt+Backspace: emergency compositor exit */
				if (syms[i] == XKB_KEY_BackSpace) {
					wl_display_terminate(server->display);
					handled = true;
					break;
				}
			}
		}

		/* Session locked: only VT switch and emergency exit pass through */
		if (!handled && server->locked)
			handled = true;

		/* Use base_sym for compositor shortcuts — immune to modifier
		 * remapping (Super on some XKB configs can transform keysyms). */

		/* ── Super+Return: launch terminal ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_Return || raw_sym == XKB_KEY_Return)) {
			launch_subprocess_tracked(server, "marshal-terminal",
				"marshal-terminal",
				(char *const[]){"marshal-terminal", NULL});
			handled = true;
		}

		/* ── Super+L: lock screen ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_l || raw_sym == XKB_KEY_l)) {
			launch_subprocess_tracked(server, "marshal-locker",
				"marshal-locker",
				(char *const[]){"marshal-locker", NULL});
			handled = true;
		}

		/* ── Super+Q: close focused app window ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_q || raw_sym == XKB_KEY_q)) {
			if (!wl_list_empty(&server->toplevels)) {
				struct marshal_toplevel *top = wl_container_of(
					server->toplevels.next, top, link);
				wlr_xdg_toplevel_send_close(top->xdg_toplevel);
			}
			handled = true;
		}

		/* ── Super+1..4: switch workspace ── */
		/* ── Super+Shift+1..4: move focused window to workspace ── */
		if (!handled && (server->modifiers & MOD_SUPER)) {
			int ws = -1;
			if (base_sym == XKB_KEY_1 || raw_sym == XKB_KEY_1) ws = 0;
			else if (base_sym == XKB_KEY_2 || raw_sym == XKB_KEY_2) ws = 1;
			else if (base_sym == XKB_KEY_3 || raw_sym == XKB_KEY_3) ws = 2;
			else if (base_sym == XKB_KEY_4 || raw_sym == XKB_KEY_4) ws = 3;

			if (ws >= 0) {
				if (server->modifiers & MOD_SHIFT)
					move_focused_to_workspace(server, ws);
				else
					switch_workspace(server, ws);
				handled = true;
			}
		}

		/* ── Super+Tab: cycle windows on current workspace ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_Tab || raw_sym == XKB_KEY_Tab)) {
			cycle_window(server);
			handled = true;
		}

		/* ── Super+Left/Right/Up/Down: window snapping ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				!wl_list_empty(&server->toplevels)) {
			struct marshal_toplevel *top = wl_container_of(
				server->toplevels.next, top, link);
			if (top->workspace == server->active_workspace &&
					!top->minimized) {
				if (base_sym == XKB_KEY_Left || raw_sym == XKB_KEY_Left) {
					snap_left(top);
					handled = true;
				} else if (base_sym == XKB_KEY_Right || raw_sym == XKB_KEY_Right) {
					snap_right(top);
					handled = true;
				} else if (base_sym == XKB_KEY_Up || raw_sym == XKB_KEY_Up) {
					if (!top->maximized) toggle_maximize(top);
					handled = true;
				} else if (base_sym == XKB_KEY_Down || raw_sym == XKB_KEY_Down) {
					restore_window(top);
					handled = true;
				}
			}
		}

		/* ── Super+F: toggle fullscreen ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_f || raw_sym == XKB_KEY_f)) {
			if (!wl_list_empty(&server->toplevels)) {
				struct marshal_toplevel *top = wl_container_of(
					server->toplevels.next, top, link);
				if (top->workspace == server->active_workspace) {
					if (top->fullscreen)
						leave_fullscreen(top);
					else
						enter_fullscreen(top);
				}
			}
			handled = true;
		}

		/* ── Super+M: minimize focused window ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_m || raw_sym == XKB_KEY_m)) {
			if (!wl_list_empty(&server->toplevels)) {
				struct marshal_toplevel *top = wl_container_of(
					server->toplevels.next, top, link);
				if (top->workspace == server->active_workspace &&
						!top->minimized)
					minimize_window(top);
			}
			handled = true;
		}

		/* ── Print Screen: screenshot ── */
		if (!handled && (base_sym == XKB_KEY_Print || raw_sym == XKB_KEY_Print)) {
			take_screenshot(false);
			handled = true;
		}

		/* ── Super+Shift+S: region screenshot ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(server->modifiers & MOD_SHIFT) &&
				(base_sym == XKB_KEY_s || raw_sym == XKB_KEY_s)) {
			take_screenshot(true);
			handled = true;
		}

		/* ── XF86 media keys: volume / mic / brightness ──
		 * These ship on every laptop keyboard but were never bound, so
		 * the audio agent (which only reacts to typed "volume up")
		 * was unreachable from the hardware keys. Shell out to wpctl
		 * (PipeWire) — it falls through to PulseAudio's compatibility
		 * layer when only PA is running, so a single binary covers
		 * both stacks. Fire-and-forget; we don't need the exit code. */
		if (!handled) {
			const char *cmd = NULL;
			if (raw_sym == XKB_KEY_XF86AudioRaiseVolume)
				cmd = "wpctl set-volume -l 1.0 @DEFAULT_AUDIO_SINK@ 5%+";
			else if (raw_sym == XKB_KEY_XF86AudioLowerVolume)
				cmd = "wpctl set-volume @DEFAULT_AUDIO_SINK@ 5%-";
			else if (raw_sym == XKB_KEY_XF86AudioMute)
				cmd = "wpctl set-mute @DEFAULT_AUDIO_SINK@ toggle";
			else if (raw_sym == XKB_KEY_XF86AudioMicMute)
				cmd = "wpctl set-mute @DEFAULT_AUDIO_SOURCE@ toggle";
			else if (raw_sym == XKB_KEY_XF86MonBrightnessUp)
				cmd = "brightnessctl set +5%";
			else if (raw_sym == XKB_KEY_XF86MonBrightnessDown)
				cmd = "brightnessctl set 5%-";
			if (cmd) {
				char *argv[] = { "sh", "-c", (char *)cmd, NULL };
				launch_subprocess("sh", argv);
				/* Force a status repoll so the on-screen volume %
				 * updates immediately instead of waiting for the
				 * next 30 s tick. */
				if (server->status)
					status_poll(server->status);
				schedule_panel_redraw(server);
				handled = true;
			}
		}

		/* ── Ctrl+R: time-machine replay of the currently expanded
		 * history card. Only fires while a card is expanded, so plain
		 * 'r' typed into the prompt input is unaffected. The replay
		 * endpoint refuses destructive plans and the refusal is shown
		 * as a banner inside the card. */
		if (!handled && (server->modifiers & MOD_CTRL) &&
				server->feed->expanded_card >= 0 &&
				(base_sym == XKB_KEY_r || raw_sym == XKB_KEY_r)) {
			feed_replay(server->feed, server->feed->expanded_card);
			handled = true;
		}

		/* ── Super+Space: cycle focus: NONE → PANEL → APP → NONE ── */
		if (!handled && (server->modifiers & MOD_SUPER) &&
				(base_sym == XKB_KEY_space || raw_sym == XKB_KEY_space)) {
			if (server->focus_mode == FOCUS_NONE) {
				server->focus_mode = FOCUS_PANEL;
				wlr_seat_keyboard_clear_focus(server->seat);
			} else if (server->focus_mode == FOCUS_PANEL) {
				/* Find top visible window on current workspace */
				struct marshal_toplevel *top = NULL;
				struct marshal_toplevel *t;
				wl_list_for_each(t, &server->toplevels, link) {
					if (t->workspace == server->active_workspace &&
							!t->minimized) {
						top = t;
						break;
					}
				}
				if (top) {
					server->focus_mode = FOCUS_APP;
					focus_toplevel(server, top);
				} else {
					server->focus_mode = FOCUS_NONE;
					wlr_seat_keyboard_clear_focus(server->seat);
				}
			} else {
				/* FOCUS_APP → NONE */
				server->focus_mode = FOCUS_NONE;
				wlr_seat_keyboard_clear_focus(server->seat);
			}
			schedule_panel_redraw(server);
			handled = true;
		}
	}

	if (handled) return;

	/* When the authorization overlay is active, keyboard input MUST
	 * reach the overlay handler regardless of focus_mode.  Without this,
	 * clicking anywhere on the dimming layer sets focus_mode=FOCUS_NONE
	 * and silently drops all subsequent key events — making the overlay
	 * unresponsive (the original bug). */
	if (server->feed->awaiting_confirm &&
			event->state == WL_KEYBOARD_KEY_STATE_PRESSED) {
		if (input_handle_key(&server->input, event->keycode,
				server->modifiers, NULL, 0)) {
			schedule_panel_redraw(server);
			wl_event_source_timer_update(server->anim_timer, 16);
		}
		return;
	}

	/* Escape closes expanded card overlay */
	if (server->feed->expanded_card >= 0 &&
			event->state == WL_KEYBOARD_KEY_STATE_PRESSED &&
			event->keycode == 1) {
		pthread_mutex_lock(&server->feed->mutex);
		server->feed->expanded_card = -1;
		server->feed->expanded_scroll = 0;
		pthread_mutex_unlock(&server->feed->mutex);
		schedule_panel_redraw(server);
		return;
	}

	if (server->focus_mode == FOCUS_NONE) {
		/* No focus target — drop keypresses */
		return;
	} else if (server->focus_mode == FOCUS_PANEL) {
		/* Route to Marshal input handler */
		if (event->state != WL_KEYBOARD_KEY_STATE_PRESSED) return;

		/* ── Clipboard / selection shortcuts ── */
		if (server->modifiers & MOD_CTRL) {
			struct marshal_input *inp = &server->input;

			if (event->keycode == KEY_C) {
				/* Ctrl+C — copy card text selection, input
				 * selection, selected card, or full input */
				if (server->lrenderer->card_sel.card_idx >= 0 &&
						server->lrenderer->card_sel.anchor !=
						server->lrenderer->card_sel.focus) {
					char buf[2048];
					int len = renderer_card_sel_text(
						server->lrenderer,
						buf, sizeof(buf));
					if (len > 0)
						panel_copy_range_to_clipboard(
							server, buf, len);
				} else if (inp->sel_anchor != -1 &&
						inp->sel_anchor != inp->sel_focus) {
					int s = inp->sel_anchor < inp->sel_focus
						? inp->sel_anchor : inp->sel_focus;
					int e = inp->sel_anchor < inp->sel_focus
						? inp->sel_focus  : inp->sel_anchor;
					panel_copy_range_to_clipboard(server,
						inp->buf + s, e - s);
				} else if (server->feed->selected_card >= 0) {
					char buf[2048];
					pthread_mutex_lock(&server->feed->mutex);
					int len = renderer_card_copy_text(
						server->feed,
						server->feed->selected_card,
						buf, sizeof(buf));
					pthread_mutex_unlock(&server->feed->mutex);
					if (len > 0)
						panel_copy_range_to_clipboard(
							server, buf, len);
				} else if (inp->len > 0) {
					panel_copy_to_clipboard(server);
				}
				return;
			}
			if (event->keycode == KEY_X) {
				/* Ctrl+X — cut selection */
				if (inp->sel_anchor != -1 &&
						inp->sel_anchor != inp->sel_focus) {
					int s = inp->sel_anchor < inp->sel_focus
						? inp->sel_anchor : inp->sel_focus;
					int e = inp->sel_anchor < inp->sel_focus
						? inp->sel_focus  : inp->sel_anchor;
					panel_copy_range_to_clipboard(server,
						inp->buf + s, e - s);
					input_delete_selection(inp);
					schedule_panel_redraw(server);
				}
				return;
			}
			if (event->keycode == KEY_V) {
				/* Ctrl+V — paste from Wayland clipboard at cursor */
				panel_paste_from_clipboard(server);
				schedule_panel_redraw(server);
				return;
			}
			if (event->keycode == KEY_A) {
				/* Ctrl+A — select all */
				if (inp->len > 0) {
					inp->sel_anchor  = 0;
					inp->sel_focus   = inp->len;
					inp->cursor_pos  = inp->len;
					schedule_panel_redraw(server);
				}
				return;
			}
			if (event->keycode == KEY_U) {
				/* Ctrl+U — clear line */
				if (inp->len > 0) {
					inp->len = 0;
					inp->cursor_pos = 0;
					inp->buf[0] = '\0';
					inp->sel_anchor = -1;
					schedule_panel_redraw(server);
				}
				return;
			}
		}

		char utf8[8] = {0};
		int utf8_len = xkb_state_key_get_utf8(
			keyboard->wlr_keyboard->xkb_state, keycode,
			utf8, sizeof(utf8));
		bool is_printable = utf8_len > 0 && (unsigned char)utf8[0] >= 0x20;

		if (input_handle_key(&server->input, event->keycode,
				server->modifiers,
				is_printable ? utf8 : NULL,
				is_printable ? utf8_len : 0)) {
			schedule_panel_redraw(server);
			wl_event_source_timer_update(server->anim_timer, 16);
		} else if (event->keycode == 1) {
			/* Escape with empty buffer — defocus the input bar */
			server->focus_mode = FOCUS_NONE;
			schedule_panel_redraw(server);
		}
	} else {
		/* Forward to focused app */
		wlr_seat_set_keyboard(server->seat, keyboard->wlr_keyboard);
		wlr_seat_keyboard_notify_key(server->seat, event->time_msec,
			event->keycode, event->state);
		if (event->state == WL_KEYBOARD_KEY_STATE_PRESSED) {
			struct wlr_surface *focused =
				server->seat->keyboard_state.focused_surface;
			fprintf(stderr, "[key] code=%u focus_mode=%d focused_surface=%p\n",
				event->keycode, server->focus_mode, (void *)focused);
		}
	}
}

static void keyboard_handle_modifiers(struct wl_listener *listener,
		void *data) {
	struct marshal_keyboard *keyboard =
		wl_container_of(listener, keyboard, modifiers);
	struct marshal_server *server = keyboard->server;

	xkb_mod_mask_t mod_mask = xkb_state_serialize_mods(
		keyboard->wlr_keyboard->xkb_state,
		XKB_STATE_MODS_EFFECTIVE);

	server->modifiers = 0;

	struct xkb_keymap *keymap = xkb_state_get_keymap(
		keyboard->wlr_keyboard->xkb_state);

	xkb_mod_index_t ctrl_idx = xkb_keymap_mod_get_index(keymap,
		XKB_MOD_NAME_CTRL);
	if (ctrl_idx != XKB_MOD_INVALID && (mod_mask & (1 << ctrl_idx)))
		server->modifiers |= MOD_CTRL;

	xkb_mod_index_t alt_idx = xkb_keymap_mod_get_index(keymap,
		XKB_MOD_NAME_ALT);
	if (alt_idx != XKB_MOD_INVALID && (mod_mask & (1 << alt_idx)))
		server->modifiers |= MOD_ALT;

	xkb_mod_index_t logo_idx = xkb_keymap_mod_get_index(keymap,
		XKB_MOD_NAME_LOGO);
	if (logo_idx != XKB_MOD_INVALID && (mod_mask & (1 << logo_idx)))
		server->modifiers |= MOD_SUPER;

	xkb_mod_index_t shift_idx = xkb_keymap_mod_get_index(keymap,
		XKB_MOD_NAME_SHIFT);
	if (shift_idx != XKB_MOD_INVALID && (mod_mask & (1 << shift_idx)))
		server->modifiers |= MOD_SHIFT;

	/* Track keyboard layout for status bar */
	if (server->status) {
		xkb_layout_index_t idx = xkb_state_serialize_layout(
			keyboard->wlr_keyboard->xkb_state,
			XKB_STATE_LAYOUT_EFFECTIVE);
		const char *name = xkb_keymap_layout_get_name(keymap, idx);
		if (name) {
			/* "English (US)" → "US"; "Russian" → "RU" */
			const char *paren = strchr(name, '(');
			if (paren && paren[1] && paren[2]) {
				server->status->kb_layout[0] = paren[1] & ~0x20;
				server->status->kb_layout[1] = paren[2] & ~0x20;
			} else if (name[0] && name[1]) {
				server->status->kb_layout[0] = name[0] & ~0x20;
				server->status->kb_layout[1] = name[1] & ~0x20;
			}
			server->status->kb_layout[2] = '\0';
		}
	}

	if (server->focus_mode == FOCUS_APP) {
		wlr_seat_set_keyboard(server->seat, keyboard->wlr_keyboard);
		wlr_seat_keyboard_notify_modifiers(server->seat,
			&keyboard->wlr_keyboard->modifiers);
	}
}

static void keyboard_destroy(struct wl_listener *listener, void *data) {
	struct marshal_keyboard *keyboard =
		wl_container_of(listener, keyboard, destroy);
	wl_list_remove(&keyboard->key.link);
	wl_list_remove(&keyboard->modifiers.link);
	wl_list_remove(&keyboard->destroy.link);
	free(keyboard);
}

/* ── Cursor handlers ── */

static struct marshal_toplevel *toplevel_at(struct marshal_server *server,
		double lx, double ly, struct wlr_surface **surface,
		double *sx, double *sy) {
	struct wlr_scene_node *node = wlr_scene_node_at(
		&server->scene->tree.node, lx, ly, sx, sy);
	if (!node) return NULL;

	/* Walk up the tree to find the nearest scene-owner. We then verify
	 * the kind tag — `node.data` may be a layer surface or an XWayland
	 * surface, neither of which is layout-compatible with marshal_toplevel.
	 * Returning a mis-typed pointer here is what previously crashed the
	 * compositor when slurp / mako / lock surfaces were clicked. */
	struct wlr_scene_tree *tree = node->parent;
	while (tree && !tree->node.data) {
		tree = tree->node.parent;
	}
	if (!tree || !tree->node.data) return NULL;

	enum marshal_scene_kind kind =
		*(enum marshal_scene_kind *)tree->node.data;
	if (kind != MARSHAL_SCENE_KIND_TOPLEVEL) {
		/* Layer-shell or XWayland — not a marshal_toplevel. The pointer
		 * notify_button() above the call site already routed the click
		 * to the right client; we just must not raise/focus it as if it
		 * were one of our managed app windows. */
		return NULL;
	}

	/* Try to resolve a wlr_surface for pointer routing */
	*surface = NULL;
	if (node->type == WLR_SCENE_NODE_BUFFER) {
		struct wlr_scene_buffer *scene_buffer =
			wlr_scene_buffer_from_node(node);
		struct wlr_scene_surface *scene_surface =
			wlr_scene_surface_try_from_buffer(scene_buffer);
		if (scene_surface)
			*surface = scene_surface->surface;
	}
	/* Rect nodes (title bar, buttons) return toplevel with surface=NULL */

	return tree->node.data;
}

static void process_cursor_motion(struct marshal_server *server,
		uint32_t time) {
	/* Window drag mode — move the grabbed window */
	if (server->cursor_mode == CURSOR_MOVE && server->grabbed_toplevel) {
		struct marshal_toplevel *tl = server->grabbed_toplevel;
		tl->x = (int)server->cursor->x - server->grab_x;
		tl->y = (int)server->cursor->y - server->grab_y;
		wlr_scene_node_set_position(&tl->frame_tree->node, tl->x, tl->y);
		return;
	}

	double sx, sy;
	struct wlr_surface *surface = NULL;
	struct marshal_toplevel *toplevel = toplevel_at(server,
		server->cursor->x, server->cursor->y, &surface, &sx, &sy);

	if (toplevel && surface) {
		/* Over an app window surface — let the app handle the cursor */
		wlr_seat_pointer_notify_enter(server->seat, surface, sx, sy);
		wlr_seat_pointer_notify_motion(server->seat, time, sx, sy);
	} else if (toplevel && !surface) {
		/* Over title bar decoration */
		wlr_cursor_set_xcursor(server->cursor, server->cursor_mgr, "default");
		wlr_seat_pointer_clear_focus(server->seat);
	} else {
		/* Over panel/desktop or taskbar */
		int oh = output_height(server);
		int bar_y = oh - INPUT_HEIGHT;
		bool in_input_field =
			server->cursor->x >= TASKBAR_ICON_W &&
			server->cursor->x < effective_panel_width(server) - TASKBAR_APPS_W &&
			server->cursor->y >= bar_y;

		bool in_card_text = false;
		if (!in_input_field) {
			int dummy;
			in_card_text = renderer_card_text_at(server->lrenderer,
				server->cursor->x, server->cursor->y, &dummy) >= 0;
		}

		wlr_cursor_set_xcursor(server->cursor, server->cursor_mgr,
			(in_input_field || in_card_text) ? "text" : "default");
		wlr_seat_pointer_clear_focus(server->seat);
	}

	/* Extend selection while LMB is held in the input bar */
	if (server->input_drag && server->focus_mode == FOCUS_PANEL) {
		int offset = renderer_input_hit_test(server->lrenderer,
			&server->input, server->cursor->x);
		if (offset != server->input.sel_focus) {
			server->input.sel_focus  = offset;
			server->input.cursor_pos = offset;
			schedule_panel_redraw(server);
		}
	}

	/* Extend card text selection while LMB is held in card area */
	if (server->card_text_drag && server->focus_mode == FOCUS_PANEL) {
		double mx = server->cursor->x;
		double my = server->cursor->y;
		int byte_off = 0;
		int card_idx = renderer_card_text_at(server->lrenderer,
			mx, my, &byte_off);
		if (card_idx == server->lrenderer->card_sel.card_idx) {
			if (byte_off != server->lrenderer->card_sel.focus) {
				server->lrenderer->card_sel.focus = byte_off;
				schedule_panel_redraw(server);
			}
		}
	}
}

static void cursor_motion_handler(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, cursor_motion);
	struct wlr_pointer_motion_event *event = data;
	wlr_cursor_move(server->cursor, &event->pointer->base,
		event->delta_x, event->delta_y);
	process_cursor_motion(server, event->time_msec);
}

static void cursor_motion_absolute_handler(struct wl_listener *listener,
		void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, cursor_motion_absolute);
	struct wlr_pointer_motion_absolute_event *event = data;
	wlr_cursor_warp_absolute(server->cursor, &event->pointer->base,
		event->x, event->y);
	process_cursor_motion(server, event->time_msec);
}

static void cursor_button_handler(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, cursor_button);
	struct wlr_pointer_button_event *event = data;

	/* Session locked: let the lock surface client handle pointer events */
	if (server->locked) {
		wlr_seat_pointer_notify_button(server->seat, event->time_msec,
			event->button, event->state);
		return;
	}

	wlr_seat_pointer_notify_button(server->seat, event->time_msec,
		event->button, event->state);

	if (event->state == WL_POINTER_BUTTON_STATE_PRESSED) {
		double mx = server->cursor->x, my = server->cursor->y;

		/* ── Auth overlay buttons take priority over everything ── */
		if (server->lrenderer->overlay_buttons_valid) {
			if (mx >= server->lrenderer->overlay_cancel_x &&
					mx < server->lrenderer->overlay_cancel_x +
						server->lrenderer->overlay_cancel_w &&
					my >= server->lrenderer->overlay_cancel_y &&
					my < server->lrenderer->overlay_cancel_y +
						server->lrenderer->overlay_cancel_h) {
				feed_cancel(server->feed);
				schedule_panel_redraw(server);
				wl_event_source_timer_update(server->anim_timer, 16);
				return;
			}
			if (mx >= server->lrenderer->overlay_confirm_x &&
					mx < server->lrenderer->overlay_confirm_x +
						server->lrenderer->overlay_confirm_w &&
					my >= server->lrenderer->overlay_confirm_y &&
					my < server->lrenderer->overlay_confirm_y +
						server->lrenderer->overlay_confirm_h) {
				feed_confirm(server->feed);
				schedule_panel_redraw(server);
				wl_event_source_timer_update(server->anim_timer, 16);
				return;
			}
			/* Overlay is modal — swallow all other clicks */
			return;
		}

		/* ── Expanded card overlay: close button or click outside ── */
		if (server->lrenderer->expanded_overlay_valid) {
			struct marshal_renderer *lr = server->lrenderer;
			/* Close button hit */
			if (mx >= lr->expanded_close_x &&
					mx < lr->expanded_close_x + lr->expanded_close_w &&
					my >= lr->expanded_close_y &&
					my < lr->expanded_close_y + lr->expanded_close_h) {
				pthread_mutex_lock(&server->feed->mutex);
				server->feed->expanded_card = -1;
				server->feed->expanded_scroll = 0;
				pthread_mutex_unlock(&server->feed->mutex);
				schedule_panel_redraw(server);
				return;
			}
			/* Click inside content area = scroll, no close */
			if (mx >= lr->expanded_content_x &&
					mx < lr->expanded_content_x + lr->expanded_content_w &&
					my >= lr->expanded_content_y &&
					my < lr->expanded_content_y + lr->expanded_content_h) {
				/* Let scroll events handle navigation */
				return;
			}
			/* Click on dim area outside panel = close */
			pthread_mutex_lock(&server->feed->mutex);
			server->feed->expanded_card = -1;
			server->feed->expanded_scroll = 0;
			pthread_mutex_unlock(&server->feed->mutex);
			schedule_panel_redraw(server);
			return;
		}

		double sx, sy;
		struct wlr_surface *surface = NULL;
		struct marshal_toplevel *toplevel = toplevel_at(server,
			mx, my, &surface, &sx, &sy);

		if (toplevel && !surface) {
			/* Click on title bar decoration */
			int rel_x = (int)mx - toplevel->x;
			int rel_y = (int)my - toplevel->y;

			if (rel_y < TITLEBAR_H) {
				const int bsz = MARSHAL_BTN_PX;
				const int gap = 6;
				int btn_r = toplevel->width - 10;
				int btn_t = (TITLEBAR_H - bsz) / 2;
				int btn_b = btn_t + bsz;

				/* Close button (rightmost) */
				if (rel_x >= btn_r - bsz && rel_x < btn_r &&
						rel_y >= btn_t && rel_y < btn_b) {
					wlr_xdg_toplevel_send_close(toplevel->xdg_toplevel);
					goto done_press;
				}
				btn_r -= bsz + gap;
				/* Maximize button */
				if (rel_x >= btn_r - bsz && rel_x < btn_r &&
						rel_y >= btn_t && rel_y < btn_b) {
					toggle_maximize(toplevel);
					goto done_press;
				}
				btn_r -= bsz + gap;
				/* Minimize button */
				if (rel_x >= btn_r - bsz && rel_x < btn_r &&
						rel_y >= btn_t && rel_y < btn_b) {
					minimize_window(toplevel);
					goto done_press;
				}

				/* Title bar drag area — start window move */
				server->cursor_mode = CURSOR_MOVE;
				server->grabbed_toplevel = toplevel;
				server->grab_x = (int)mx - toplevel->x;
				server->grab_y = (int)my - toplevel->y;
			}

			server->focus_mode = FOCUS_APP;
			input_clear_selection(&server->input);
			if (server->status) server->status->dropdown_open = false;
			focus_toplevel(server, toplevel);
		} else if (toplevel && surface) {
			/* Click on app window surface — focus it */
			server->focus_mode = FOCUS_APP;
			input_clear_selection(&server->input);
			if (server->status) server->status->dropdown_open = false;
			focus_toplevel(server, toplevel);
		} else {
			int oh = output_height(server);
			int bar_y = oh - INPUT_HEIGHT;

			/* Click inside an open dropdown? */
			if (server->status && server->status->dropdown_open) {
				struct marshal_renderer *lr = server->lrenderer;
				if (mx >= lr->dropdown_x &&
						mx < lr->dropdown_x + lr->dropdown_w &&
						my >= lr->dropdown_y &&
						my < lr->dropdown_y + lr->dropdown_h) {
					/* Translate click to dropdown-relative coords —
					 * the section hit rects are in absolute panel
					 * coords because that's what draw_dropdown
					 * recorded. */
					const char *cmd = NULL;

					/* WiFi row → nmcli (polkit-friendly), fall back to
					 * rfkill if NM isn't installed. rfkill alone fails
					 * silently on most distros because /dev/rfkill is
					 * root-only without a udev rule, which is why the
					 * old version "did nothing." */
					if (lr->qs_wifi_w > 0 &&
							mx >= lr->qs_wifi_x &&
							mx < lr->qs_wifi_x + lr->qs_wifi_w &&
							my >= lr->qs_wifi_y &&
							my < lr->qs_wifi_y + lr->qs_wifi_h) {
						cmd = server->status->wifi_connected
							? "nmcli radio wifi off 2>/dev/null || "
							  "rfkill block wifi"
							: "nmcli radio wifi on 2>/dev/null || "
							  "rfkill unblock wifi";
					}
					/* Bluetooth row → bluetoothctl (uses polkit),
					 * fall back to rfkill. */
					else if (lr->qs_bt_w > 0 &&
							mx >= lr->qs_bt_x &&
							mx < lr->qs_bt_x + lr->qs_bt_w &&
							my >= lr->qs_bt_y &&
							my < lr->qs_bt_y + lr->qs_bt_h) {
						cmd = server->status->bt_enabled
							? "bluetoothctl power off 2>/dev/null || "
							  "rfkill block bluetooth"
							: "bluetoothctl power on 2>/dev/null || "
							  "rfkill unblock bluetooth";
					}
					/* Volume bar → set level by horizontal position.
					 * Clamp to [0,100]; pct=0 doesn't unmute, so we
					 * also unmute on any click. */
					else if (lr->qs_vol_w > 0 &&
							mx >= lr->qs_vol_x &&
							mx < lr->qs_vol_x + lr->qs_vol_w &&
							my >= lr->qs_vol_y &&
							my < lr->qs_vol_y + lr->qs_vol_h) {
						int pct = (int)((mx - lr->qs_vol_x) * 100
							/ lr->qs_vol_w);
						if (pct < 0) pct = 0;
						if (pct > 100) pct = 100;
						char buf[128];
						snprintf(buf, sizeof(buf),
							"wpctl set-mute @DEFAULT_AUDIO_SINK@ 0 && "
							"wpctl set-volume -l 1.0 "
							"@DEFAULT_AUDIO_SINK@ %d%%",
							pct);
						char *argv[] = { "sh", "-c", buf, NULL };
						launch_subprocess("sh", argv);
						if (server->status)
							status_poll(server->status);
						schedule_panel_redraw(server);
						goto done_press;
					}

					if (cmd) {
						char *argv[] = { "sh", "-c", (char *)cmd, NULL };
						launch_subprocess("sh", argv);
						/* status_poll is sysfs/popen — runs on the
						 * main thread. The toggled state takes a
						 * moment to propagate (NetworkManager,
						 * bluetoothd), so the immediate poll often
						 * still sees the OLD state. We re-poll on
						 * the next animator tick to catch up. */
						if (server->status)
							status_poll(server->status);
						wl_event_source_timer_update(
							server->anim_timer, 250);
					}

					/* Inside dropdown — consume click */
					schedule_panel_redraw(server);
					goto done_press;
				}
			}

			if (my >= bar_y) {
				/* Click in the taskbar region */
				struct marshal_renderer *lr = server->lrenderer;

				/* History icon? */
				if (mx >= lr->history_icon_x &&
						mx < lr->history_icon_x +
							lr->history_icon_w &&
						server->status) {
					server->status->history_open =
						!server->status->history_open;
					if (server->status->dropdown_open)
						server->status->dropdown_open = false;
					schedule_panel_redraw(server);
					wlr_seat_keyboard_clear_focus(server->seat);
					goto done_press;
				}

				/* Status zone? (time, wifi, bt, battery) */
				if (mx >= lr->status_zone_left_x &&
						server->status) {
					server->status->dropdown_open =
						!server->status->dropdown_open;
					schedule_panel_redraw(server);
					wlr_seat_keyboard_clear_focus(server->seat);
					goto done_press;
				}

				/* Running-app monogram? Click focuses (or restores +
				 * focuses, if the app was minimised between renders). */
				for (int ai = 0; ai < lr->app_count; ai++) {
					struct marshal_bar_app *a = &lr->apps[ai];
					if (a->hit_w <= 0) continue;
					if (mx < a->hit_x ||
							mx >= a->hit_x + a->hit_w ||
							my < a->hit_y ||
							my >= a->hit_y + a->hit_h)
						continue;
					/* Walk the toplevel list with the same filter the
					 * snapshot block uses (active workspace, minimized
					 * INCLUDED) so the index `ai` matches the painted
					 * order. If the user clicked a minimized window's
					 * icon, unminimize before focusing — that's the
					 * whole reason the icon is on the bar. */
					struct marshal_toplevel *t;
					int idx = 0;
					wl_list_for_each(t, &server->toplevels, link) {
						if (t->workspace != server->active_workspace)
							continue;
						if (idx == ai) {
							if (t->minimized)
								unminimize_window(t);
							server->focus_mode = FOCUS_APP;
							focus_toplevel(server, t);
							break;
						}
						idx++;
					}
					schedule_panel_redraw(server);
					goto done_press;
				}

				/* Otherwise: input field click */
				if (server->status)
					server->status->dropdown_open = false;
				server->focus_mode = FOCUS_PANEL;

				int offset = renderer_input_hit_test(
					server->lrenderer, &server->input, mx);

				bool dbl =
					(event->time_msec - server->last_click_ms <= 300) &&
					(offset == server->last_click_offset);

				if (dbl) {
					input_select_word(&server->input, offset);
					server->input_drag = false;
				} else {
					server->input.sel_anchor  = offset;
					server->input.sel_focus   = offset;
					server->input.cursor_pos  = offset;
					server->input_drag = true;
				}

				server->last_click_ms     = event->time_msec;
				server->last_click_offset = offset;
			} else {
				/* Click on feed area — try character-level text hit first */
				int byte_off = 0;
				int text_card = renderer_card_text_at(
					server->lrenderer, mx, my, &byte_off);
				if (text_card >= 0) {
					/* Start character-level text selection */
					server->lrenderer->card_sel.card_idx = text_card;
					server->lrenderer->card_sel.anchor = byte_off;
					server->lrenderer->card_sel.focus = byte_off;
					server->card_text_drag = true;
					pthread_mutex_lock(&server->feed->mutex);
					server->feed->selected_card = text_card;
					pthread_mutex_unlock(&server->feed->mutex);
				} else {
					/* Click outside text — toggle card expansion */
					int card_idx = renderer_card_hit_test(
						server->lrenderer, server->feed, my);
					bool became_expanded = false;
					pthread_mutex_lock(&server->feed->mutex);
					server->feed->selected_card = card_idx;
					/* Toggle expansion: click same card again to collapse */
					if (card_idx >= 0 && card_idx == server->feed->expanded_card) {
						server->feed->expanded_card = -1;
						server->feed->expanded_scroll = 0;
					} else {
						server->feed->expanded_card = card_idx;
						server->feed->expanded_scroll = 0;
						became_expanded = (card_idx >= 0);
					}
					pthread_mutex_unlock(&server->feed->mutex);
					server->lrenderer->card_sel.card_idx = -1;
					server->card_text_drag = false;
					/* Time-machine: lazy-load audit detail when a history
					 * card is expanded. No-op for non-history cards. */
					if (became_expanded)
						feed_load_detail(server->feed, card_idx);
				}

				server->focus_mode = FOCUS_PANEL;
				input_clear_selection(&server->input);
				server->input_drag = false;
				if (server->status)
					server->status->dropdown_open = false;
			}
			wlr_seat_keyboard_clear_focus(server->seat);
		}
	done_press:
		schedule_panel_redraw(server);
	}

	if (event->state == WL_POINTER_BUTTON_STATE_RELEASED) {
		if (server->cursor_mode != CURSOR_PASSTHROUGH) {
			server->cursor_mode = CURSOR_PASSTHROUGH;
			server->grabbed_toplevel = NULL;
		}
		server->input_drag = false;
		server->card_text_drag = false;
		/* Single click with no drag → degenerate anchor, clear it */
		if (server->input.sel_anchor != -1 &&
				server->input.sel_anchor == server->input.sel_focus) {
			server->input.sel_anchor = -1;
		}
		/* Same for card text: degenerate selection → clear */
		if (server->lrenderer->card_sel.card_idx >= 0 &&
				server->lrenderer->card_sel.anchor ==
				server->lrenderer->card_sel.focus) {
			server->lrenderer->card_sel.card_idx = -1;
		}
	}
}

static void cursor_axis_handler(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, cursor_axis);
	struct wlr_pointer_axis_event *event = data;

	/* Scroll within expanded card overlay */
	if (server->feed->expanded_card >= 0 &&
			event->orientation == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pthread_mutex_lock(&server->feed->mutex);
		server->feed->expanded_scroll += event->delta;
		if (server->feed->expanded_scroll < 0)
			server->feed->expanded_scroll = 0;
		pthread_mutex_unlock(&server->feed->mutex);
		schedule_panel_redraw(server);
		return;
	}

	/* Scroll the panel feed when the cursor is over the desktop (not a window) */
	double scroll_sx, scroll_sy;
	struct wlr_surface *scroll_surface = NULL;
	struct marshal_toplevel *scroll_tl = toplevel_at(server,
		server->cursor->x, server->cursor->y,
		&scroll_surface, &scroll_sx, &scroll_sy);
	if (!scroll_tl &&
			event->orientation == WL_POINTER_AXIS_VERTICAL_SCROLL) {
		pthread_mutex_lock(&server->feed->mutex);
		/* Wayland delta: positive = scroll down (toward user).
		 * We want scroll-up (negative delta) to increase offset
		 * (reveal older cards above), so negate. */
		server->feed->scroll_offset -= event->delta;
		if (server->feed->scroll_offset < 0)
			server->feed->scroll_offset = 0;
		pthread_mutex_unlock(&server->feed->mutex);
		schedule_panel_redraw(server);
		return;
	}

	wlr_seat_pointer_notify_axis(server->seat, event->time_msec,
		event->orientation, event->delta, event->delta_discrete,
		event->source, event->relative_direction);
}

static void cursor_frame_handler(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, cursor_frame);
	wlr_seat_pointer_notify_frame(server->seat);
}

static void request_set_cursor_handler(struct wl_listener *listener,
		void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, request_set_cursor);
	struct wlr_seat_pointer_request_set_cursor_event *event = data;
	struct wlr_seat_client *focused_client =
		server->seat->pointer_state.focused_client;
	if (focused_client == event->seat_client) {
		wlr_cursor_set_surface(server->cursor, event->surface,
			event->hotspot_x, event->hotspot_y);
	}
}

/* ── New input device ── */

static void server_new_input(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_input);
	struct wlr_input_device *device = data;

	if (device->type == WLR_INPUT_DEVICE_KEYBOARD) {
		struct marshal_keyboard *keyboard = calloc(1, sizeof(*keyboard));
		keyboard->server = server;
		keyboard->wlr_keyboard = wlr_keyboard_from_input_device(device);

		struct xkb_context *context =
			xkb_context_new(XKB_CONTEXT_NO_FLAGS);
		struct xkb_keymap *keymap = xkb_keymap_new_from_names(context,
			NULL, XKB_KEYMAP_COMPILE_NO_FLAGS);
		wlr_keyboard_set_keymap(keyboard->wlr_keyboard, keymap);
		xkb_keymap_unref(keymap);
		xkb_context_unref(context);
		wlr_keyboard_set_repeat_info(keyboard->wlr_keyboard, 25, 600);

		keyboard->key.notify = keyboard_handle_key;
		wl_signal_add(&keyboard->wlr_keyboard->events.key,
			&keyboard->key);

		keyboard->modifiers.notify = keyboard_handle_modifiers;
		wl_signal_add(&keyboard->wlr_keyboard->events.modifiers,
			&keyboard->modifiers);

		keyboard->destroy.notify = keyboard_destroy;
		wl_signal_add(&device->events.destroy, &keyboard->destroy);

		wlr_seat_set_keyboard(server->seat, keyboard->wlr_keyboard);
	} else if (device->type == WLR_INPUT_DEVICE_POINTER) {
		wlr_cursor_attach_input_device(server->cursor, device);
	}

	/* Advertise seat capabilities to clients */
	uint32_t caps = WL_SEAT_CAPABILITY_POINTER;
	if (wlr_seat_get_keyboard(server->seat))
		caps |= WL_SEAT_CAPABILITY_KEYBOARD;
	wlr_seat_set_capabilities(server->seat, caps);
}

/* ── New output ── */

static void server_new_output(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_output);
	struct wlr_output *wlr_output = data;

	wlr_output_init_render(wlr_output, server->allocator, server->renderer);

	struct wlr_output_state state;
	wlr_output_state_init(&state);
	wlr_output_state_set_enabled(&state, true);

	struct wlr_output_mode *mode = wlr_output_preferred_mode(wlr_output);
	if (mode) {
		wlr_output_state_set_mode(&state, mode);
	}

	wlr_output_commit_state(wlr_output, &state);
	wlr_output_state_finish(&state);

	struct marshal_output *output = calloc(1, sizeof(*output));
	output->server = server;
	output->wlr_output = wlr_output;

	/* Create scene output */
	output->scene_output = wlr_scene_output_create(server->scene,
		wlr_output);
	struct wlr_output_layout_output *lo =
		wlr_output_layout_add_auto(server->output_layout, wlr_output);
	wlr_scene_output_layout_add_output(server->scene_layout, lo,
		output->scene_output);

	output->frame.notify = output_frame;
	wl_signal_add(&wlr_output->events.frame, &output->frame);

	output->request_state.notify = output_request_state;
	wl_signal_add(&wlr_output->events.request_state,
		&output->request_state);

	output->destroy.notify = output_destroy;
	wl_signal_add(&wlr_output->events.destroy, &output->destroy);

	wl_list_insert(&server->outputs, &output->link);

	/* Update layout for the new output dimensions */
	relayout_toplevels(server);
	schedule_panel_redraw(server);
}

/* ── Backend destroy ── */

static void backend_destroy_handler(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, backend_destroy);
	wl_display_terminate(server->display);
}

/* ── Wayland data-device: honour selection requests from clients ── */

static void handle_request_set_selection(struct wl_listener *listener,
		void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, request_set_selection);
	struct wlr_seat_request_set_selection_event *event = data;
	wlr_seat_set_selection(server->seat, event->source, event->serial);
}

/* ── XDG decoration: request server-side decorations ── */

static void decoration_handle_request_mode(struct wl_listener *listener,
		void *data) {
	struct wlr_xdg_toplevel_decoration_v1 *decoration = data;
	wlr_xdg_toplevel_decoration_v1_set_mode(decoration,
		WLR_XDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);
}

struct marshal_decoration {
	struct wl_listener request_mode;
	struct wl_listener destroy;
};

static void decoration_handle_destroy(struct wl_listener *listener,
		void *data) {
	struct marshal_decoration *deco =
		wl_container_of(listener, deco, destroy);
	wl_list_remove(&deco->request_mode.link);
	wl_list_remove(&deco->destroy.link);
	free(deco);
}

static void server_new_decoration(struct wl_listener *listener, void *data) {
	struct wlr_xdg_toplevel_decoration_v1 *decoration = data;

	/* Immediately request SSD and listen for future mode requests */
	wlr_xdg_toplevel_decoration_v1_set_mode(decoration,
		WLR_XDG_TOPLEVEL_DECORATION_V1_MODE_SERVER_SIDE);

	struct marshal_decoration *deco = calloc(1, sizeof(*deco));
	deco->request_mode.notify = decoration_handle_request_mode;
	wl_signal_add(&decoration->events.request_mode, &deco->request_mode);
	deco->destroy.notify = decoration_handle_destroy;
	wl_signal_add(&decoration->events.destroy, &deco->destroy);
}

/* ── Layer shell handlers ── */

static void layer_surface_map(struct wl_listener *listener, void *data) {
	struct marshal_layer_surface *ls =
		wl_container_of(listener, ls, map);
	wlr_scene_node_set_enabled(&ls->scene->tree->node, true);
}

static void layer_surface_unmap(struct wl_listener *listener, void *data) {
	struct marshal_layer_surface *ls =
		wl_container_of(listener, ls, unmap);
	wlr_scene_node_set_enabled(&ls->scene->tree->node, false);
}

static void layer_surface_commit(struct wl_listener *listener, void *data) {
	struct marshal_layer_surface *ls =
		wl_container_of(listener, ls, commit);
	if (ls->layer_surface->initial_commit) {
		/* Let wlroots arrange the layer surface on its output */
		struct marshal_output *out;
		wl_list_for_each(out, &ls->server->outputs, link) {
			struct wlr_output *wo = out->wlr_output;
			int ow, oh;
			wlr_output_effective_resolution(wo, &ow, &oh);
			struct wlr_box full = { .x = 0, .y = 0,
				.width = ow, .height = oh };
			wlr_scene_layer_surface_v1_configure(ls->scene, &full, &full);
			break;
		}
	}
}

static void layer_surface_destroy(struct wl_listener *listener, void *data) {
	struct marshal_layer_surface *ls =
		wl_container_of(listener, ls, destroy);
	wl_list_remove(&ls->map.link);
	wl_list_remove(&ls->unmap.link);
	wl_list_remove(&ls->commit.link);
	wl_list_remove(&ls->destroy.link);
	free(ls);
}

static void server_new_layer_surface(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_layer_surface);
	struct wlr_layer_surface_v1 *layer_surface = data;

	/* Assign to first output if client didn't specify */
	if (!layer_surface->output) {
		struct marshal_output *out;
		wl_list_for_each(out, &server->outputs, link) {
			layer_surface->output = out->wlr_output;
			break;
		}
		if (!layer_surface->output) {
			wlr_layer_surface_v1_destroy(layer_surface);
			return;
		}
	}

	enum zwlr_layer_shell_v1_layer layer = layer_surface->pending.layer;
	if (layer > ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY)
		layer = ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY;

	struct wlr_scene_tree *parent = server->layer_trees[layer];

	struct marshal_layer_surface *ls = calloc(1, sizeof(*ls));
	ls->kind = MARSHAL_SCENE_KIND_LAYER;
	ls->server = server;
	ls->layer_surface = layer_surface;
	ls->scene = wlr_scene_layer_surface_v1_create(parent, layer_surface);
	ls->scene->tree->node.data = ls;

	ls->map.notify = layer_surface_map;
	wl_signal_add(&layer_surface->surface->events.map, &ls->map);
	ls->unmap.notify = layer_surface_unmap;
	wl_signal_add(&layer_surface->surface->events.unmap, &ls->unmap);
	ls->commit.notify = layer_surface_commit;
	wl_signal_add(&layer_surface->surface->events.commit, &ls->commit);
	ls->destroy.notify = layer_surface_destroy;
	wl_signal_add(&layer_surface->events.destroy, &ls->destroy);
}

/* ── Text input / IME relay ── */

static void handle_text_input_enable(struct wl_listener *listener, void *data) {
	/* Forward enable to input method */
	struct wlr_text_input_v3 *text_input = data;
	struct marshal_server *server = text_input->seat->data;
	if (!server || !server->input_method) return;
	server->active_text_input = text_input;
	wlr_input_method_v2_send_activate(server->input_method);

	/* Forward content type and surrounding text */
	wlr_input_method_v2_send_content_type(server->input_method,
		text_input->current.content_type.hint,
		text_input->current.content_type.purpose);
	if (text_input->current.surrounding.text) {
		wlr_input_method_v2_send_surrounding_text(server->input_method,
			text_input->current.surrounding.text,
			text_input->current.surrounding.cursor,
			text_input->current.surrounding.anchor);
	}
	wlr_input_method_v2_send_done(server->input_method);
}

static void handle_text_input_commit(struct wl_listener *listener, void *data) {
	struct wlr_text_input_v3 *text_input = data;
	struct marshal_server *server = text_input->seat->data;
	if (!server || !server->input_method) return;
	if (server->active_text_input != text_input) return;

	/* Forward updated state to input method */
	if (text_input->current.surrounding.text) {
		wlr_input_method_v2_send_surrounding_text(server->input_method,
			text_input->current.surrounding.text,
			text_input->current.surrounding.cursor,
			text_input->current.surrounding.anchor);
	}
	wlr_input_method_v2_send_content_type(server->input_method,
		text_input->current.content_type.hint,
		text_input->current.content_type.purpose);
	wlr_input_method_v2_send_done(server->input_method);
}

static void handle_text_input_disable(struct wl_listener *listener,
		void *data) {
	struct wlr_text_input_v3 *text_input = data;
	struct marshal_server *server = text_input->seat->data;
	if (!server || !server->input_method) return;
	if (server->active_text_input == text_input) {
		wlr_input_method_v2_send_deactivate(server->input_method);
		wlr_input_method_v2_send_done(server->input_method);
		server->active_text_input = NULL;
	}
}

static void handle_text_input_destroy(struct wl_listener *listener,
		void *data) {
	struct wlr_text_input_v3 *text_input = data;
	struct marshal_server *server = text_input->seat->data;
	if (!server) return;
	if (server->active_text_input == text_input)
		server->active_text_input = NULL;
}

struct marshal_text_input {
	struct wl_listener enable;
	struct wl_listener commit;
	struct wl_listener disable;
	struct wl_listener destroy;
};

static void server_new_text_input(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_text_input);
	struct wlr_text_input_v3 *text_input = data;

	struct marshal_text_input *ti = calloc(1, sizeof(*ti));
	if (!ti) return;

	/* Store server pointer in seat->data for the per-input-event callbacks */
	server->seat->data = server;

	ti->enable.notify = handle_text_input_enable;
	wl_signal_add(&text_input->events.enable, &ti->enable);
	ti->commit.notify = handle_text_input_commit;
	wl_signal_add(&text_input->events.commit, &ti->commit);
	ti->disable.notify = handle_text_input_disable;
	wl_signal_add(&text_input->events.disable, &ti->disable);
	ti->destroy.notify = handle_text_input_destroy;
	wl_signal_add(&text_input->events.destroy, &ti->destroy);
}

/* Input method commit: forward committed string to text input */
static void handle_input_method_commit(struct wl_listener *listener,
		void *data) {
	struct wlr_input_method_v2 *im = data;
	struct marshal_server *server = im->seat->data;
	if (!server || !server->active_text_input) return;

	struct wlr_text_input_v3 *ti = server->active_text_input;
	if (im->current.commit_text) {
		wlr_text_input_v3_send_commit_string(ti,
			im->current.commit_text);
	}
	if (im->current.preedit.text) {
		wlr_text_input_v3_send_preedit_string(ti,
			im->current.preedit.text,
			im->current.preedit.cursor_begin,
			im->current.preedit.cursor_end);
	}
	if (im->current.delete.before_length || im->current.delete.after_length) {
		wlr_text_input_v3_send_delete_surrounding_text(ti,
			im->current.delete.before_length,
			im->current.delete.after_length);
	}
	wlr_text_input_v3_send_done(ti);
}

static void handle_input_method_destroy(struct wl_listener *listener,
		void *data) {
	struct wlr_input_method_v2 *im = data;
	struct marshal_server *server = im->seat->data;
	if (server) server->input_method = NULL;
}

struct marshal_input_method {
	struct wl_listener commit;
	struct wl_listener destroy;
};

static void server_new_input_method(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_input_method);
	struct wlr_input_method_v2 *im = data;

	/* Only one input method at a time */
	if (server->input_method) {
		wlr_input_method_v2_send_unavailable(im);
		return;
	}
	server->input_method = im;
	server->seat->data = server;

	struct marshal_input_method *lim = calloc(1, sizeof(*lim));
	if (!lim) return;

	lim->commit.notify = handle_input_method_commit;
	wl_signal_add(&im->events.commit, &lim->commit);
	lim->destroy.notify = handle_input_method_destroy;
	wl_signal_add(&im->events.destroy, &lim->destroy);
}

/* ── XWayland handlers ── */

static void update_xwayland_decorations(struct marshal_xwayland_surface *xs);

static void xwayland_surface_map(struct wl_listener *listener, void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, map);
	struct marshal_server *server = xs->server;

	bool is_or = xs->xsurface->override_redirect;

	/* Only real toplevels go in the toplevel list (used by the bar-icon
	 * snapshot, focus cycling, and map/unmap focus chain). OR popups
	 * are children — including them would put a duplicate "App" icon
	 * on the bar for every Qt menu the user opens. */
	if (!is_or)
		wl_list_insert(&server->toplevels, &xs->link);

	wlr_scene_node_set_position(&xs->frame_tree->node, xs->x, xs->y);
	if (!is_or)
		update_xwayland_decorations(xs);

	emit_window_event(server, "window_opened",
		xs->xsurface->class, xs->xsurface->title,
		xs->xsurface->pid, server->active_workspace);

	wlr_scene_node_raise_to_top(&xs->frame_tree->node);

	if (!is_or) {
		/* Focus the new X11 window */
		server->focus_mode = FOCUS_APP;
		struct wlr_keyboard *keyboard = wlr_seat_get_keyboard(server->seat);
		if (keyboard) {
			wlr_seat_keyboard_notify_enter(server->seat,
				xs->xsurface->surface,
				keyboard->keycodes, keyboard->num_keycodes,
				&keyboard->modifiers);
		}
		wlr_xwayland_surface_activate(xs->xsurface, true);
	}
	schedule_panel_redraw(server);
}

static void xwayland_surface_unmap(struct wl_listener *listener, void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, unmap);
	struct marshal_server *server = xs->server;

	emit_window_event(server, "window_closed",
		xs->xsurface->class, "", xs->xsurface->pid,
		xs->workspace);

	/* Symmetric with map: only real toplevels were inserted into the list. */
	if (!xs->xsurface->override_redirect)
		wl_list_remove(&xs->link);

	if (!wl_list_empty(&server->toplevels)) {
		/* Focus next toplevel (could be xdg or xwayland) */
		struct marshal_toplevel *next = wl_container_of(
			server->toplevels.next, next, link);
		focus_toplevel(server, next);
	} else {
		server->focus_mode = FOCUS_NONE;
		wlr_seat_keyboard_clear_focus(server->seat);
		schedule_panel_redraw(server);
	}
}

static void xwayland_surface_destroy(struct wl_listener *listener, void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, destroy);

	wl_list_remove(&xs->map.link);
	wl_list_remove(&xs->unmap.link);
	wl_list_remove(&xs->destroy.link);
	wl_list_remove(&xs->request_configure.link);
	wl_list_remove(&xs->request_maximize.link);
	wl_list_remove(&xs->request_fullscreen.link);
	wl_list_remove(&xs->set_geometry.link);

	if (xs->server->grabbed_toplevel ==
			(struct marshal_toplevel *)xs) {
		xs->server->cursor_mode = CURSOR_PASSTHROUGH;
		xs->server->grabbed_toplevel = NULL;
	}

	if (xs->scene_tree) {
		wlr_scene_node_reparent(&xs->scene_tree->node,
			xs->server->app_tree);
		wlr_scene_node_set_enabled(&xs->scene_tree->node, false);
	}
	if (xs->frame_tree)
		wlr_scene_node_destroy(&xs->frame_tree->node);

	free(xs);
}

static void xwayland_surface_request_configure(struct wl_listener *listener,
		void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, request_configure);
	struct wlr_xwayland_surface_configure_event *ev = data;
	wlr_xwayland_surface_configure(xs->xsurface,
		ev->x, ev->y, ev->width, ev->height);
	xs->x = ev->x;
	xs->y = ev->y;
	xs->width = ev->width;
	xs->height = ev->height;
	wlr_scene_node_set_position(&xs->frame_tree->node, xs->x, xs->y);
	if (!xs->xsurface->override_redirect)
		update_xwayland_decorations(xs);
}

static void xwayland_surface_request_maximize(struct wl_listener *listener,
		void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, request_maximize);
	struct marshal_server *server = xs->server;
	int ow = output_width(server);
	int oh = output_height(server);

	if (!xs->maximized) {
		xs->saved_x = xs->x; xs->saved_y = xs->y;
		xs->saved_w = xs->width; xs->saved_h = xs->height;
		xs->maximized = true;
		xs->x = 0; xs->y = 0;
		xs->width = ow;
		xs->height = oh - INPUT_HEIGHT - TITLEBAR_H;
	} else {
		xs->maximized = false;
		xs->x = xs->saved_x; xs->y = xs->saved_y;
		xs->width = xs->saved_w; xs->height = xs->saved_h;
	}
	wlr_xwayland_surface_configure(xs->xsurface,
		xs->x, xs->y + TITLEBAR_H, xs->width, xs->height);
	wlr_scene_node_set_position(&xs->frame_tree->node, xs->x, xs->y);
	update_xwayland_decorations(xs);
	schedule_panel_redraw(server);
}

static void xwayland_surface_request_fullscreen(struct wl_listener *listener,
		void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, request_fullscreen);
	if (!xs->maximized)
		xwayland_surface_request_maximize(listener, data);
}

static void xwayland_surface_set_geometry(struct wl_listener *listener,
		void *data) {
	struct marshal_xwayland_surface *xs =
		wl_container_of(listener, xs, set_geometry);
	(void)data;
	/* For OR popups, the X server has just told us where the surface
	 * wants to be. Push that into our scene tree so a click on a Qt
	 * "File" button opens the dropdown UNDER the button instead of at
	 * stale coordinates. */
	if (xs->xsurface->override_redirect) {
		xs->x = xs->xsurface->x;
		xs->y = xs->xsurface->y;
		xs->width  = xs->xsurface->width;
		xs->height = xs->xsurface->height;
		wlr_scene_node_set_position(&xs->frame_tree->node,
			xs->x, xs->y);
		return;
	}
	update_xwayland_decorations(xs);
}

static void update_xwayland_decorations(struct marshal_xwayland_surface *xs) {
	if (!xs->titlebar_bg) return;  /* OR popup — no chrome */
	int w = xs->width > 0 ? xs->width : 800;
	wlr_scene_rect_set_size(xs->titlebar_bg, w, TITLEBAR_H);
	const int bsz = MARSHAL_BTN_PX;
	const int gap = 6;
	const int margin = 10;
	int btn_y = (TITLEBAR_H - bsz) / 2;
	wlr_scene_node_set_position(&xs->btn_close->node,
		w - bsz - margin, btn_y);
	wlr_scene_node_set_position(&xs->btn_max->node,
		w - 2 * bsz - margin - gap, btn_y);
	wlr_scene_node_set_position(&xs->btn_min->node,
		w - 3 * bsz - margin - 2 * gap, btn_y);
}

static void server_xwayland_new_surface(struct wl_listener *listener,
		void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, xwayland_new_surface);
	struct wlr_xwayland_surface *xsurface = data;

	struct marshal_xwayland_surface *xs = calloc(1, sizeof(*xs));
	xs->kind = MARSHAL_SCENE_KIND_XWAYLAND;
	xs->server = server;
	xs->xsurface = xsurface;

	/* OR (override_redirect) and modal/transient dialogs from Qt apps —
	 * menus, tooltips, dropdowns, the OBS screen-picker — must NOT be
	 * decorated and must use the X server's exact placement. Wrapping
	 * them in our SSD frame and re-centering them was why OBS's "File"
	 * dropdown looked detached, OR popups landed in the wrong place,
	 * and Qt menus were unclickable. Skip every UI hack: build a flat
	 * scene-subsurface-tree at the X server's coordinates and let the
	 * client drive positioning. */
	bool is_or = xsurface->override_redirect;

	if (is_or) {
		xs->frame_tree = wlr_scene_tree_create(server->app_tree);
		xs->frame_tree->node.data = xs;
		xs->scene_tree = wlr_scene_subsurface_tree_create(
			xs->frame_tree, xsurface->surface);
		xs->scene_tree->node.data = xs;
		/* OR surfaces aren't toplevels — they don't get a title bar,
		 * close/max/min buttons, or focus management. They follow the
		 * X-side coordinates exactly. */
		xs->titlebar_bg = NULL;
		xs->btn_close   = NULL;
		xs->btn_max     = NULL;
		xs->btn_min     = NULL;
		xs->x = xsurface->x;
		xs->y = xsurface->y;
		xs->width  = xsurface->width;
		xs->height = xsurface->height;
		wlr_scene_node_set_position(&xs->frame_tree->node,
			xs->x, xs->y);
	} else {
		/* Regular toplevel: SSD frame, default centred placement. */
		xs->frame_tree = wlr_scene_tree_create(server->app_tree);
		xs->frame_tree->node.data = xs;

		float tb_color[4] = {0.898f, 0.898f, 0.898f, 1.0f};
		xs->titlebar_bg = wlr_scene_rect_create(
			xs->frame_tree, 800, TITLEBAR_H, tb_color);

		xs->btn_close = create_ps_button(xs->frame_tree,
			MARSHAL_PS_CIRCLE);
		xs->btn_max = create_ps_button(xs->frame_tree,
			MARSHAL_PS_TRIANGLE);
		xs->btn_min = create_ps_button(xs->frame_tree,
			MARSHAL_PS_CROSS);

		xs->scene_tree = wlr_scene_subsurface_tree_create(
			xs->frame_tree, xsurface->surface);
		wlr_scene_node_set_position(&xs->scene_tree->node, 0, TITLEBAR_H);
		xs->scene_tree->node.data = xs;

		int ow = output_width(server);
		int oh = output_height(server);
		int usable_h = oh - INPUT_HEIGHT;
		xs->width  = xsurface->width > 0 ? xsurface->width : ow * 7 / 10;
		xs->height = xsurface->height > 0 ? xsurface->height : usable_h * 7 / 10;
		xs->x = (ow - xs->width) / 2;
		xs->y = (usable_h - xs->height - TITLEBAR_H) / 2;
	}

	xs->map.notify = xwayland_surface_map;
	wl_signal_add(&xsurface->surface->events.map, &xs->map);
	xs->unmap.notify = xwayland_surface_unmap;
	wl_signal_add(&xsurface->surface->events.unmap, &xs->unmap);
	xs->destroy.notify = xwayland_surface_destroy;
	wl_signal_add(&xsurface->events.destroy, &xs->destroy);
	xs->request_configure.notify = xwayland_surface_request_configure;
	wl_signal_add(&xsurface->events.request_configure,
		&xs->request_configure);
	xs->request_maximize.notify = xwayland_surface_request_maximize;
	wl_signal_add(&xsurface->events.request_maximize,
		&xs->request_maximize);
	xs->request_fullscreen.notify = xwayland_surface_request_fullscreen;
	wl_signal_add(&xsurface->events.request_fullscreen,
		&xs->request_fullscreen);
	xs->set_geometry.notify = xwayland_surface_set_geometry;
	wl_signal_add(&xsurface->events.set_geometry, &xs->set_geometry);
}

static void server_xwayland_ready(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, xwayland_ready);
	/* Set DISPLAY so child processes can find the XWayland socket */
	if (server->xwayland) {
		setenv("DISPLAY", server->xwayland->display_name, 1);
		fprintf(stderr, "XWayland ready on %s\n",
			server->xwayland->display_name);
	}
}

/* ── Session lock (ext-session-lock-v1) ── */

struct marshal_lock_surface {
	struct marshal_server *server;
	struct wlr_session_lock_surface_v1 *lock_surface;
	struct wlr_scene_tree *scene_tree;
	struct wl_listener map;
	struct wl_listener destroy;
	struct wl_listener surface_commit;
};

static void lock_surface_configure(struct marshal_lock_surface *ls) {
	struct wlr_output *output = ls->lock_surface->output;
	wlr_session_lock_surface_v1_configure(ls->lock_surface,
		output->width, output->height);
}

static void lock_surface_handle_map(struct wl_listener *listener, void *data) {
	(void)data;
	struct marshal_lock_surface *ls = wl_container_of(listener, ls, map);
	/* Check if all outputs have a mapped lock surface — if so, send locked */
	struct marshal_server *server = ls->server;
	if (server->active_session_lock && !server->locked) {
		wlr_session_lock_v1_send_locked(server->active_session_lock);
		server->locked = true;
	}
}

static void lock_surface_handle_destroy(struct wl_listener *listener,
		void *data) {
	(void)data;
	struct marshal_lock_surface *ls = wl_container_of(listener, ls, destroy);
	wl_list_remove(&ls->map.link);
	wl_list_remove(&ls->destroy.link);
	wl_list_remove(&ls->surface_commit.link);
	free(ls);
}

static void lock_surface_handle_commit(struct wl_listener *listener,
		void *data) {
	(void)data;
	struct marshal_lock_surface *ls =
		wl_container_of(listener, ls, surface_commit);
	if (!ls->lock_surface->configured)
		return;
}

static void handle_session_lock_new_surface(struct wl_listener *listener,
		void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, session_lock_new_surface);
	struct wlr_session_lock_surface_v1 *lock_surface = data;

	struct marshal_lock_surface *ls = calloc(1, sizeof(*ls));
	if (!ls) return;

	ls->server = server;
	ls->lock_surface = lock_surface;
	ls->scene_tree = wlr_scene_subsurface_tree_create(server->lock_tree,
		lock_surface->surface);

	ls->map.notify = lock_surface_handle_map;
	wl_signal_add(&lock_surface->surface->events.map, &ls->map);
	ls->destroy.notify = lock_surface_handle_destroy;
	wl_signal_add(&lock_surface->events.destroy, &ls->destroy);
	ls->surface_commit.notify = lock_surface_handle_commit;
	wl_signal_add(&lock_surface->surface->events.commit,
		&ls->surface_commit);

	lock_surface_configure(ls);
}

static void handle_session_lock_unlock(struct wl_listener *listener,
		void *data) {
	(void)data;
	struct marshal_server *server =
		wl_container_of(listener, server, session_lock_unlock);
	wl_list_remove(&server->session_lock_new_surface.link);
	wl_list_remove(&server->session_lock_unlock.link);
	wl_list_remove(&server->session_lock_destroy.link);
	server->active_session_lock = NULL;
	server->locked = false;
	wlr_scene_node_set_enabled(&server->lock_tree->node, false);
}

static void handle_session_lock_destroy(struct wl_listener *listener,
		void *data) {
	(void)data;
	struct marshal_server *server =
		wl_container_of(listener, server, session_lock_destroy);
	wl_list_remove(&server->session_lock_new_surface.link);
	wl_list_remove(&server->session_lock_unlock.link);
	wl_list_remove(&server->session_lock_destroy.link);
	server->active_session_lock = NULL;
	/* If client crashed while locked, scene stays up to prevent peek-through.
	 * User recovers via VT switch.  Clearing locked here so a new locker
	 * can reconnect. */
	server->locked = false;
	wlr_scene_node_set_enabled(&server->lock_tree->node, false);
}

static void handle_new_session_lock(struct wl_listener *listener, void *data) {
	struct marshal_server *server =
		wl_container_of(listener, server, new_session_lock);
	struct wlr_session_lock_v1 *lock = data;

	if (server->active_session_lock) {
		wlr_session_lock_v1_destroy(lock);
		return;
	}

	server->active_session_lock = lock;
	server->locked = false;

	/* Show the lock layer above everything */
	wlr_scene_node_set_enabled(&server->lock_tree->node, true);

	server->session_lock_new_surface.notify =
		handle_session_lock_new_surface;
	wl_signal_add(&lock->events.new_surface,
		&server->session_lock_new_surface);
	server->session_lock_unlock.notify = handle_session_lock_unlock;
	wl_signal_add(&lock->events.unlock, &server->session_lock_unlock);
	server->session_lock_destroy.notify = handle_session_lock_destroy;
	wl_signal_add(&lock->events.destroy, &server->session_lock_destroy);
}

/* ── Event broadcast (compositor → agentd) ── */

static int event_accept_cb(int fd, uint32_t mask, void *data) {
	(void)mask;
	struct marshal_server *server = data;
	int client = accept4(fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
	if (client < 0) return 0;
	if (server->event_client_count < 8)
		server->event_clients[server->event_client_count++] = client;
	else
		close(client);
	return 0;
}

/* ── agentd → compositor proactive push ── */

static void proactive_client_drop(struct marshal_server *server, int slot) {
	if (slot < 0 || slot >= server->proactive_client_count) return;
	if (server->proactive_clients[slot].src)
		wl_event_source_remove(server->proactive_clients[slot].src);
	if (server->proactive_clients[slot].fd >= 0)
		close(server->proactive_clients[slot].fd);
	int last = --server->proactive_client_count;
	if (slot != last)
		server->proactive_clients[slot] = server->proactive_clients[last];
	memset(&server->proactive_clients[last], 0,
		sizeof(server->proactive_clients[last]));
	server->proactive_clients[last].fd = -1;
}

static int proactive_read_cb(int fd, uint32_t mask, void *data) {
	struct marshal_server *server = data;
	int slot = -1;
	for (int i = 0; i < server->proactive_client_count; i++) {
		if (server->proactive_clients[i].fd == fd) { slot = i; break; }
	}
	if (slot < 0) return 0;

	if (mask & (WL_EVENT_HANGUP | WL_EVENT_ERROR)) {
		proactive_client_drop(server, slot);
		return 0;
	}

	char *buf = server->proactive_clients[slot].buf;
	size_t *blen = &server->proactive_clients[slot].buf_len;
	size_t cap = sizeof(server->proactive_clients[slot].buf);

	ssize_t n = read(fd, buf + *blen, cap - 1 - *blen);
	if (n == 0) { proactive_client_drop(server, slot); return 0; }
	if (n < 0) {
		if (errno == EAGAIN || errno == EINTR) return 0;
		proactive_client_drop(server, slot);
		return 0;
	}
	*blen += (size_t)n;
	buf[*blen] = '\0';

	/* Extract complete newline-delimited frames. */
	for (;;) {
		char *nl = memchr(buf, '\n', *blen);
		if (!nl) {
			/* Oversized frame with no newline — drop to resync. */
			if (*blen >= cap - 1) *blen = 0;
			break;
		}
		*nl = '\0';
		if (server->feed)
			feed_insert_proactive(server->feed, buf);
		size_t frame_len = (size_t)(nl - buf) + 1;
		size_t remaining = *blen - frame_len;
		memmove(buf, nl + 1, remaining);
		*blen = remaining;
		buf[*blen] = '\0';
	}
	return 0;
}

static int proactive_accept_cb(int fd, uint32_t mask __attribute__((unused)),
		void *data) {
	struct marshal_server *server = data;
	int client = accept4(fd, NULL, NULL, SOCK_NONBLOCK | SOCK_CLOEXEC);
	if (client < 0) return 0;

	if (server->proactive_client_count >=
			(int)(sizeof(server->proactive_clients) /
				sizeof(server->proactive_clients[0]))) {
		close(client);
		return 0;
	}

	int slot = server->proactive_client_count++;
	server->proactive_clients[slot].fd = client;
	server->proactive_clients[slot].buf_len = 0;
	server->proactive_clients[slot].src = wl_event_loop_add_fd(
		server->event_loop, client,
		WL_EVENT_READABLE, proactive_read_cb, server);
	if (!server->proactive_clients[slot].src) {
		close(client);
		server->proactive_client_count--;
	}
	return 0;
}

static void emit_event(struct marshal_server *server, const char *json) {
	size_t len = strlen(json);
	for (int i = 0; i < server->event_client_count; ) {
		ssize_t n = write(server->event_clients[i], json, len);
		if (n < 0 && (errno == EPIPE || errno == ECONNRESET ||
				errno == EBADF)) {
			close(server->event_clients[i]);
			server->event_clients[i] =
				server->event_clients[--server->event_client_count];
		} else {
			/* EAGAIN/EWOULDBLOCK: drop event for slow client */
			i++;
		}
	}
}

static void emit_window_event(struct marshal_server *server,
		const char *type, const char *app_id, const char *title,
		pid_t pid, int workspace) {
	cJSON *obj = cJSON_CreateObject();
	cJSON_AddStringToObject(obj, "type", type);
	cJSON_AddStringToObject(obj, "app_id", app_id ? app_id : "");
	cJSON_AddStringToObject(obj, "title", title ? title : "");
	if (pid > 0)
		cJSON_AddNumberToObject(obj, "pid", (double)pid);
	cJSON_AddNumberToObject(obj, "workspace", workspace);
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	double ms = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
	cJSON_AddNumberToObject(obj, "ts_ms", ms);

	char *s = cJSON_PrintUnformatted(obj);
	if (s) {
		size_t slen = strlen(s);
		char *line = malloc(slen + 2);
		if (line) {
			memcpy(line, s, slen);
			line[slen] = '\n';
			line[slen + 1] = '\0';
			emit_event(server, line);
			free(line);
		}
		cJSON_free(s);
	}
	cJSON_Delete(obj);
}

/* ── Child process tracking ── */

static pid_t launch_subprocess_tracked(struct marshal_server *server,
		const char *app_id, const char *path, char *const argv[]) {
	pid_t pid = fork();
	if (pid == 0) {
		int maxfd = sysconf(_SC_OPEN_MAX);
		for (int fd = 3; fd < maxfd && fd < 1024; fd++)
			close(fd);
		execvp(path, argv);
		_exit(127);
	}
	if (pid > 0 && server->child_pid_count < 64) {
		int idx = server->child_pid_count++;
		server->child_pids[idx].pid = pid;
		strncpy(server->child_pids[idx].app_id, app_id,
			sizeof(server->child_pids[idx].app_id) - 1);
		server->child_pids[idx].app_id[63] = '\0';
	}
	return pid;
}

/* ── SIGCHLD handler: reap zombies + emit child_exited events ── */

static int sigchld_handler(int signal_number, void *data) {
	(void)signal_number;
	struct marshal_server *server = data;
	int status;
	pid_t pid;
	while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
		int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
		const char *app_id = "";
		for (int i = 0; i < server->child_pid_count; i++) {
			if (server->child_pids[i].pid == pid) {
				app_id = server->child_pids[i].app_id;
				/* Emit event before removing from table */
				if (strlen(app_id) > 0) {
					cJSON *obj = cJSON_CreateObject();
					cJSON_AddStringToObject(obj, "type",
						"child_exited");
					cJSON_AddStringToObject(obj, "app_id",
						app_id);
					cJSON_AddNumberToObject(obj, "pid",
						(double)pid);
					cJSON_AddNumberToObject(obj, "exit_code",
						(double)exit_code);
					struct timespec ts;
					clock_gettime(CLOCK_MONOTONIC, &ts);
					cJSON_AddNumberToObject(obj, "ts_ms",
						ts.tv_sec * 1000.0 +
						ts.tv_nsec / 1e6);
					char *s = cJSON_PrintUnformatted(obj);
					if (s) {
						size_t slen = strlen(s);
						char *line = malloc(slen + 2);
						if (line) {
							memcpy(line, s, slen);
							line[slen] = '\n';
							line[slen + 1] = '\0';
							emit_event(server, line);
							free(line);
						}
						cJSON_free(s);
					}
					cJSON_Delete(obj);
				}
				/* Remove from table */
				server->child_pids[i] =
					server->child_pids[
						--server->child_pid_count];
				break;
			}
		}
	}
	return 0;
}

/* ── Main ── */

int main(int argc, char *argv[]) {
	wlr_log_init(WLR_DEBUG, NULL);

	struct marshal_server server = {0};
	wl_list_init(&server.outputs);
	wl_list_init(&server.toplevels);
	server.focus_mode = FOCUS_NONE;

	server.display = wl_display_create();
	server.event_loop = wl_display_get_event_loop(server.display);

	server.backend = wlr_backend_autocreate(server.event_loop,
		&server.session);
	if (!server.backend) {
		fprintf(stderr, "Failed to create wlr_backend\n");
		return 1;
	}

	server.renderer = wlr_renderer_autocreate(server.backend);
	if (!server.renderer) {
		fprintf(stderr, "Failed to create wlr_renderer\n");
		return 1;
	}
	wlr_renderer_init_wl_display(server.renderer, server.display);

	server.allocator = wlr_allocator_autocreate(server.backend,
		server.renderer);
	if (!server.allocator) {
		fprintf(stderr, "Failed to create wlr_allocator\n");
		return 1;
	}

	/* Wayland compositor globals — required for clients.
	 *
	 * wl_subcompositor must be advertised separately from wl_compositor —
	 * wlroots does not auto-create it. Without it, Qt-Wayland clients
	 * (OBS Studio, Qt apps in general) can't compose popups/tooltips and
	 * log "Can't create subsurface, not supported by the compositor".
	 * The screen-picker portal dialog renders garbled until this is set. */
	server.compositor =
		wlr_compositor_create(server.display, 6, server.renderer);
	wlr_subcompositor_create(server.display);
	wlr_data_device_manager_create(server.display);

	/* Output layout */
	server.output_layout = wlr_output_layout_create(server.display);

	/* Scene graph */
	server.scene = wlr_scene_create();
	server.scene_layout = wlr_scene_attach_output_layout(server.scene,
		server.output_layout);

	/* Scene tree structure (z-order, bottom to top):
	 *   scene root
	 *     ├── layer_background  (desktop widgets: background layer)
	 *     ├── panel_bg          (full screen, desktop background)
	 *     ├── panel_tree        → panel_scene_buf (Cairo: wallpaper + feed)
	 *     ├── layer_bottom      (layer shell bottom)
	 *     ├── app_tree          (floating windows with SSD title bars)
	 *     │   └── [frame_tree per window: titlebar rects + xdg/xwayland surface]
	 *     ├── taskbar_tree      (full-width taskbar, above app windows)
	 *     │   ├── taskbar_bg
	 *     │   └── taskbar_scene_buf
	 *     ├── layer_top         (notifications, screen keyboard)
	 *     └── layer_overlay     (OSDs, auth dialogs, lock screen)
	 */

	/* Layer shell: background */
	server.layer_trees[ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND] =
		wlr_scene_tree_create(&server.scene->tree);

	float panel_bg_color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
	server.panel_bg = wlr_scene_rect_create(&server.scene->tree,
		1920, 1080, panel_bg_color);

	server.panel_tree = wlr_scene_tree_create(&server.scene->tree);
	server.panel_scene_buf = wlr_scene_buffer_create(server.panel_tree,
		NULL);

	/* Layer shell: bottom */
	server.layer_trees[ZWLR_LAYER_SHELL_V1_LAYER_BOTTOM] =
		wlr_scene_tree_create(&server.scene->tree);

	server.app_tree = wlr_scene_tree_create(&server.scene->tree);
	wlr_scene_node_set_position(&server.app_tree->node, 0, 0);

	/* Taskbar overlay — full width at bottom, above app windows */
	server.taskbar_tree = wlr_scene_tree_create(&server.scene->tree);
	float taskbar_bg_color[4] = {0.96f, 0.96f, 0.96f, 1.0f};
	server.taskbar_bg = wlr_scene_rect_create(server.taskbar_tree,
		1920, INPUT_HEIGHT, taskbar_bg_color);
	server.taskbar_scene_buf = wlr_scene_buffer_create(
		server.taskbar_tree, NULL);

	/* Layer shell: top and overlay (above taskbar) */
	server.layer_trees[ZWLR_LAYER_SHELL_V1_LAYER_TOP] =
		wlr_scene_tree_create(&server.scene->tree);
	server.layer_trees[ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY] =
		wlr_scene_tree_create(&server.scene->tree);

	/* Session lock: above overlay, hidden until a lock client connects */
	server.lock_tree = wlr_scene_tree_create(&server.scene->tree);
	wlr_scene_node_set_enabled(&server.lock_tree->node, false);

	/* ext-session-lock-v1: allows marshal-locker to inhibit input */
	server.session_lock_mgr =
		wlr_session_lock_manager_v1_create(server.display);
	server.new_session_lock.notify = handle_new_session_lock;
	wl_signal_add(&server.session_lock_mgr->events.new_lock,
		&server.new_session_lock);

	/* xdg-shell */
	server.xdg_shell = wlr_xdg_shell_create(server.display, 6);
	server.new_xdg_toplevel.notify = server_new_xdg_toplevel;
	wl_signal_add(&server.xdg_shell->events.new_toplevel,
		&server.new_xdg_toplevel);
	server.new_xdg_popup.notify = server_new_xdg_popup;
	wl_signal_add(&server.xdg_shell->events.new_popup,
		&server.new_xdg_popup);

	/* xdg-decoration: tell clients we handle titlebars (SSD) */
	server.decoration_mgr =
		wlr_xdg_decoration_manager_v1_create(server.display);
	server.new_decoration.notify = server_new_decoration;
	wl_signal_add(&server.decoration_mgr->events.new_toplevel_decoration,
		&server.new_decoration);

	/* presentation-time: frame timing for video players / games */
	server.presentation = wlr_presentation_create(server.display,
		server.backend);

	/* screencopy: screenshots and screen sharing */
	server.screencopy_mgr =
		wlr_screencopy_manager_v1_create(server.display);

	/* xdg-output: logical output info for HiDPI */
	server.xdg_output_mgr = wlr_xdg_output_manager_v1_create(
		server.display, server.output_layout);

	/* fractional-scale: HiDPI fractional scaling */
	server.fractional_scale_mgr =
		wlr_fractional_scale_manager_v1_create(server.display, 1);

	/* layer-shell: notifications, lock screen, overlays, desktop widgets */
	server.layer_shell = wlr_layer_shell_v1_create(server.display, 4);
	server.new_layer_surface.notify = server_new_layer_surface;
	wl_signal_add(&server.layer_shell->events.new_surface,
		&server.new_layer_surface);

	/* text-input + input-method: IME support for non-Latin scripts */
	server.text_input_mgr =
		wlr_text_input_manager_v3_create(server.display);
	server.new_text_input.notify = server_new_text_input;
	wl_signal_add(&server.text_input_mgr->events.text_input,
		&server.new_text_input);

	server.input_method_mgr =
		wlr_input_method_manager_v2_create(server.display);
	server.new_input_method.notify = server_new_input_method;
	wl_signal_add(&server.input_method_mgr->events.input_method,
		&server.new_input_method);

	/* XWayland: X11 application compatibility */
	server.xwayland = wlr_xwayland_create(server.display,
		server.compositor, true);
	if (server.xwayland) {
		server.xwayland_ready.notify = server_xwayland_ready;
		wl_signal_add(&server.xwayland->events.ready,
			&server.xwayland_ready);
		server.xwayland_new_surface.notify = server_xwayland_new_surface;
		wl_signal_add(&server.xwayland->events.new_surface,
			&server.xwayland_new_surface);
	} else {
		fprintf(stderr, "Warning: XWayland not available\n");
	}

	/* Cursor */
	server.cursor = wlr_cursor_create();
	wlr_cursor_attach_output_layout(server.cursor, server.output_layout);

	server.cursor_mgr = wlr_xcursor_manager_create(NULL, 24);
	wlr_xcursor_manager_load(server.cursor_mgr, 1.0);
	wlr_cursor_set_xcursor(server.cursor, server.cursor_mgr, "default");

	server.cursor_motion.notify = cursor_motion_handler;
	wl_signal_add(&server.cursor->events.motion, &server.cursor_motion);

	server.cursor_motion_absolute.notify = cursor_motion_absolute_handler;
	wl_signal_add(&server.cursor->events.motion_absolute,
		&server.cursor_motion_absolute);

	server.cursor_button.notify = cursor_button_handler;
	wl_signal_add(&server.cursor->events.button, &server.cursor_button);

	server.cursor_axis.notify = cursor_axis_handler;
	wl_signal_add(&server.cursor->events.axis, &server.cursor_axis);

	server.cursor_frame.notify = cursor_frame_handler;
	wl_signal_add(&server.cursor->events.frame, &server.cursor_frame);

	/* Seat */
	server.seat = wlr_seat_create(server.display, "seat0");
	server.request_set_cursor.notify = request_set_cursor_handler;
	wl_signal_add(&server.seat->events.request_set_cursor,
		&server.request_set_cursor);
	server.request_set_selection.notify = handle_request_set_selection;
	wl_signal_add(&server.seat->events.request_set_selection,
		&server.request_set_selection);

	/* Primary selection: middle-click paste for X11 and Wayland apps */
	wlr_primary_selection_v1_device_manager_create(server.display);

	/* Tell XWayland about the seat so X11 apps receive keyboard input */
	if (server.xwayland)
		wlr_xwayland_set_seat(server.xwayland, server.seat);

	/* Input devices */
	server.new_output.notify = server_new_output;
	wl_signal_add(&server.backend->events.new_output, &server.new_output);

	server.new_input.notify = server_new_input;
	wl_signal_add(&server.backend->events.new_input, &server.new_input);

	server.backend_destroy.notify = backend_destroy_handler;
	wl_signal_add(&server.backend->events.destroy, &server.backend_destroy);

	/* Feed + Input + Renderer + Status */
	const char *api_url = getenv("MARSHAL_API_URL");
	if (!api_url) api_url = "http://127.0.0.1:8765";
	server.feed = feed_create(api_url);
	input_init(&server.input, server.feed);
	server.lrenderer = renderer_create();
	server.status = status_create();
	server.lrenderer->status = server.status;

	/* Load desktop wallpaper */
	const char *wp = getenv("MARSHAL_WALLPAPER");
	if (!wp) wp = "chromatic1.jpeg";
	renderer_load_wallpaper(server.lrenderer, wp);

	/* Load history, briefing, and active watchers */
	feed_load_history(server.feed);
	feed_load_briefing(server.feed);
	feed_load_watchers(server.feed);

	/* Wakeup pipe for HTTP thread notifications */
	int flags = fcntl(server.feed->wakeup_pipe[0], F_GETFL);
	fcntl(server.feed->wakeup_pipe[0], F_SETFL, flags | O_NONBLOCK);
	wl_event_loop_add_fd(server.event_loop, server.feed->wakeup_pipe[0],
		WL_EVENT_READABLE, wakeup_handler, &server);

	/* Cursor blink timer */
	server.cursor_timer = wl_event_loop_add_timer(server.event_loop,
		cursor_timer_cb, &server);
	wl_event_source_timer_update(server.cursor_timer, 530);

	/* Animation timer */
	server.anim_timer = wl_event_loop_add_timer(server.event_loop,
		anim_timer_cb, &server);

	/* Briefing retry: if indexer wasn't ready at boot, retry once after 30s */
	server.briefing_retry_timer = wl_event_loop_add_timer(
		server.event_loop, briefing_retry_cb, &server);
	wl_event_source_timer_update(server.briefing_retry_timer, 30000);

	/* Status bar: update clock every second, full poll every 30s */
	server.status_timer = wl_event_loop_add_timer(server.event_loop,
		status_timer_cb, &server);
	wl_event_source_timer_update(server.status_timer, 1000);

	/* Reap zombie children from fork() calls (terminal, locker, screenshot) */
	wl_event_loop_add_signal(server.event_loop, SIGCHLD, sigchld_handler,
		&server);

	/* logind: lock screen before suspend */
	setup_logind_sleep_monitor(&server);

	/* Start backend */
	if (!wlr_backend_start(server.backend)) {
		fprintf(stderr, "Failed to start backend\n");
		wlr_backend_destroy(server.backend);
		wl_display_destroy(server.display);
		return 1;
	}

	/* Set WAYLAND_DISPLAY so child processes can connect */
	const char *wl_socket = wl_display_add_socket_auto(server.display);
	if (!wl_socket) {
		fprintf(stderr, "Failed to open Wayland socket\n");
		wlr_backend_destroy(server.backend);
		wl_display_destroy(server.display);
		return 1;
	}
	setenv("WAYLAND_DISPLAY", wl_socket, 1);
	fprintf(stderr, "Marshal compositor running on %s\n", wl_socket);

	/* Store the socket name directly in the feed struct so feed threads
	 * read the compositor's OWN socket, not an inherited env value. */
	if (server.feed)
		snprintf(server.feed->wayland_display,
			sizeof(server.feed->wayland_display), "%s", wl_socket);

	/* Write display socket to ~/.marshal/wayland-display so the API server
	 * and agents can discover which compositor to connect to. */
	{
		const char *home = getenv("HOME");
		if (home) {
			char path[512];
			snprintf(path, sizeof(path), "%s/.marshal", home);
			mkdir(path, 0700);
			snprintf(path, sizeof(path),
				"%s/.marshal/wayland-display", home);
			FILE *f = fopen(path, "w");
			if (f) {
				fprintf(f, "%s\n", wl_socket);
				fclose(f);
			}
		}
	}

	/* Compositor → agentd event broadcast socket */
	{
		const char *xdg = getenv("XDG_RUNTIME_DIR");
		if (!xdg) xdg = "/tmp";
		char sock_path[256];
		snprintf(sock_path, sizeof(sock_path),
			"%s/marshal-compositor-events.sock", xdg);
		unlink(sock_path);

		server.event_srv_fd = socket(AF_UNIX,
			SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
		if (server.event_srv_fd >= 0) {
			struct sockaddr_un addr = {.sun_family = AF_UNIX};
			strncpy(addr.sun_path, sock_path,
				sizeof(addr.sun_path) - 1);
			if (bind(server.event_srv_fd,
					(struct sockaddr *)&addr,
					sizeof(addr)) == 0 &&
					listen(server.event_srv_fd, 8) == 0) {
				server.event_srv_src = wl_event_loop_add_fd(
					server.event_loop,
					server.event_srv_fd,
					WL_EVENT_READABLE,
					event_accept_cb, &server);
				fprintf(stderr,
					"Event socket: %s\n", sock_path);
			} else {
				close(server.event_srv_fd);
				server.event_srv_fd = -1;
				fprintf(stderr,
					"Warning: event socket bind failed\n");
			}
		}
	}

	/* agentd → compositor proactive-intent socket */
	{
		server.proactive_srv_fd = -1;
		for (int i = 0; i < (int)(sizeof(server.proactive_clients) /
				sizeof(server.proactive_clients[0])); i++)
			server.proactive_clients[i].fd = -1;

		const char *xdg = getenv("XDG_RUNTIME_DIR");
		if (!xdg) xdg = "/tmp";
		char sock_path[256];
		snprintf(sock_path, sizeof(sock_path),
			"%s/marshal-proactive.sock", xdg);
		unlink(sock_path);

		int fd = socket(AF_UNIX,
			SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
		if (fd >= 0) {
			struct sockaddr_un addr = {.sun_family = AF_UNIX};
			strncpy(addr.sun_path, sock_path,
				sizeof(addr.sun_path) - 1);
			if (bind(fd, (struct sockaddr *)&addr,
					sizeof(addr)) == 0 &&
					listen(fd, 4) == 0) {
				server.proactive_srv_fd = fd;
				server.proactive_srv_src = wl_event_loop_add_fd(
					server.event_loop, fd,
					WL_EVENT_READABLE,
					proactive_accept_cb, &server);
				fprintf(stderr,
					"Proactive socket: %s\n", sock_path);
			} else {
				close(fd);
				fprintf(stderr,
					"Warning: proactive socket bind failed\n");
			}
		}
	}

	/* Mark panel dirty for initial render */
	server.panel_dirty = true;

	wl_display_run(server.display);

	/* Cleanup */
	/* Close event socket and all connected clients */
	for (int i = 0; i < server.event_client_count; i++)
		close(server.event_clients[i]);
	if (server.event_srv_fd >= 0)
		close(server.event_srv_fd);

	/* Close proactive socket and all connected clients */
	for (int i = 0; i < server.proactive_client_count; i++) {
		if (server.proactive_clients[i].src)
			wl_event_source_remove(server.proactive_clients[i].src);
		if (server.proactive_clients[i].fd >= 0)
			close(server.proactive_clients[i].fd);
	}
	if (server.proactive_srv_src)
		wl_event_source_remove(server.proactive_srv_src);
	if (server.proactive_srv_fd >= 0)
		close(server.proactive_srv_fd);

	wl_display_destroy_clients(server.display);
	if (server.logind_event)
		wl_event_source_remove(server.logind_event);
	if (server.sleep_slot)
		sd_bus_slot_unref(server.sleep_slot);
	if (server.logind_bus)
		sd_bus_unref(server.logind_bus);
	wlr_scene_node_destroy(&server.scene->tree.node);
	wlr_cursor_destroy(server.cursor);
	wlr_xcursor_manager_destroy(server.cursor_mgr);
	feed_destroy(server.feed);
	renderer_destroy(server.lrenderer);
	status_destroy(server.status);
	wl_display_destroy(server.display);

	return 0;
}
