/*
 * Leaves OS compositor — wlr_scene-based Wayland compositor.
 *
 * Layout: Leaves AI panel (left, Cairo-rendered) + app windows (right, xdg_shell).
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
#include <sys/stat.h>
#include <linux/input-event-codes.h>

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
#include <wlr/types/wlr_data_device.h>
#include <wlr/types/wlr_keyboard.h>
#include <wlr/types/wlr_seat.h>
#include <wlr/types/wlr_input_device.h>
#include <wlr/types/wlr_pointer.h>
#include <wlr/types/wlr_buffer.h>
#include <wlr/interfaces/wlr_buffer.h>
#include <wlr/types/wlr_xdg_decoration_v1.h>
#include <wlr/backend/session.h>
#include <wlr/util/log.h>
#include <xkbcommon/xkbcommon.h>
#include <drm_fourcc.h>

#include "renderer.h"
#include "feed.h"
#include "input.h"
#include "geometry.h"
#include "status.h"

/* ── Layout constants ── */

#define PANEL_WIDTH 420  /* fixed width for the AI panel */

/* ── Modifier bit-flags ── */

#define MOD_CTRL  (1 << 0)
#define MOD_ALT   (1 << 1)
#define MOD_SUPER (1 << 2)

/* ── Focus mode ── */

enum leaves_focus_mode {
	FOCUS_NONE,   /* keyboard input is idle (no target) */
	FOCUS_PANEL,  /* keyboard goes to the Leaves input bar */
	FOCUS_APP,    /* keyboard goes to the focused toplevel */
};

/* ── Panel buffer (custom wlr_buffer wrapping Cairo pixel data) ── */

struct leaves_panel_buffer {
	struct wlr_buffer base;
	void *data;
	size_t stride;
	int width, height;
};

static void panel_buffer_destroy(struct wlr_buffer *wlr_buf) {
	struct leaves_panel_buffer *buf = wl_container_of(wlr_buf, buf, base);
	/* data is owned by Cairo — do not free */
	free(buf);
}

static bool panel_buffer_begin_data_ptr_access(struct wlr_buffer *wlr_buf,
		uint32_t flags, void **data, uint32_t *format, size_t *stride) {
	struct leaves_panel_buffer *buf = wl_container_of(wlr_buf, buf, base);
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

static struct leaves_panel_buffer *panel_buffer_create(void *data,
		int width, int height, size_t stride) {
	struct leaves_panel_buffer *buf = calloc(1, sizeof(*buf));
	if (!buf) return NULL;
	wlr_buffer_init(&buf->base, &panel_buffer_impl, width, height);
	buf->data = data;
	buf->stride = stride;
	buf->width = width;
	buf->height = height;
	return buf;
}

/* ── Toplevel (managed app window) ── */

struct leaves_toplevel {
	struct wl_list link;  /* leaves_server.toplevels */
	struct leaves_server *server;
	struct wlr_xdg_toplevel *xdg_toplevel;
	struct wlr_scene_tree *scene_tree;

	struct wl_listener map;
	struct wl_listener unmap;
	struct wl_listener commit;
	struct wl_listener destroy;
	struct wl_listener request_maximize;
	struct wl_listener request_fullscreen;
};

/* ── Output ── */

struct leaves_output {
	struct wl_list link;
	struct leaves_server *server;
	struct wlr_output *wlr_output;
	struct wlr_scene_output *scene_output;
	struct wl_listener frame;
	struct wl_listener request_state;
	struct wl_listener destroy;
};

/* ── Keyboard ── */

struct leaves_keyboard {
	struct leaves_server *server;
	struct wlr_keyboard *wlr_keyboard;
	struct wl_listener key;
	struct wl_listener modifiers;
	struct wl_listener destroy;
};

/* ── Server ── */

struct leaves_server {
	struct wl_display *display;
	struct wl_event_loop *event_loop;
	struct wlr_backend *backend;
	struct wlr_session *session;
	struct wlr_renderer *renderer;
	struct wlr_allocator *allocator;
	struct wlr_output_layout *output_layout;

	/* Scene graph */
	struct wlr_scene *scene;
	struct wlr_scene_output_layout *scene_layout;
	struct wlr_scene_tree *panel_tree;  /* parent for panel buffer */
	struct wlr_scene_tree *app_tree;    /* parent for app windows */
	struct wlr_scene_buffer *panel_scene_buf;
	struct wlr_scene_rect *panel_bg;    /* background behind panel */

	struct wl_list outputs;  /* leaves_output */
	struct wlr_seat *seat;

	/* Wayland protocols */
	struct wlr_xdg_shell *xdg_shell;
	struct wlr_xdg_decoration_manager_v1 *decoration_mgr;

	/* Listeners (decoration) */
	struct wl_listener new_decoration;

	/* Cursor */
	struct wlr_cursor *cursor;
	struct wlr_xcursor_manager *cursor_mgr;

	/* App windows */
	struct wl_list toplevels;  /* leaves_toplevel */

	/* Leaves subsystems */
	struct leaves_feed *feed;
	struct leaves_input input;
	struct leaves_renderer *lrenderer;

	/* Focus */
	enum leaves_focus_mode focus_mode;

	/* Modifier state */
	uint32_t modifiers;

	/* Timers */
	struct wl_event_source *cursor_timer;
	struct wl_event_source *anim_timer;
	struct wl_event_source *briefing_retry_timer;
	struct wl_event_source *status_timer;

	/* System status bar */
	struct leaves_status *status;

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

static void schedule_panel_redraw(struct leaves_server *server);
static void update_panel_buffer(struct leaves_server *server);
static void focus_toplevel(struct leaves_server *server,
	struct leaves_toplevel *toplevel);
static void relayout_toplevels(struct leaves_server *server);

/* ── Panel dimensions ── */

static int output_height(struct leaves_server *server) {
	struct leaves_output *out;
	wl_list_for_each(out, &server->outputs, link) {
		int w, h;
		wlr_output_effective_resolution(out->wlr_output, &w, &h);
		return h;
	}
	return 1080;
}

static int output_width(struct leaves_server *server) {
	struct leaves_output *out;
	wl_list_for_each(out, &server->outputs, link) {
		int w, h;
		wlr_output_effective_resolution(out->wlr_output, &w, &h);
		return w;
	}
	return 1920;
}

static int effective_panel_width(struct leaves_server *server) {
	if (wl_list_empty(&server->toplevels))
		return output_width(server);  /* full screen when no apps open */
	return PANEL_WIDTH;
}

/* ── Panel clipboard source ── */

struct leaves_clipboard_source {
	struct wlr_data_source base;
	char text[1024];
	int len;
};

static void clipboard_source_send(struct wlr_data_source *wlr_source,
		const char *mime_type, int32_t fd) {
	struct leaves_clipboard_source *src =
		wl_container_of(wlr_source, src, base);
	(void)mime_type;
	write(fd, src->text, src->len);
	close(fd);
}

static void clipboard_source_destroy(struct wlr_data_source *wlr_source) {
	struct leaves_clipboard_source *src =
		wl_container_of(wlr_source, src, base);
	free(src);
}

static const struct wlr_data_source_impl clipboard_source_impl = {
	.send    = clipboard_source_send,
	.destroy = clipboard_source_destroy,
};

/* Copy an arbitrary text range to the Wayland selection. */
static void panel_copy_range_to_clipboard(struct leaves_server *server,
		const char *text, int len) {
	if (len <= 0) return;
	if (len > (int)sizeof(((struct leaves_clipboard_source *)0)->text) - 1)
		len = (int)sizeof(((struct leaves_clipboard_source *)0)->text) - 1;

	struct leaves_clipboard_source *src = calloc(1, sizeof(*src));
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
static void panel_copy_to_clipboard(struct leaves_server *server) {
	panel_copy_range_to_clipboard(server,
		server->input.buf, server->input.len);
}

/* Paste the Wayland selection into the panel input bar (Ctrl+V).
 * Uses a pipe + 100 ms select() timeout.  For remote Wayland clients the
 * display must be flushed first so they receive the send request. */
static void panel_paste_from_clipboard(struct leaves_server *server) {
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
	struct leaves_server *server = data;
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
	struct leaves_server *server = data;
	float dt = 1.0f / 60.0f;
	if (feed_animate(server->feed, dt)) {
		schedule_panel_redraw(server);
		wl_event_source_timer_update(server->anim_timer, 16);
	}
	return 0;
}

/* ── Briefing retry timer ── */

static int briefing_retry_cb(void *data) {
	struct leaves_server *server = data;
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
	struct leaves_server *server = data;
	static int tick = 0;

	status_update_clock(server->status);
	if (tick++ % 30 == 0)   /* full system poll every 30 s */
		status_poll(server->status);

	schedule_panel_redraw(server);
	wl_event_source_timer_update(server->status_timer, 1000);
	return 0;
}

/* ── Wakeup pipe ── */

static int wakeup_handler(int fd, uint32_t mask __attribute__((unused)),
		void *data) {
	struct leaves_server *server = data;
	char byte;
	while (read(fd, &byte, 1) == 1) {}
	schedule_panel_redraw(server);
	wl_event_source_timer_update(server->anim_timer, 16);
	return 0;
}

/* ── Panel rendering ── */

static void schedule_panel_redraw(struct leaves_server *server) {
	server->panel_dirty = true;
	/* Schedule frame on all outputs so the scene gets re-committed */
	struct leaves_output *output;
	wl_list_for_each(output, &server->outputs, link) {
		wlr_output_schedule_frame(output->wlr_output);
	}
}

static void update_panel_buffer(struct leaves_server *server) {
	if (!server->panel_dirty) return;
	server->panel_dirty = false;

	int pw = effective_panel_width(server);
	int ph = output_height(server);

	/* Ensure cairo surface matches panel size */
	if (server->lrenderer->width != pw ||
			server->lrenderer->height != ph) {
		renderer_resize(server->lrenderer, pw, ph);
	}

	int stride;
	unsigned char *pixels = renderer_draw_frame(server->lrenderer,
		server->feed, &server->input, &stride);
	if (!pixels) return;

	/* Create new wlr_buffer wrapping the Cairo pixel data */
	struct leaves_panel_buffer *pbuf = panel_buffer_create(pixels,
		pw, ph, (size_t)stride);
	if (!pbuf) return;

	/* Update the scene buffer node */
	wlr_scene_buffer_set_buffer(server->panel_scene_buf, &pbuf->base);

	/* We transferred ownership to the scene; drop our reference */
	wlr_buffer_drop(&pbuf->base);
}

/* ── Output frame handler ── */

static void output_frame(struct wl_listener *listener, void *data) {
	struct leaves_output *output = wl_container_of(listener, output, frame);
	struct leaves_server *server = output->server;

	/* Re-render panel if dirty */
	update_panel_buffer(server);

	struct wlr_scene_output *scene_output = output->scene_output;
	wlr_scene_output_commit(scene_output, NULL);

	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	wlr_scene_output_send_frame_done(scene_output, &now);
}

static void output_request_state(struct wl_listener *listener, void *data) {
	struct leaves_output *output =
		wl_container_of(listener, output, request_state);
	const struct wlr_output_event_request_state *event = data;
	wlr_output_commit_state(output->wlr_output, event->state);
	schedule_panel_redraw(output->server);
	relayout_toplevels(output->server);
}

static void output_destroy(struct wl_listener *listener, void *data) {
	struct leaves_output *output = wl_container_of(listener, output, destroy);
	wl_list_remove(&output->frame.link);
	wl_list_remove(&output->request_state.link);
	wl_list_remove(&output->destroy.link);
	wl_list_remove(&output->link);
	free(output);
}

/* ── Toplevel management ── */

static void toplevel_map(struct wl_listener *listener, void *data) {
	struct leaves_toplevel *toplevel =
		wl_container_of(listener, toplevel, map);
	struct leaves_server *server = toplevel->server;

	wl_list_insert(&server->toplevels, &toplevel->link);

	/* Position in app area and configure size */
	relayout_toplevels(server);

	/* Focus the new window */
	server->focus_mode = FOCUS_APP;
	focus_toplevel(server, toplevel);
	schedule_panel_redraw(server);
}

static void toplevel_unmap(struct wl_listener *listener, void *data) {
	struct leaves_toplevel *toplevel =
		wl_container_of(listener, toplevel, unmap);
	struct leaves_server *server = toplevel->server;

	wl_list_remove(&toplevel->link);

	/* If this was the focused window, switch focus */
	if (!wl_list_empty(&server->toplevels)) {
		struct leaves_toplevel *next = wl_container_of(
			server->toplevels.next, next, link);
		focus_toplevel(server, next);
	} else {
		server->focus_mode = FOCUS_NONE;
		wlr_seat_keyboard_clear_focus(server->seat);
		schedule_panel_redraw(server);
	}
}

static void toplevel_commit(struct wl_listener *listener, void *data) {
	struct leaves_toplevel *toplevel =
		wl_container_of(listener, toplevel, commit);

	if (toplevel->xdg_toplevel->base->initial_commit) {
		/* Send initial configure with the app area dimensions.
		 * Don't use effective_panel_width here — the toplevel isn't
		 * in server->toplevels yet (added in toplevel_map). */
		struct leaves_server *server = toplevel->server;
		int ow = output_width(server);
		int oh = output_height(server);
		int app_w = ow - PANEL_WIDTH;

		wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel, app_w, oh);
		wlr_xdg_toplevel_set_activated(toplevel->xdg_toplevel, true);
	}
}

static void toplevel_destroy(struct wl_listener *listener, void *data) {
	struct leaves_toplevel *toplevel =
		wl_container_of(listener, toplevel, destroy);

	wl_list_remove(&toplevel->map.link);
	wl_list_remove(&toplevel->unmap.link);
	wl_list_remove(&toplevel->commit.link);
	wl_list_remove(&toplevel->destroy.link);
	wl_list_remove(&toplevel->request_maximize.link);
	wl_list_remove(&toplevel->request_fullscreen.link);
	free(toplevel);
}

static void toplevel_request_maximize(struct wl_listener *listener,
		void *data) {
	struct leaves_toplevel *toplevel =
		wl_container_of(listener, toplevel, request_maximize);
	/* Always maximize to the app area */
	wlr_xdg_toplevel_set_maximized(toplevel->xdg_toplevel, true);
	relayout_toplevels(toplevel->server);
}

static void toplevel_request_fullscreen(struct wl_listener *listener,
		void *data) {
	struct leaves_toplevel *toplevel =
		wl_container_of(listener, toplevel, request_fullscreen);
	/* For now, treat fullscreen same as maximized (fill app area) */
	wlr_xdg_toplevel_set_fullscreen(toplevel->xdg_toplevel, false);
	relayout_toplevels(toplevel->server);
}

static void focus_toplevel(struct leaves_server *server,
		struct leaves_toplevel *toplevel) {
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
	wlr_scene_node_raise_to_top(&toplevel->scene_tree->node);
	wlr_xdg_toplevel_set_activated(toplevel->xdg_toplevel, true);

	struct wlr_keyboard *keyboard = wlr_seat_get_keyboard(server->seat);
	if (keyboard) {
		wlr_seat_keyboard_notify_enter(server->seat,
			toplevel->xdg_toplevel->base->surface,
			keyboard->keycodes, keyboard->num_keycodes,
			&keyboard->modifiers);
	}
}

static void relayout_toplevels(struct leaves_server *server) {
	int ow = output_width(server);
	int oh = output_height(server);
	int pw = effective_panel_width(server);

	/* Resize panel background to full output */
	wlr_scene_rect_set_size(server->panel_bg, ow, oh);

	/* Resize panel renderer to match effective width */
	renderer_resize(server->lrenderer, pw, oh);

	/* Move app tree */
	wlr_scene_node_set_position(&server->app_tree->node, pw, 0);

	/* Configure each toplevel */
	int app_w = ow - pw;
	if (app_w < 100) app_w = 100;
	struct leaves_toplevel *toplevel;
	wl_list_for_each(toplevel, &server->toplevels, link) {
		wlr_xdg_toplevel_set_size(toplevel->xdg_toplevel, app_w, oh);
		wlr_scene_node_set_position(&toplevel->scene_tree->node, 0, 0);
	}

	schedule_panel_redraw(server);
}

/* ── xdg_shell handlers ── */

static void server_new_xdg_toplevel(struct wl_listener *listener, void *data) {
	struct leaves_server *server =
		wl_container_of(listener, server, new_xdg_toplevel);
	struct wlr_xdg_toplevel *xdg_toplevel = data;

	struct leaves_toplevel *toplevel = calloc(1, sizeof(*toplevel));
	toplevel->server = server;
	toplevel->xdg_toplevel = xdg_toplevel;

	/* Create scene tree for this toplevel under the app_tree */
	toplevel->scene_tree = wlr_scene_xdg_surface_create(
		server->app_tree, xdg_toplevel->base);
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
	struct leaves_server *server =
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
	struct leaves_keyboard *keyboard =
		wl_container_of(listener, keyboard, key);
	struct leaves_server *server = keyboard->server;
	struct wlr_keyboard_key_event *event = data;

	uint32_t keycode = event->keycode + 8;
	const xkb_keysym_t *syms;
	int nsyms = xkb_state_key_get_syms(
		keyboard->wlr_keyboard->xkb_state, keycode, &syms);

	bool handled = false;

	if (event->state == WL_KEYBOARD_KEY_STATE_PRESSED) {
		for (int i = 0; i < nsyms; i++) {
			/* ── VT switch: Ctrl+Alt+F1..F12 ── */
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

			/* ── Super+Q: close focused app window ── */
			if ((server->modifiers & MOD_SUPER) &&
					syms[i] == XKB_KEY_q) {
				if (!wl_list_empty(&server->toplevels)) {
					struct leaves_toplevel *top = wl_container_of(
						server->toplevels.next, top, link);
					wlr_xdg_toplevel_send_close(top->xdg_toplevel);
				}
				handled = true;
				break;
			}

			/* ── Super+Space: cycle focus: NONE → PANEL → APP → NONE ── */
			if ((server->modifiers & MOD_SUPER) &&
					syms[i] == XKB_KEY_space) {
				if (server->focus_mode == FOCUS_NONE) {
					server->focus_mode = FOCUS_PANEL;
					wlr_seat_keyboard_clear_focus(server->seat);
				} else if (server->focus_mode == FOCUS_PANEL &&
						!wl_list_empty(&server->toplevels)) {
					server->focus_mode = FOCUS_APP;
					struct leaves_toplevel *top = wl_container_of(
						server->toplevels.next, top, link);
					focus_toplevel(server, top);
				} else {
					/* FOCUS_APP → NONE, or PANEL with no apps → NONE */
					server->focus_mode = FOCUS_NONE;
					wlr_seat_keyboard_clear_focus(server->seat);
				}
				schedule_panel_redraw(server);
				handled = true;
				break;
			}
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

	if (server->focus_mode == FOCUS_NONE) {
		/* No focus target — drop keypresses */
		return;
	} else if (server->focus_mode == FOCUS_PANEL) {
		/* Route to Leaves input handler */
		if (event->state != WL_KEYBOARD_KEY_STATE_PRESSED) return;

		/* ── Clipboard / selection shortcuts ── */
		if (server->modifiers & MOD_CTRL) {
			struct leaves_input *inp = &server->input;

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
	}
}

static void keyboard_handle_modifiers(struct wl_listener *listener,
		void *data) {
	struct leaves_keyboard *keyboard =
		wl_container_of(listener, keyboard, modifiers);
	struct leaves_server *server = keyboard->server;

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
	struct leaves_keyboard *keyboard =
		wl_container_of(listener, keyboard, destroy);
	wl_list_remove(&keyboard->key.link);
	wl_list_remove(&keyboard->modifiers.link);
	wl_list_remove(&keyboard->destroy.link);
	free(keyboard);
}

/* ── Cursor handlers ── */

static struct leaves_toplevel *toplevel_at(struct leaves_server *server,
		double lx, double ly, struct wlr_surface **surface,
		double *sx, double *sy) {
	struct wlr_scene_node *node = wlr_scene_node_at(
		&server->scene->tree.node, lx, ly, sx, sy);
	if (!node || node->type != WLR_SCENE_NODE_BUFFER) return NULL;

	struct wlr_scene_buffer *scene_buffer =
		wlr_scene_buffer_from_node(node);
	struct wlr_scene_surface *scene_surface =
		wlr_scene_surface_try_from_buffer(scene_buffer);
	if (!scene_surface) return NULL;

	*surface = scene_surface->surface;

	/* Walk up the tree to find the toplevel */
	struct wlr_scene_tree *tree = node->parent;
	while (tree && !tree->node.data) {
		tree = tree->node.parent;
	}
	if (!tree) return NULL;
	return tree->node.data;
}

static void process_cursor_motion(struct leaves_server *server,
		uint32_t time) {
	double sx, sy;
	struct wlr_surface *surface = NULL;
	struct leaves_toplevel *toplevel = toplevel_at(server,
		server->cursor->x, server->cursor->y, &surface, &sx, &sy);

	if (!toplevel) {
		/* Show I-beam cursor when hovering over the editable input field
		 * or over selectable card text */
		int oh = output_height(server);
		int bar_y = oh - INPUT_HEIGHT;
		bool in_input_field =
			server->cursor->x >= TASKBAR_ICON_W &&
			server->cursor->x <  effective_panel_width(server) - TASKBAR_APPS_W &&
			server->cursor->y >= bar_y;

		bool in_card_text = false;
		if (!in_input_field &&
				server->cursor->x < effective_panel_width(server)) {
			int dummy;
			in_card_text = renderer_card_text_at(server->lrenderer,
				server->cursor->x, server->cursor->y, &dummy) >= 0;
		}

		wlr_cursor_set_xcursor(server->cursor, server->cursor_mgr,
			(in_input_field || in_card_text) ? "text" : "default");
		wlr_seat_pointer_clear_focus(server->seat);
	} else {
		wlr_cursor_set_xcursor(server->cursor, server->cursor_mgr, "default");
		wlr_seat_pointer_notify_enter(server->seat, surface, sx, sy);
		wlr_seat_pointer_notify_motion(server->seat, time, sx, sy);
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
	struct leaves_server *server =
		wl_container_of(listener, server, cursor_motion);
	struct wlr_pointer_motion_event *event = data;
	wlr_cursor_move(server->cursor, &event->pointer->base,
		event->delta_x, event->delta_y);
	process_cursor_motion(server, event->time_msec);
}

static void cursor_motion_absolute_handler(struct wl_listener *listener,
		void *data) {
	struct leaves_server *server =
		wl_container_of(listener, server, cursor_motion_absolute);
	struct wlr_pointer_motion_absolute_event *event = data;
	wlr_cursor_warp_absolute(server->cursor, &event->pointer->base,
		event->x, event->y);
	process_cursor_motion(server, event->time_msec);
}

static void cursor_button_handler(struct wl_listener *listener, void *data) {
	struct leaves_server *server =
		wl_container_of(listener, server, cursor_button);
	struct wlr_pointer_button_event *event = data;

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

		double sx, sy;
		struct wlr_surface *surface = NULL;
		struct leaves_toplevel *toplevel = toplevel_at(server,
			mx, my, &surface, &sx, &sy);

		if (toplevel) {
			/* Click on app window — focus it */
			server->focus_mode = FOCUS_APP;
			input_clear_selection(&server->input);
			if (server->status) server->status->dropdown_open = false;
			focus_toplevel(server, toplevel);
		} else if (mx < effective_panel_width(server)) {
			int oh = output_height(server);
			int bar_y = oh - INPUT_HEIGHT;

			/* Click inside an open dropdown? */
			if (server->status && server->status->dropdown_open) {
				struct leaves_renderer *lr = server->lrenderer;
				if (mx >= lr->dropdown_x &&
						mx < lr->dropdown_x + lr->dropdown_w &&
						my >= lr->dropdown_y &&
						my < lr->dropdown_y + lr->dropdown_h) {
					/* Inside dropdown — consume click */
					schedule_panel_redraw(server);
					goto done_press;
				}
			}

			if (my >= bar_y) {
				/* Click in the taskbar region */
				struct leaves_renderer *lr = server->lrenderer;

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
					pthread_mutex_lock(&server->feed->mutex);
					server->feed->selected_card = card_idx;
					/* Toggle expansion: click same card again to collapse */
					if (card_idx >= 0 && card_idx == server->feed->expanded_card)
						server->feed->expanded_card = -1;
					else
						server->feed->expanded_card = card_idx;
					pthread_mutex_unlock(&server->feed->mutex);
					server->lrenderer->card_sel.card_idx = -1;
					server->card_text_drag = false;
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
	struct leaves_server *server =
		wl_container_of(listener, server, cursor_axis);
	struct wlr_pointer_axis_event *event = data;

	/* Scroll the panel feed when the cursor is over the panel area */
	if (server->cursor->x < effective_panel_width(server) &&
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
	struct leaves_server *server =
		wl_container_of(listener, server, cursor_frame);
	wlr_seat_pointer_notify_frame(server->seat);
}

static void request_set_cursor_handler(struct wl_listener *listener,
		void *data) {
	struct leaves_server *server =
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
	struct leaves_server *server =
		wl_container_of(listener, server, new_input);
	struct wlr_input_device *device = data;

	if (device->type == WLR_INPUT_DEVICE_KEYBOARD) {
		struct leaves_keyboard *keyboard = calloc(1, sizeof(*keyboard));
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
}

/* ── New output ── */

static void server_new_output(struct wl_listener *listener, void *data) {
	struct leaves_server *server =
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

	struct leaves_output *output = calloc(1, sizeof(*output));
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
	struct leaves_server *server =
		wl_container_of(listener, server, backend_destroy);
	wl_display_terminate(server->display);
}

/* ── Wayland data-device: honour selection requests from clients ── */

static void handle_request_set_selection(struct wl_listener *listener,
		void *data) {
	struct leaves_server *server =
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

struct leaves_decoration {
	struct wl_listener request_mode;
	struct wl_listener destroy;
};

static void decoration_handle_destroy(struct wl_listener *listener,
		void *data) {
	struct leaves_decoration *deco =
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

	struct leaves_decoration *deco = calloc(1, sizeof(*deco));
	deco->request_mode.notify = decoration_handle_request_mode;
	wl_signal_add(&decoration->events.request_mode, &deco->request_mode);
	deco->destroy.notify = decoration_handle_destroy;
	wl_signal_add(&decoration->events.destroy, &deco->destroy);
}

/* ── Main ── */

int main(int argc, char *argv[]) {
	wlr_log_init(WLR_DEBUG, NULL);

	struct leaves_server server = {0};
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

	/* Wayland compositor globals — required for clients */
	wlr_compositor_create(server.display, 6, server.renderer);
	wlr_data_device_manager_create(server.display);

	/* Output layout */
	server.output_layout = wlr_output_layout_create(server.display);

	/* Scene graph */
	server.scene = wlr_scene_create();
	server.scene_layout = wlr_scene_attach_output_layout(server.scene,
		server.output_layout);

	/* Scene tree structure:
	 *   scene root
	 *     ├── panel_bg (solid rect, dark background)
	 *     ├── panel_tree
	 *     │   └── panel_scene_buf (Cairo-rendered panel)
	 *     └── app_tree (positioned at x=PANEL_WIDTH)
	 *         └── [xdg_surface nodes from client windows]
	 */
	float panel_bg_color[4] = {1.0f, 1.0f, 1.0f, 1.0f}; /* match BG_BASE #FFFFFF */
	server.panel_bg = wlr_scene_rect_create(&server.scene->tree,
		PANEL_WIDTH, 1080, panel_bg_color);

	server.panel_tree = wlr_scene_tree_create(&server.scene->tree);
	server.panel_scene_buf = wlr_scene_buffer_create(server.panel_tree,
		NULL);

	server.app_tree = wlr_scene_tree_create(&server.scene->tree);
	wlr_scene_node_set_position(&server.app_tree->node, PANEL_WIDTH, 0);

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

	/* Input devices */
	server.new_output.notify = server_new_output;
	wl_signal_add(&server.backend->events.new_output, &server.new_output);

	server.new_input.notify = server_new_input;
	wl_signal_add(&server.backend->events.new_input, &server.new_input);

	server.backend_destroy.notify = backend_destroy_handler;
	wl_signal_add(&server.backend->events.destroy, &server.backend_destroy);

	/* Feed + Input + Renderer + Status */
	const char *api_url = getenv("LEAVES_API_URL");
	if (!api_url) api_url = "http://127.0.0.1:8765";
	server.feed = feed_create(api_url);
	input_init(&server.input, server.feed);
	server.lrenderer = renderer_create();
	server.status = status_create();
	server.lrenderer->status = server.status;

	/* Load desktop wallpaper */
	const char *wp = getenv("LEAVES_WALLPAPER");
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

	/* Start backend */
	if (!wlr_backend_start(server.backend)) {
		fprintf(stderr, "Failed to start backend\n");
		wlr_backend_destroy(server.backend);
		wl_display_destroy(server.display);
		return 1;
	}

	/* Set WAYLAND_DISPLAY so child processes can connect */
	const char *socket = wl_display_add_socket_auto(server.display);
	if (!socket) {
		fprintf(stderr, "Failed to open Wayland socket\n");
		wlr_backend_destroy(server.backend);
		wl_display_destroy(server.display);
		return 1;
	}
	setenv("WAYLAND_DISPLAY", socket, true);
	fprintf(stderr, "Leaves compositor running on %s\n", socket);

	/* Write display socket to ~/.leaves/wayland-display so the API server
	 * and agents can discover which compositor to connect to. */
	{
		const char *home = getenv("HOME");
		if (home) {
			char path[512];
			snprintf(path, sizeof(path), "%s/.leaves", home);
			mkdir(path, 0700);
			snprintf(path, sizeof(path),
				"%s/.leaves/wayland-display", home);
			FILE *f = fopen(path, "w");
			if (f) {
				fprintf(f, "%s\n", socket);
				fclose(f);
			}
		}
	}

	/* Mark panel dirty for initial render */
	server.panel_dirty = true;

	wl_display_run(server.display);

	/* Cleanup */
	wl_display_destroy_clients(server.display);
	wlr_scene_node_destroy(&server.scene->tree.node);
	wlr_cursor_destroy(server.cursor);
	wlr_xcursor_manager_destroy(server.cursor_mgr);
	feed_destroy(server.feed);
	renderer_destroy(server.lrenderer);
	status_destroy(server.status);
	wl_display_destroy(server.display);

	return 0;
}
