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

	/* Panel dirty flag */
	bool panel_dirty;

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

	if (server->focus_mode == FOCUS_NONE) {
		/* No focus target — drop keypresses */
		return;
	} else if (server->focus_mode == FOCUS_PANEL) {
		/* Route to Leaves input handler */
		if (event->state != WL_KEYBOARD_KEY_STATE_PRESSED) return;

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
		/* Cursor is over the panel or empty space */
		wlr_cursor_set_xcursor(server->cursor, server->cursor_mgr,
			"default");
		wlr_seat_pointer_clear_focus(server->seat);
	} else {
		wlr_seat_pointer_notify_enter(server->seat, surface, sx, sy);
		wlr_seat_pointer_notify_motion(server->seat, time, sx, sy);
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
		double sx, sy;
		struct wlr_surface *surface = NULL;
		struct leaves_toplevel *toplevel = toplevel_at(server,
			server->cursor->x, server->cursor->y, &surface, &sx, &sy);

		if (toplevel) {
			/* Click on app window — focus it */
			server->focus_mode = FOCUS_APP;
			focus_toplevel(server, toplevel);
		} else if (server->cursor->x < effective_panel_width(server)) {
			/* Click on panel — input bar focuses panel, feed defocuses */
			int oh = output_height(server);
			int bar_y = oh - INPUT_HEIGHT;
			if (server->cursor->y >= bar_y) {
				server->focus_mode = FOCUS_PANEL;
			} else {
				server->focus_mode = FOCUS_NONE;
			}
			wlr_seat_keyboard_clear_focus(server->seat);
		}
		schedule_panel_redraw(server);
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
		server->feed->scroll_offset += event->delta;
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

	/* Input devices */
	server.new_output.notify = server_new_output;
	wl_signal_add(&server.backend->events.new_output, &server.new_output);

	server.new_input.notify = server_new_input;
	wl_signal_add(&server.backend->events.new_input, &server.new_input);

	server.backend_destroy.notify = backend_destroy_handler;
	wl_signal_add(&server.backend->events.destroy, &server.backend_destroy);

	/* Feed + Input + Renderer */
	const char *api_url = getenv("LEAVES_API_URL");
	if (!api_url) api_url = "http://127.0.0.1:8765";
	server.feed = feed_create(api_url);
	input_init(&server.input, server.feed);
	server.lrenderer = renderer_create();

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
	wl_display_destroy(server.display);

	return 0;
}
