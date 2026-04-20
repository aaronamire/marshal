/*
 * marshal-locker — Wayland screen locker for Marshal.
 *
 * Architecture:
 *   - ext-session-lock-v1 protocol (compositor-confirmed input isolation)
 *   - PAM authentication via pam_authenticate()
 *   - Cairo + Pango rendering with Marshal visual language
 *   - Triggered by compositor (Super+L) or logind PrepareForSleep
 *
 * Build: meson (see meson.build in this directory)
 */

#define _GNU_SOURCE
#include <cairo.h>
#include <errno.h>
#include <math.h>
#include <pango/pangocairo.h>
#include <pwd.h>
#include <security/pam_appl.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include <wayland-client.h>
#include <xkbcommon/xkbcommon.h>

#include "ext-session-lock-v1-client-protocol.h"

/* ── Configuration ── */

#define MAX_PASSWORD    256
#define FONT_TIME       "Geist Semi-Bold 64"
#define FONT_DATE       "Geist 18"
#define FONT_INPUT      "Geist 16"
#define FONT_STATUS     "Geist 13"

/* ── Wayland globals ── */

static struct wl_display    *display;
static struct wl_registry   *registry;
static struct wl_compositor *compositor;
static struct wl_shm        *shm;
static struct wl_seat       *seat;
static struct wl_keyboard   *keyboard;

static struct ext_session_lock_manager_v1 *lock_manager;
static struct ext_session_lock_v1         *session_lock;

/* ── Per-output lock surface state ──
 * ext-session-lock-v1 requires a lock surface on every output before the
 * compositor sends the `locked` event. We track one context per output and
 * iterate on configure / render / cleanup. */

#define MAX_LOCK_OUTPUTS 8

struct output_ctx {
	uint32_t          name;            /* wl_registry name */
	struct wl_output *wl_output;
	struct wl_surface *surface;
	struct ext_session_lock_surface_v1 *lock_surface;

	/* SHM buffer */
	struct wl_buffer *buffer;
	void   *shm_data;
	int     shm_size;
	int     width;
	int     height;

	/* Cairo */
	cairo_surface_t *cairo_surface;
	cairo_t         *cr;

	bool configured;
	bool needs_redraw;
};

static struct output_ctx outputs[MAX_LOCK_OUTPUTS];
static int output_count;

/* ── Locker state ── */

struct locker {
	PangoFontDescription *font_time;
	PangoFontDescription *font_date;
	PangoFontDescription *font_input;
	PangoFontDescription *font_status;

	/* Keyboard */
	struct xkb_context *xkb_ctx;
	struct xkb_keymap  *xkb_keymap;
	struct xkb_state   *xkb_state;

	/* Password input */
	char     password[MAX_PASSWORD];
	int      pw_len;
	bool     auth_failed;
	bool     authenticating;
	char     status_msg[128];

	/* Session state */
	bool locked;
};

static struct locker lock;

static void mark_all_dirty(void) {
	for (int i = 0; i < output_count; i++)
		outputs[i].needs_redraw = true;
}

/* ── SHM buffer creation ── */

static int create_shm_file(size_t size) {
	char name[] = "/marshal-lock-XXXXXX";
	int fd = memfd_create(name, MFD_CLOEXEC);
	if (fd < 0) return -1;
	if (ftruncate(fd, size) < 0) { close(fd); return -1; }
	return fd;
}

static void buffer_release(void *data, struct wl_buffer *buf) {
	(void)data; (void)buf;
}

static const struct wl_buffer_listener buffer_listener = {
	.release = buffer_release,
};

static bool create_buffer(struct output_ctx *ctx, int width, int height) {
	int stride = width * 4;
	int size = stride * height;

	if (ctx->buffer) {
		wl_buffer_destroy(ctx->buffer);
		ctx->buffer = NULL;
	}
	if (ctx->shm_data) {
		munmap(ctx->shm_data, ctx->shm_size);
		ctx->shm_data = NULL;
	}

	int fd = create_shm_file(size);
	if (fd < 0) return false;

	ctx->shm_data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (ctx->shm_data == MAP_FAILED) {
		ctx->shm_data = NULL;
		close(fd);
		return false;
	}
	ctx->shm_size = size;

	struct wl_shm_pool *pool = wl_shm_create_pool(shm, fd, size);
	ctx->buffer = wl_shm_pool_create_buffer(pool, 0, width, height,
		stride, WL_SHM_FORMAT_ARGB8888);
	wl_shm_pool_destroy(pool);
	close(fd);

	wl_buffer_add_listener(ctx->buffer, &buffer_listener, NULL);

	if (ctx->cr) { cairo_destroy(ctx->cr); ctx->cr = NULL; }
	if (ctx->cairo_surface) {
		cairo_surface_destroy(ctx->cairo_surface);
		ctx->cairo_surface = NULL;
	}

	ctx->cairo_surface = cairo_image_surface_create_for_data(
		ctx->shm_data, CAIRO_FORMAT_ARGB32, width, height, stride);
	ctx->cr = cairo_create(ctx->cairo_surface);

	ctx->width = width;
	ctx->height = height;
	return true;
}

/* ── Rendering ── */

static void render(struct output_ctx *ctx) {
	if (!ctx->cr) return;
	cairo_t *cr = ctx->cr;
	int w = ctx->width;
	int h = ctx->height;

	/* Background — dark with subtle gradient */
	cairo_set_source_rgb(cr, 0.06, 0.06, 0.08);
	cairo_paint(cr);

	/* Subtle radial gradient overlay */
	cairo_pattern_t *grad = cairo_pattern_create_radial(
		w / 2.0, h / 2.0, 0, w / 2.0, h / 2.0, w * 0.6);
	cairo_pattern_add_color_stop_rgba(grad, 0, 0.10, 0.10, 0.14, 0.3);
	cairo_pattern_add_color_stop_rgba(grad, 1, 0.06, 0.06, 0.08, 0.0);
	cairo_set_source(cr, grad);
	cairo_paint(cr);
	cairo_pattern_destroy(grad);

	/* Time */
	time_t now = time(NULL);
	struct tm *tm = localtime(&now);
	char time_str[16], date_str[64];
	strftime(time_str, sizeof(time_str), "%I:%M", tm);
	strftime(date_str, sizeof(date_str), "%A, %B %e", tm);

	/* Remove leading zero from hour */
	const char *time_display = time_str;
	if (time_display[0] == '0') time_display++;

	double center_y = h * 0.35;

	/* Draw time */
	PangoLayout *layout = pango_cairo_create_layout(cr);
	pango_layout_set_font_description(layout, lock.font_time);
	pango_layout_set_text(layout, time_display, -1);
	pango_layout_set_alignment(layout, PANGO_ALIGN_CENTER);

	PangoRectangle ext;
	pango_layout_get_pixel_extents(layout, NULL, &ext);
	cairo_set_source_rgba(cr, 0.93, 0.93, 0.94, 1.0);
	cairo_move_to(cr, (w - ext.width) / 2.0, center_y - ext.height);
	pango_cairo_show_layout(cr, layout);
	g_object_unref(layout);

	/* Draw date */
	layout = pango_cairo_create_layout(cr);
	pango_layout_set_font_description(layout, lock.font_date);
	pango_layout_set_text(layout, date_str, -1);
	pango_layout_get_pixel_extents(layout, NULL, &ext);
	cairo_set_source_rgba(cr, 0.70, 0.70, 0.72, 1.0);
	cairo_move_to(cr, (w - ext.width) / 2.0, center_y + 12);
	pango_cairo_show_layout(cr, layout);
	g_object_unref(layout);

	/* Password input area */
	double input_y = h * 0.55;
	double input_w = 280;
	double input_h = 44;
	double input_x = (w - input_w) / 2.0;
	double r = 8;

	/* Input field background */
	cairo_new_sub_path(cr);
	cairo_arc(cr, input_x + input_w - r, input_y + r, r, -M_PI / 2, 0);
	cairo_arc(cr, input_x + input_w - r, input_y + input_h - r, r, 0, M_PI / 2);
	cairo_arc(cr, input_x + r, input_y + input_h - r, r, M_PI / 2, M_PI);
	cairo_arc(cr, input_x + r, input_y + r, r, M_PI, 3 * M_PI / 2);
	cairo_close_path(cr);

	if (lock.auth_failed) {
		cairo_set_source_rgba(cr, 0.86, 0.15, 0.15, 0.15);
	} else {
		cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.08);
	}
	cairo_fill_preserve(cr);

	/* Input field border */
	if (lock.auth_failed) {
		cairo_set_source_rgba(cr, 0.86, 0.15, 0.15, 0.5);
	} else {
		cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.15);
	}
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);

	/* Password dots */
	if (lock.pw_len > 0) {
		double dot_r = 4;
		double dot_gap = 14;
		int dots = lock.pw_len;
		if (dots > 18) dots = 18;  /* max visible dots */
		double dots_w = dots * dot_gap;
		double dots_x = input_x + (input_w - dots_w) / 2.0 + dot_gap / 2.0;
		cairo_set_source_rgba(cr, 0.88, 0.88, 0.89, 0.9);
		for (int i = 0; i < dots; i++) {
			cairo_arc(cr, dots_x + i * dot_gap, input_y + input_h / 2.0,
				dot_r, 0, 2 * M_PI);
			cairo_fill(cr);
		}
	} else {
		/* Placeholder */
		layout = pango_cairo_create_layout(cr);
		pango_layout_set_font_description(layout, lock.font_input);
		pango_layout_set_text(layout, "Enter password", -1);
		pango_layout_get_pixel_extents(layout, NULL, &ext);
		cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.25);
		cairo_move_to(cr, input_x + (input_w - ext.width) / 2.0,
			input_y + (input_h - ext.height) / 2.0);
		pango_cairo_show_layout(cr, layout);
		g_object_unref(layout);
	}

	/* Status message */
	if (lock.status_msg[0]) {
		layout = pango_cairo_create_layout(cr);
		pango_layout_set_font_description(layout, lock.font_status);
		pango_layout_set_text(layout, lock.status_msg, -1);
		pango_layout_get_pixel_extents(layout, NULL, &ext);
		if (lock.auth_failed) {
			cairo_set_source_rgba(cr, 0.94, 0.33, 0.33, 0.9);
		} else {
			cairo_set_source_rgba(cr, 0.70, 0.70, 0.72, 0.8);
		}
		cairo_move_to(cr, (w - ext.width) / 2.0, input_y + input_h + 16);
		pango_cairo_show_layout(cr, layout);
		g_object_unref(layout);
	}

	cairo_surface_flush(ctx->cairo_surface);

	wl_surface_attach(ctx->surface, ctx->buffer, 0, 0);
	wl_surface_damage_buffer(ctx->surface, 0, 0, w, h);
	wl_surface_commit(ctx->surface);
}

static void render_all(void) {
	for (int i = 0; i < output_count; i++) {
		if (outputs[i].configured) {
			render(&outputs[i]);
			outputs[i].needs_redraw = false;
		}
	}
}

/* ── PAM authentication ── */

static int pam_conversation(int num_msg, const struct pam_message **msg,
		struct pam_response **resp, void *data) {
	(void)data;
	struct pam_response *reply = calloc(num_msg, sizeof(struct pam_response));
	if (!reply) return PAM_CONV_ERR;

	for (int i = 0; i < num_msg; i++) {
		switch (msg[i]->msg_style) {
		case PAM_PROMPT_ECHO_OFF:
		case PAM_PROMPT_ECHO_ON:
			reply[i].resp = strdup(lock.password);
			break;
		case PAM_ERROR_MSG:
		case PAM_TEXT_INFO:
			break;
		default:
			free(reply);
			return PAM_CONV_ERR;
		}
	}
	*resp = reply;
	return PAM_SUCCESS;
}

static bool authenticate(void) {
	struct passwd *pw = getpwuid(getuid());
	if (!pw) return false;

	struct pam_conv conv = { pam_conversation, NULL };
	pam_handle_t *pamh = NULL;

	int ret = pam_start("system-local-login", pw->pw_name, &conv, &pamh);
	if (ret != PAM_SUCCESS) {
		/* Fallback to "login" service */
		ret = pam_start("login", pw->pw_name, &conv, &pamh);
		if (ret != PAM_SUCCESS) return false;
	}

	ret = pam_authenticate(pamh, 0);
	bool success = (ret == PAM_SUCCESS);

	if (success) {
		pam_setcred(pamh, PAM_REFRESH_CRED);
	}

	pam_end(pamh, ret);
	return success;
}

/* ── Keyboard handling ── */

static void kbd_keymap(void *data, struct wl_keyboard *kbd,
		uint32_t format, int fd, uint32_t size) {
	(void)data; (void)kbd;
	if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) { close(fd); return; }

	char *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) { close(fd); return; }

	if (lock.xkb_keymap) xkb_keymap_unref(lock.xkb_keymap);
	if (lock.xkb_state) xkb_state_unref(lock.xkb_state);

	lock.xkb_keymap = xkb_keymap_new_from_string(lock.xkb_ctx, map,
		XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
	munmap(map, size);
	close(fd);

	if (lock.xkb_keymap)
		lock.xkb_state = xkb_state_new(lock.xkb_keymap);
}

static void kbd_enter(void *data, struct wl_keyboard *kbd, uint32_t serial,
		struct wl_surface *surface, struct wl_array *keys) {
	(void)data; (void)kbd; (void)serial; (void)surface; (void)keys;
}

static void kbd_leave(void *data, struct wl_keyboard *kbd, uint32_t serial,
		struct wl_surface *surface) {
	(void)data; (void)kbd; (void)serial; (void)surface;
}

static void kbd_key(void *data, struct wl_keyboard *kbd, uint32_t serial,
		uint32_t time, uint32_t key, uint32_t state) {
	(void)data; (void)kbd; (void)serial; (void)time;
	if (state != WL_KEYBOARD_KEY_STATE_PRESSED || !lock.xkb_state)
		return;

	uint32_t keycode = key + 8;
	xkb_keysym_t sym = xkb_state_key_get_one_sym(lock.xkb_state, keycode);

	if (sym == XKB_KEY_Return || sym == XKB_KEY_KP_Enter) {
		/* Attempt authentication */
		lock.authenticating = true;
		snprintf(lock.status_msg, sizeof(lock.status_msg), "Verifying...");
		mark_all_dirty();

		if (authenticate()) {
			lock.locked = false;
		} else {
			lock.auth_failed = true;
			lock.pw_len = 0;
			lock.password[0] = '\0';
			snprintf(lock.status_msg, sizeof(lock.status_msg),
				"Authentication failed");
		}
		lock.authenticating = false;
		mark_all_dirty();
		return;
	}

	if (sym == XKB_KEY_BackSpace) {
		if (lock.pw_len > 0) {
			lock.password[--lock.pw_len] = '\0';
			lock.auth_failed = false;
			lock.status_msg[0] = '\0';
			mark_all_dirty();
		}
		return;
	}

	if (sym == XKB_KEY_Escape) {
		/* Clear input */
		lock.pw_len = 0;
		lock.password[0] = '\0';
		lock.auth_failed = false;
		lock.status_msg[0] = '\0';
		mark_all_dirty();
		return;
	}

	/* Regular character input */
	char buf[8];
	int len = xkb_state_key_get_utf8(lock.xkb_state, keycode, buf, sizeof(buf));
	if (len > 0 && lock.pw_len + len < MAX_PASSWORD - 1) {
		memcpy(lock.password + lock.pw_len, buf, len);
		lock.pw_len += len;
		lock.password[lock.pw_len] = '\0';
		lock.auth_failed = false;
		lock.status_msg[0] = '\0';
		mark_all_dirty();
	}
}

static void kbd_modifiers(void *data, struct wl_keyboard *kbd, uint32_t serial,
		uint32_t depressed, uint32_t latched, uint32_t locked_mods,
		uint32_t group) {
	(void)data; (void)kbd; (void)serial;
	if (lock.xkb_state)
		xkb_state_update_mask(lock.xkb_state, depressed, latched,
			locked_mods, 0, 0, group);
}

static void kbd_repeat_info(void *data, struct wl_keyboard *kbd,
		int32_t rate, int32_t delay) {
	(void)data; (void)kbd; (void)rate; (void)delay;
}

static const struct wl_keyboard_listener keyboard_listener = {
	.keymap = kbd_keymap,
	.enter = kbd_enter,
	.leave = kbd_leave,
	.key = kbd_key,
	.modifiers = kbd_modifiers,
	.repeat_info = kbd_repeat_info,
};

/* ── Seat ── */

static void seat_caps(void *data, struct wl_seat *s, uint32_t caps) {
	(void)data;
	if (caps & WL_SEAT_CAPABILITY_KEYBOARD) {
		if (keyboard) wl_keyboard_destroy(keyboard);
		keyboard = wl_seat_get_keyboard(s);
		wl_keyboard_add_listener(keyboard, &keyboard_listener, NULL);
	}
}

static void seat_name(void *data, struct wl_seat *s, const char *name) {
	(void)data; (void)s; (void)name;
}

static const struct wl_seat_listener seat_listener = {
	.capabilities = seat_caps,
	.name = seat_name,
};

/* ── ext-session-lock-v1 surface ── */

static void lock_surface_configure(void *data,
		struct ext_session_lock_surface_v1 *surface,
		uint32_t serial, uint32_t width, uint32_t height) {
	struct output_ctx *ctx = data;
	ext_session_lock_surface_v1_ack_configure(surface, serial);

	if ((int)width != ctx->width || (int)height != ctx->height)
		create_buffer(ctx, width, height);

	ctx->configured = true;
	ctx->needs_redraw = true;
}

static const struct ext_session_lock_surface_v1_listener
lock_surface_listener = {
	.configure = lock_surface_configure,
};

/* ── ext-session-lock-v1 session lock ── */

static void session_lock_locked(void *data,
		struct ext_session_lock_v1 *lock_obj) {
	(void)data; (void)lock_obj;
	/* Compositor confirmed: all outputs are locked, input is isolated */
	lock.locked = true;
}

static void session_lock_finished(void *data,
		struct ext_session_lock_v1 *lock_obj) {
	(void)data; (void)lock_obj;
	/* Compositor denied the lock (e.g. another locker is running) */
	fprintf(stderr, "marshal-locker: compositor denied lock request\n");
	lock.locked = false;
}

static const struct ext_session_lock_v1_listener session_lock_listener = {
	.locked = session_lock_locked,
	.finished = session_lock_finished,
};

/* ── Registry ── */

static void registry_global(void *data, struct wl_registry *reg,
		uint32_t name, const char *interface, uint32_t version) {
	(void)data; (void)version;
	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		compositor = wl_registry_bind(reg, name,
			&wl_compositor_interface, 4);
	} else if (strcmp(interface, wl_shm_interface.name) == 0) {
		shm = wl_registry_bind(reg, name, &wl_shm_interface, 1);
	} else if (strcmp(interface, wl_seat_interface.name) == 0) {
		seat = wl_registry_bind(reg, name, &wl_seat_interface, 5);
		wl_seat_add_listener(seat, &seat_listener, NULL);
	} else if (strcmp(interface, wl_output_interface.name) == 0) {
		if (output_count < MAX_LOCK_OUTPUTS) {
			struct output_ctx *ctx = &outputs[output_count++];
			memset(ctx, 0, sizeof(*ctx));
			ctx->name = name;
			ctx->wl_output = wl_registry_bind(reg, name,
				&wl_output_interface, 1);
		} else {
			fprintf(stderr,
				"marshal-locker: too many outputs (>%d), ignoring\n",
				MAX_LOCK_OUTPUTS);
		}
	} else if (strcmp(interface,
			ext_session_lock_manager_v1_interface.name) == 0) {
		lock_manager = wl_registry_bind(reg, name,
			&ext_session_lock_manager_v1_interface, 1);
	}
}

static void registry_remove(void *data, struct wl_registry *reg,
		uint32_t name) {
	(void)data; (void)reg; (void)name;
}

static const struct wl_registry_listener registry_listener = {
	.global = registry_global,
	.global_remove = registry_remove,
};

/* ── Main ── */

int main(int argc, char *argv[]) {
	(void)argc; (void)argv;

	/* Initialize fonts */
	lock.font_time   = pango_font_description_from_string(FONT_TIME);
	lock.font_date   = pango_font_description_from_string(FONT_DATE);
	lock.font_input  = pango_font_description_from_string(FONT_INPUT);
	lock.font_status = pango_font_description_from_string(FONT_STATUS);

	/* XKB */
	lock.xkb_ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);

	/* Wayland connect */
	display = wl_display_connect(NULL);
	if (!display) {
		fprintf(stderr, "marshal-locker: cannot connect to Wayland\n");
		return 1;
	}

	registry = wl_display_get_registry(display);
	wl_registry_add_listener(registry, &registry_listener, NULL);
	wl_display_roundtrip(display);  /* globals */
	wl_display_roundtrip(display);  /* initial events (seat caps, etc.) */

	if (!lock_manager) {
		fprintf(stderr,
			"marshal-locker: compositor lacks ext-session-lock-v1\n");
		return 1;
	}
	if (output_count == 0) {
		fprintf(stderr, "marshal-locker: no outputs available\n");
		return 1;
	}

	/* Lock the session — compositor blanks outputs and inhibits input */
	session_lock = ext_session_lock_manager_v1_lock(lock_manager);
	ext_session_lock_v1_add_listener(session_lock, &session_lock_listener,
		NULL);

	/* Create a lock surface on every output. ext-session-lock-v1 requires
	 * this before the compositor will send the `locked` event — miss one
	 * and we hang here forever. */
	for (int i = 0; i < output_count; i++) {
		struct output_ctx *ctx = &outputs[i];
		ctx->surface = wl_compositor_create_surface(compositor);
		ctx->lock_surface = ext_session_lock_v1_get_lock_surface(
			session_lock, ctx->surface, ctx->wl_output);
		ext_session_lock_surface_v1_add_listener(ctx->lock_surface,
			&lock_surface_listener, ctx);
		wl_surface_commit(ctx->surface);
	}

	/* Wait for every surface to be configured */
	for (;;) {
		bool all_configured = true;
		for (int i = 0; i < output_count; i++) {
			if (!outputs[i].configured) { all_configured = false; break; }
		}
		if (all_configured) break;
		if (wl_display_dispatch(display) < 0) {
			fprintf(stderr,
				"marshal-locker: display disconnected before configure\n");
			return 1;
		}
	}

	/* Render initial frame on all outputs */
	render_all();

	/* Wait for compositor to confirm lock */
	while (!lock.locked && wl_display_dispatch(display) >= 0)
		;

	if (!lock.locked) {
		fprintf(stderr, "marshal-locker: failed to acquire lock\n");
		return 1;
	}

	/* Main loop — locked, waiting for PAM auth */
	while (lock.locked) {
		render_all();
		if (wl_display_dispatch(display) < 0)
			break;
	}

	/* Successful auth — unlock the session */
	ext_session_lock_v1_unlock_and_destroy(session_lock);
	wl_display_roundtrip(display);

	/* Cleanup — securely clear password */
	explicit_bzero(lock.password, sizeof(lock.password));

	for (int i = 0; i < output_count; i++) {
		struct output_ctx *ctx = &outputs[i];
		if (ctx->lock_surface) ext_session_lock_surface_v1_destroy(ctx->lock_surface);
		if (ctx->surface) wl_surface_destroy(ctx->surface);
		if (ctx->buffer) wl_buffer_destroy(ctx->buffer);
		if (ctx->shm_data) munmap(ctx->shm_data, ctx->shm_size);
		if (ctx->cr) cairo_destroy(ctx->cr);
		if (ctx->cairo_surface) cairo_surface_destroy(ctx->cairo_surface);
		if (ctx->wl_output) wl_output_release(ctx->wl_output);
	}
	if (lock.xkb_state) xkb_state_unref(lock.xkb_state);
	if (lock.xkb_keymap) xkb_keymap_unref(lock.xkb_keymap);
	if (lock.xkb_ctx) xkb_context_unref(lock.xkb_ctx);
	pango_font_description_free(lock.font_time);
	pango_font_description_free(lock.font_date);
	pango_font_description_free(lock.font_input);
	pango_font_description_free(lock.font_status);
	if (lock_manager)
		ext_session_lock_manager_v1_destroy(lock_manager);
	if (keyboard) wl_keyboard_destroy(keyboard);
	if (seat) wl_seat_destroy(seat);
	if (shm) wl_shm_destroy(shm);
	if (compositor) wl_compositor_destroy(compositor);
	if (registry) wl_registry_destroy(registry);
	if (display) wl_display_disconnect(display);

	return 0;
}
