/*
 * leaves-locker — Wayland screen locker for Leaves OS.
 *
 * Architecture:
 *   - Layer-shell overlay surface (covers entire screen)
 *   - wlr-input-inhibitor grabs all input
 *   - PAM authentication via pam_authenticate()
 *   - Cairo + Pango rendering with Leaves visual language
 *   - Triggered by compositor (SIGUSR1) or keyboard shortcut (Super+L)
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

#include "wlr-layer-shell-unstable-v1-client-protocol.h"
#include "wlr-input-inhibitor-unstable-v1-client-protocol.h"

/* ── Configuration ── */

#define MAX_PASSWORD    256
#define FONT_TIME       "Geist Semi-Bold 64"
#define FONT_DATE       "Geist 18"
#define FONT_INPUT      "Geist 16"
#define FONT_STATUS     "Geist 13"

/* ── Wayland globals ── */

static struct wl_display    *wl_display;
static struct wl_registry   *wl_registry;
static struct wl_compositor *wl_compositor;
static struct wl_shm        *wl_shm;
static struct wl_seat       *wl_seat;
static struct wl_keyboard   *wl_keyboard;
static struct wl_output     *wl_output;
static struct zwlr_layer_shell_v1          *layer_shell;
static struct zwlr_input_inhibit_manager_v1 *inhibit_manager;
static struct zwlr_input_inhibitor_v1       *inhibitor;

/* ── Locker state ── */

struct locker {
	struct wl_surface              *surface;
	struct zwlr_layer_surface_v1   *layer_surface;

	/* SHM buffer */
	struct wl_buffer *buffer;
	void   *shm_data;
	int     shm_size;
	int     width;
	int     height;

	/* Cairo */
	cairo_surface_t      *cairo_surface;
	cairo_t              *cr;
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

	/* State */
	bool configured;
	bool locked;
	bool needs_redraw;
};

static struct locker lock;

/* ── SHM buffer creation ── */

static int create_shm_file(size_t size) {
	char name[] = "/leaves-lock-XXXXXX";
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

static bool create_buffer(int width, int height) {
	int stride = width * 4;
	int size = stride * height;

	if (lock.buffer) {
		wl_buffer_destroy(lock.buffer);
		munmap(lock.shm_data, lock.shm_size);
	}

	int fd = create_shm_file(size);
	if (fd < 0) return false;

	lock.shm_data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (lock.shm_data == MAP_FAILED) { close(fd); return false; }
	lock.shm_size = size;

	struct wl_shm_pool *pool = wl_shm_create_pool(wl_shm, fd, size);
	lock.buffer = wl_shm_pool_create_buffer(pool, 0, width, height,
		stride, WL_SHM_FORMAT_ARGB8888);
	wl_shm_pool_destroy(pool);
	close(fd);

	wl_buffer_add_listener(lock.buffer, &buffer_listener, NULL);

	if (lock.cr) cairo_destroy(lock.cr);
	if (lock.cairo_surface) cairo_surface_destroy(lock.cairo_surface);

	lock.cairo_surface = cairo_image_surface_create_for_data(
		lock.shm_data, CAIRO_FORMAT_ARGB32, width, height, stride);
	lock.cr = cairo_create(lock.cairo_surface);

	lock.width = width;
	lock.height = height;
	return true;
}

/* ── Rendering ── */

static void render(void) {
	if (!lock.cr) return;
	cairo_t *cr = lock.cr;
	int w = lock.width;
	int h = lock.height;

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

	cairo_surface_flush(lock.cairo_surface);

	wl_surface_attach(lock.surface, lock.buffer, 0, 0);
	wl_surface_damage_buffer(lock.surface, 0, 0, w, h);
	wl_surface_commit(lock.surface);
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
		lock.needs_redraw = true;

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
		lock.needs_redraw = true;
		return;
	}

	if (sym == XKB_KEY_BackSpace) {
		if (lock.pw_len > 0) {
			lock.password[--lock.pw_len] = '\0';
			lock.auth_failed = false;
			lock.status_msg[0] = '\0';
			lock.needs_redraw = true;
		}
		return;
	}

	if (sym == XKB_KEY_Escape) {
		/* Clear input */
		lock.pw_len = 0;
		lock.password[0] = '\0';
		lock.auth_failed = false;
		lock.status_msg[0] = '\0';
		lock.needs_redraw = true;
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
		lock.needs_redraw = true;
	}
}

static void kbd_modifiers(void *data, struct wl_keyboard *kbd, uint32_t serial,
		uint32_t depressed, uint32_t latched, uint32_t locked, uint32_t group) {
	(void)data; (void)kbd; (void)serial;
	if (lock.xkb_state)
		xkb_state_update_mask(lock.xkb_state, depressed, latched, locked, 0, 0, group);
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

static void seat_caps(void *data, struct wl_seat *seat, uint32_t caps) {
	(void)data;
	if (caps & WL_SEAT_CAPABILITY_KEYBOARD) {
		if (wl_keyboard) wl_keyboard_destroy(wl_keyboard);
		wl_keyboard = wl_seat_get_keyboard(seat);
		wl_keyboard_add_listener(wl_keyboard, &keyboard_listener, NULL);
	}
}

static void seat_name(void *data, struct wl_seat *seat, const char *name) {
	(void)data; (void)seat; (void)name;
}

static const struct wl_seat_listener seat_listener = {
	.capabilities = seat_caps,
	.name = seat_name,
};

/* ── Layer surface ── */

static void layer_surface_configure(void *data,
		struct zwlr_layer_surface_v1 *surface,
		uint32_t serial, uint32_t width, uint32_t height) {
	(void)data;
	zwlr_layer_surface_v1_ack_configure(surface, serial);
	lock.configured = true;

	if ((int)width != lock.width || (int)height != lock.height) {
		create_buffer(width, height);
	}
	lock.needs_redraw = true;
}

static void layer_surface_closed(void *data,
		struct zwlr_layer_surface_v1 *surface) {
	(void)data; (void)surface;
	/* Should not happen for a locker — if it does, re-lock */
}

static const struct zwlr_layer_surface_v1_listener layer_surface_listener = {
	.configure = layer_surface_configure,
	.closed = layer_surface_closed,
};

/* ── Registry ── */

static void registry_global(void *data, struct wl_registry *reg,
		uint32_t name, const char *interface, uint32_t version) {
	(void)data; (void)version;
	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		wl_compositor = wl_registry_bind(reg, name, &wl_compositor_interface, 4);
	} else if (strcmp(interface, wl_shm_interface.name) == 0) {
		wl_shm = wl_registry_bind(reg, name, &wl_shm_interface, 1);
	} else if (strcmp(interface, wl_seat_interface.name) == 0) {
		wl_seat = wl_registry_bind(reg, name, &wl_seat_interface, 5);
		wl_seat_add_listener(wl_seat, &seat_listener, NULL);
	} else if (strcmp(interface, wl_output_interface.name) == 0) {
		if (!wl_output)
			wl_output = wl_registry_bind(reg, name, &wl_output_interface, 1);
	} else if (strcmp(interface, zwlr_layer_shell_v1_interface.name) == 0) {
		layer_shell = wl_registry_bind(reg, name,
			&zwlr_layer_shell_v1_interface, 1);
	} else if (strcmp(interface, zwlr_input_inhibit_manager_v1_interface.name) == 0) {
		inhibit_manager = wl_registry_bind(reg, name,
			&zwlr_input_inhibit_manager_v1_interface, 1);
	}
}

static void registry_remove(void *data, struct wl_registry *reg, uint32_t name) {
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
	wl_display = wl_display_connect(NULL);
	if (!wl_display) {
		fprintf(stderr, "leaves-locker: cannot connect to Wayland\n");
		return 1;
	}

	wl_registry = wl_display_get_registry(wl_display);
	wl_registry_add_listener(wl_registry, &registry_listener, NULL);
	wl_display_roundtrip(wl_display);

	if (!layer_shell) {
		fprintf(stderr, "leaves-locker: compositor lacks layer-shell support\n");
		return 1;
	}

	/* Grab input */
	if (inhibit_manager) {
		inhibitor = zwlr_input_inhibit_manager_v1_get_inhibitor(inhibit_manager);
	}

	/* Create layer surface — OVERLAY layer, covers all of screen */
	lock.surface = wl_compositor_create_surface(wl_compositor);
	lock.layer_surface = zwlr_layer_shell_v1_get_layer_surface(
		layer_shell, lock.surface, wl_output,
		ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, "leaves-locker");

	/* Anchor to all edges → full screen */
	zwlr_layer_surface_v1_set_anchor(lock.layer_surface,
		ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
		ZWLR_LAYER_SURFACE_V1_ANCHOR_BOTTOM |
		ZWLR_LAYER_SURFACE_V1_ANCHOR_LEFT |
		ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
	zwlr_layer_surface_v1_set_exclusive_zone(lock.layer_surface, -1);
	zwlr_layer_surface_v1_set_keyboard_interactivity(lock.layer_surface, 1);
	zwlr_layer_surface_v1_add_listener(lock.layer_surface,
		&layer_surface_listener, NULL);

	wl_surface_commit(lock.surface);

	/* Wait for configure */
	while (!lock.configured)
		wl_display_dispatch(wl_display);

	lock.locked = true;

	/* Main loop */
	while (lock.locked) {
		if (lock.needs_redraw) {
			render();
			lock.needs_redraw = false;
		}
		if (wl_display_dispatch(wl_display) < 0)
			break;
	}

	/* Cleanup — securely clear password */
	explicit_bzero(lock.password, sizeof(lock.password));

	if (inhibitor) zwlr_input_inhibitor_v1_destroy(inhibitor);
	if (lock.layer_surface) zwlr_layer_surface_v1_destroy(lock.layer_surface);
	if (lock.surface) wl_surface_destroy(lock.surface);
	if (lock.buffer) wl_buffer_destroy(lock.buffer);
	if (lock.shm_data) munmap(lock.shm_data, lock.shm_size);
	if (lock.cr) cairo_destroy(lock.cr);
	if (lock.cairo_surface) cairo_surface_destroy(lock.cairo_surface);
	if (lock.xkb_state) xkb_state_unref(lock.xkb_state);
	if (lock.xkb_keymap) xkb_keymap_unref(lock.xkb_keymap);
	if (lock.xkb_ctx) xkb_context_unref(lock.xkb_ctx);
	pango_font_description_free(lock.font_time);
	pango_font_description_free(lock.font_date);
	pango_font_description_free(lock.font_input);
	pango_font_description_free(lock.font_status);
	if (inhibit_manager) zwlr_input_inhibit_manager_v1_destroy(inhibit_manager);
	if (layer_shell) zwlr_layer_shell_v1_destroy(layer_shell);
	if (wl_keyboard) wl_keyboard_destroy(wl_keyboard);
	if (wl_seat) wl_seat_destroy(wl_seat);
	if (wl_output) wl_output_release(wl_output);
	if (wl_shm) wl_shm_destroy(wl_shm);
	if (wl_compositor) wl_compositor_destroy(wl_compositor);
	if (wl_registry) wl_registry_destroy(wl_registry);
	if (wl_display) wl_display_disconnect(wl_display);

	return 0;
}
