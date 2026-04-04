/*
 * leaves-notifyd — Notification daemon for Leaves OS.
 *
 * Implements org.freedesktop.Notifications D-Bus interface.
 * Renders notifications as layer-shell popup surfaces using Cairo/Pango.
 * Integrates with the Leaves intent feed via Unix socket.
 *
 * Architecture:
 *   - sd-bus for D-Bus interface
 *   - Layer-shell TOP layer surfaces for notification popups
 *   - Auto-dismiss after timeout (default 5s)
 *   - Max 3 visible notifications, stacked from top-right
 *
 * Build: meson (see meson.build in this directory)
 */

#define _GNU_SOURCE
#include <cairo.h>
#include <errno.h>
#include <math.h>
#include <pango/pangocairo.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <systemd/sd-bus.h>
#include <time.h>
#include <unistd.h>
#include <wayland-client.h>

#include "wlr-layer-shell-unstable-v1-client-protocol.h"

/* ── Configuration ── */

#define MAX_NOTIFICATIONS  3
#define NOTIF_WIDTH        360
#define NOTIF_HEIGHT       88
#define NOTIF_GAP          8
#define NOTIF_MARGIN       16
#define NOTIF_RADIUS       10
#define NOTIF_PAD          14
#define NOTIF_TIMEOUT_MS   5000
#define FONT_TITLE         "Geist Semi-Bold 14"
#define FONT_BODY          "Geist 13"
#define FONT_APP           "Geist 11"

/* ── Wayland globals ── */

static struct wl_display    *wl_display;
static struct wl_registry   *wl_registry;
static struct wl_compositor *wl_compositor;
static struct wl_shm        *wl_shm;
static struct wl_output     *wl_output;
static struct zwlr_layer_shell_v1 *layer_shell;

/* ── Notification data ── */

struct notification {
	uint32_t id;
	char     summary[256];
	char     body[512];
	char     app_name[128];
	int      timeout_ms;
	uint64_t show_time_ms;

	/* Wayland surface */
	struct wl_surface            *surface;
	struct zwlr_layer_surface_v1 *layer_surface;
	struct wl_buffer             *buffer;
	void   *shm_data;
	int     shm_size;
	bool    configured;
	bool    active;
};

static struct notification notifications[MAX_NOTIFICATIONS];
static uint32_t next_id = 1;

/* ── Time helper ── */

static uint64_t now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* ── SHM buffer ── */

static int create_shm_file(size_t size) {
	int fd = memfd_create("leaves-notif", MFD_CLOEXEC);
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

static bool create_notif_buffer(struct notification *n) {
	int stride = NOTIF_WIDTH * 4;
	int size = stride * NOTIF_HEIGHT;

	int fd = create_shm_file(size);
	if (fd < 0) return false;

	n->shm_data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (n->shm_data == MAP_FAILED) { close(fd); return false; }
	n->shm_size = size;

	struct wl_shm_pool *pool = wl_shm_create_pool(wl_shm, fd, size);
	n->buffer = wl_shm_pool_create_buffer(pool, 0, NOTIF_WIDTH, NOTIF_HEIGHT,
		stride, WL_SHM_FORMAT_ARGB8888);
	wl_shm_pool_destroy(pool);
	close(fd);

	wl_buffer_add_listener(n->buffer, &buffer_listener, NULL);
	return true;
}

/* ── Notification rendering ── */

static void render_notification(struct notification *n) {
	if (!n->shm_data || !n->buffer) return;

	int w = NOTIF_WIDTH;
	int h = NOTIF_HEIGHT;
	int stride = w * 4;

	cairo_surface_t *cs = cairo_image_surface_create_for_data(
		n->shm_data, CAIRO_FORMAT_ARGB32, w, h, stride);
	cairo_t *cr = cairo_create(cs);

	/* Clear */
	cairo_set_operator(cr, CAIRO_OPERATOR_SOURCE);
	cairo_set_source_rgba(cr, 0, 0, 0, 0);
	cairo_paint(cr);
	cairo_set_operator(cr, CAIRO_OPERATOR_OVER);

	/* Rounded rectangle background */
	double r = NOTIF_RADIUS;
	cairo_new_sub_path(cr);
	cairo_arc(cr, w - r, r, r, -M_PI / 2, 0);
	cairo_arc(cr, w - r, h - r, r, 0, M_PI / 2);
	cairo_arc(cr, r, h - r, r, M_PI / 2, M_PI);
	cairo_arc(cr, r, r, r, M_PI, 3 * M_PI / 2);
	cairo_close_path(cr);

	/* Background with slight transparency */
	cairo_set_source_rgba(cr, 0.12, 0.12, 0.14, 0.95);
	cairo_fill_preserve(cr);

	/* Border */
	cairo_set_source_rgba(cr, 1.0, 1.0, 1.0, 0.12);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);

	/* Accent bar on left */
	cairo_set_source_rgba(cr, 0.15, 0.39, 0.92, 1.0); /* Leaves accent blue */
	cairo_rectangle(cr, 0, NOTIF_RADIUS, 3, h - 2 * NOTIF_RADIUS);
	cairo_fill(cr);

	int text_x = NOTIF_PAD + 4;  /* after accent bar */
	int text_w = w - text_x - NOTIF_PAD;

	/* App name */
	if (n->app_name[0]) {
		PangoFontDescription *fd = pango_font_description_from_string(FONT_APP);
		PangoLayout *layout = pango_cairo_create_layout(cr);
		pango_layout_set_font_description(layout, fd);
		pango_layout_set_text(layout, n->app_name, -1);
		pango_layout_set_width(layout, text_w * PANGO_SCALE);
		pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);

		cairo_set_source_rgba(cr, 0.55, 0.55, 0.58, 1.0);
		cairo_move_to(cr, text_x, NOTIF_PAD - 2);
		pango_cairo_show_layout(cr, layout);

		g_object_unref(layout);
		pango_font_description_free(fd);
	}

	/* Summary (title) */
	int title_y = NOTIF_PAD + 14;
	{
		PangoFontDescription *fd = pango_font_description_from_string(FONT_TITLE);
		PangoLayout *layout = pango_cairo_create_layout(cr);
		pango_layout_set_font_description(layout, fd);
		pango_layout_set_text(layout, n->summary, -1);
		pango_layout_set_width(layout, text_w * PANGO_SCALE);
		pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);

		cairo_set_source_rgba(cr, 0.93, 0.93, 0.94, 1.0);
		cairo_move_to(cr, text_x, title_y);
		pango_cairo_show_layout(cr, layout);

		g_object_unref(layout);
		pango_font_description_free(fd);
	}

	/* Body */
	if (n->body[0]) {
		PangoFontDescription *fd = pango_font_description_from_string(FONT_BODY);
		PangoLayout *layout = pango_cairo_create_layout(cr);
		pango_layout_set_font_description(layout, fd);
		pango_layout_set_text(layout, n->body, -1);
		pango_layout_set_width(layout, text_w * PANGO_SCALE);
		pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
		pango_layout_set_height(layout, 2 * PANGO_SCALE); /* max 2 lines */

		cairo_set_source_rgba(cr, 0.70, 0.70, 0.72, 1.0);
		cairo_move_to(cr, text_x, title_y + 22);
		pango_cairo_show_layout(cr, layout);

		g_object_unref(layout);
		pango_font_description_free(fd);
	}

	cairo_surface_flush(cs);
	cairo_destroy(cr);
	cairo_surface_destroy(cs);

	wl_surface_attach(n->surface, n->buffer, 0, 0);
	wl_surface_damage_buffer(n->surface, 0, 0, w, h);
	wl_surface_commit(n->surface);
}

/* ── Layer surface callbacks ── */

static void notif_layer_configure(void *data,
		struct zwlr_layer_surface_v1 *surface,
		uint32_t serial, uint32_t width, uint32_t height) {
	(void)width; (void)height;
	struct notification *n = data;
	zwlr_layer_surface_v1_ack_configure(surface, serial);
	n->configured = true;
}

static void notif_layer_closed(void *data,
		struct zwlr_layer_surface_v1 *surface) {
	(void)surface;
	struct notification *n = data;
	n->active = false;
}

static const struct zwlr_layer_surface_v1_listener notif_layer_listener = {
	.configure = notif_layer_configure,
	.closed = notif_layer_closed,
};

/* ── Notification lifecycle ── */

static int find_slot(void) {
	for (int i = 0; i < MAX_NOTIFICATIONS; i++) {
		if (!notifications[i].active) return i;
	}
	/* Evict oldest */
	int oldest = 0;
	uint64_t oldest_time = UINT64_MAX;
	for (int i = 0; i < MAX_NOTIFICATIONS; i++) {
		if (notifications[i].show_time_ms < oldest_time) {
			oldest_time = notifications[i].show_time_ms;
			oldest = i;
		}
	}
	return oldest;
}

static void dismiss_notification(struct notification *n) {
	if (!n->active) return;
	n->active = false;
	if (n->layer_surface) {
		zwlr_layer_surface_v1_destroy(n->layer_surface);
		n->layer_surface = NULL;
	}
	if (n->surface) {
		wl_surface_destroy(n->surface);
		n->surface = NULL;
	}
	if (n->buffer) {
		wl_buffer_destroy(n->buffer);
		n->buffer = NULL;
	}
	if (n->shm_data) {
		munmap(n->shm_data, n->shm_size);
		n->shm_data = NULL;
	}
}

static uint32_t show_notification(const char *app, const char *summary,
		const char *body, int timeout_ms) {
	int slot = find_slot();
	struct notification *n = &notifications[slot];

	/* Dismiss if reusing slot */
	if (n->active) dismiss_notification(n);

	n->id = next_id++;
	snprintf(n->summary, sizeof(n->summary), "%s", summary);
	snprintf(n->body, sizeof(n->body), "%s", body);
	snprintf(n->app_name, sizeof(n->app_name), "%s", app ? app : "");
	n->timeout_ms = timeout_ms > 0 ? timeout_ms : NOTIF_TIMEOUT_MS;
	n->show_time_ms = now_ms();
	n->active = true;
	n->configured = false;

	/* Calculate vertical offset based on slot position */
	int y_offset = NOTIF_MARGIN + slot * (NOTIF_HEIGHT + NOTIF_GAP);

	/* Create Wayland surface */
	n->surface = wl_compositor_create_surface(wl_compositor);
	n->layer_surface = zwlr_layer_shell_v1_get_layer_surface(
		layer_shell, n->surface, wl_output,
		ZWLR_LAYER_SHELL_V1_LAYER_TOP, "leaves-notification");

	zwlr_layer_surface_v1_set_size(n->layer_surface, NOTIF_WIDTH, NOTIF_HEIGHT);
	zwlr_layer_surface_v1_set_anchor(n->layer_surface,
		ZWLR_LAYER_SURFACE_V1_ANCHOR_TOP |
		ZWLR_LAYER_SURFACE_V1_ANCHOR_RIGHT);
	zwlr_layer_surface_v1_set_margin(n->layer_surface,
		y_offset, NOTIF_MARGIN, 0, 0);
	zwlr_layer_surface_v1_add_listener(n->layer_surface,
		&notif_layer_listener, n);

	wl_surface_commit(n->surface);

	/* Wait for configure, then create buffer and render */
	wl_display_roundtrip(wl_display);

	if (n->configured) {
		create_notif_buffer(n);
		render_notification(n);
	}

	return n->id;
}

/* ── D-Bus interface: org.freedesktop.Notifications ── */

static int method_notify(sd_bus_message *m, void *userdata, sd_bus_error *error) {
	(void)userdata; (void)error;

	const char *app_name, *app_icon, *summary, *body;
	uint32_t replaces_id;
	int32_t timeout;

	int r = sd_bus_message_read(m, "susss", &app_name, &replaces_id,
		&app_icon, &summary, &body);
	if (r < 0) return r;

	/* Skip actions array */
	r = sd_bus_message_skip(m, "as");
	if (r < 0) return r;

	/* Skip hints dict */
	r = sd_bus_message_skip(m, "a{sv}");
	if (r < 0) return r;

	/* Read timeout */
	r = sd_bus_message_read(m, "i", &timeout);
	if (r < 0) return r;

	uint32_t id = show_notification(app_name, summary, body, timeout);

	return sd_bus_reply_method_return(m, "u", id);
}

static int method_close(sd_bus_message *m, void *userdata, sd_bus_error *error) {
	(void)userdata; (void)error;
	uint32_t id;
	int r = sd_bus_message_read(m, "u", &id);
	if (r < 0) return r;

	for (int i = 0; i < MAX_NOTIFICATIONS; i++) {
		if (notifications[i].active && notifications[i].id == id) {
			dismiss_notification(&notifications[i]);
			break;
		}
	}
	return sd_bus_reply_method_return(m, "");
}

static int method_get_capabilities(sd_bus_message *m, void *userdata,
		sd_bus_error *error) {
	(void)userdata; (void)error;
	sd_bus_message *reply = NULL;
	int r = sd_bus_message_new_method_return(m, &reply);
	if (r < 0) return r;

	r = sd_bus_message_open_container(reply, 'a', "s");
	if (r < 0) return r;
	sd_bus_message_append(reply, "s", "body");
	sd_bus_message_append(reply, "s", "persistence");
	r = sd_bus_message_close_container(reply);
	if (r < 0) return r;

	return sd_bus_send(NULL, reply, NULL);
}

static int method_get_server_info(sd_bus_message *m, void *userdata,
		sd_bus_error *error) {
	(void)userdata; (void)error;
	return sd_bus_reply_method_return(m, "ssss",
		"leaves-notifyd",     /* name */
		"Leaves OS",          /* vendor */
		"0.1.0",              /* version */
		"1.2");               /* spec version */
}

static const sd_bus_vtable notifications_vtable[] = {
	SD_BUS_VTABLE_START(0),
	SD_BUS_METHOD("Notify", "susssasa{sv}i", "u", method_notify, SD_BUS_VTABLE_UNPRIVILEGED),
	SD_BUS_METHOD("CloseNotification", "u", "", method_close, SD_BUS_VTABLE_UNPRIVILEGED),
	SD_BUS_METHOD("GetCapabilities", "", "as", method_get_capabilities, SD_BUS_VTABLE_UNPRIVILEGED),
	SD_BUS_METHOD("GetServerInformation", "", "ssss", method_get_server_info, SD_BUS_VTABLE_UNPRIVILEGED),
	SD_BUS_VTABLE_END,
};

/* ── Wayland registry ── */

static void registry_global(void *data, struct wl_registry *reg,
		uint32_t name, const char *interface, uint32_t version) {
	(void)data; (void)version;
	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		wl_compositor = wl_registry_bind(reg, name, &wl_compositor_interface, 4);
	} else if (strcmp(interface, wl_shm_interface.name) == 0) {
		wl_shm = wl_registry_bind(reg, name, &wl_shm_interface, 1);
	} else if (strcmp(interface, wl_output_interface.name) == 0) {
		if (!wl_output)
			wl_output = wl_registry_bind(reg, name, &wl_output_interface, 1);
	} else if (strcmp(interface, zwlr_layer_shell_v1_interface.name) == 0) {
		layer_shell = wl_registry_bind(reg, name, &zwlr_layer_shell_v1_interface, 1);
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

	/* Wayland connection */
	wl_display = wl_display_connect(NULL);
	if (!wl_display) {
		fprintf(stderr, "leaves-notifyd: cannot connect to Wayland\n");
		return 1;
	}

	wl_registry = wl_display_get_registry(wl_display);
	wl_registry_add_listener(wl_registry, &registry_listener, NULL);
	wl_display_roundtrip(wl_display);

	if (!layer_shell) {
		fprintf(stderr, "leaves-notifyd: compositor lacks layer-shell support\n");
		return 1;
	}

	/* D-Bus session bus */
	sd_bus *bus = NULL;
	int r = sd_bus_open_user(&bus);
	if (r < 0) {
		fprintf(stderr, "leaves-notifyd: failed to connect to session bus: %s\n",
			strerror(-r));
		return 1;
	}

	r = sd_bus_add_object_vtable(bus,
		NULL,
		"/org/freedesktop/Notifications",
		"org.freedesktop.Notifications",
		notifications_vtable,
		NULL);
	if (r < 0) {
		fprintf(stderr, "leaves-notifyd: failed to add vtable: %s\n", strerror(-r));
		return 1;
	}

	r = sd_bus_request_name(bus, "org.freedesktop.Notifications",
		SD_BUS_NAME_REPLACE_EXISTING | SD_BUS_NAME_ALLOW_REPLACEMENT);
	if (r < 0) {
		fprintf(stderr, "leaves-notifyd: failed to acquire bus name: %s\n",
			strerror(-r));
		return 1;
	}

	fprintf(stderr, "leaves-notifyd: listening on org.freedesktop.Notifications\n");

	/* Main event loop: poll both Wayland and D-Bus fds */
	int wl_fd = wl_display_get_fd(wl_display);
	int bus_fd = sd_bus_get_fd(bus);

	for (;;) {
		/* Check notification timeouts */
		uint64_t now = now_ms();
		for (int i = 0; i < MAX_NOTIFICATIONS; i++) {
			if (notifications[i].active &&
					now - notifications[i].show_time_ms > (uint64_t)notifications[i].timeout_ms) {
				dismiss_notification(&notifications[i]);
			}
		}

		/* Flush Wayland */
		wl_display_flush(wl_display);

		/* Process pending D-Bus messages */
		while (sd_bus_process(bus, NULL) > 0)
			;

		struct pollfd fds[2] = {
			{ .fd = wl_fd,  .events = POLLIN },
			{ .fd = bus_fd, .events = POLLIN },
		};

		int timeout = 1000;  /* check timeouts every second */
		poll(fds, 2, timeout);

		if (fds[0].revents & POLLIN) {
			wl_display_dispatch(wl_display);
		}
		/* D-Bus events are processed at top of loop */
	}

	sd_bus_unref(bus);
	wl_display_disconnect(wl_display);
	return 0;
}
