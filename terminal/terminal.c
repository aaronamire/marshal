/*
 * leaves-terminal — Minimal Wayland-native terminal emulator for Leaves OS.
 *
 * Architecture:
 *   - Wayland client via libwayland-client + xdg-shell
 *   - Terminal state machine via libtsm (VT100/ANSI escape handling)
 *   - PTY pair via posix_openpt() → fork/exec shell
 *   - Cairo + Pango rendering with Geist Mono font
 *   - Poll loop: PTY read → tsm_vte_input → tsm_screen update → render
 *
 * Build: meson (see meson.build in this directory)
 */

#define _GNU_SOURCE
#include <cairo.h>
#include <errno.h>
#include <fcntl.h>
#include <libtsm.h>
#include <locale.h>
#include <math.h>
#include <pango/pangocairo.h>
#include <poll.h>
#include <pty.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <wayland-client.h>
#include <xkbcommon/xkbcommon.h>

#include "xdg-shell-client-protocol.h"

/* ── Configuration ── */

#define TERM_FONT         "Geist Mono 14"
#define TERM_SCROLLBACK   4096
#define TERM_PAD_X        8
#define TERM_PAD_Y        4
#define TERM_INITIAL_COLS 80
#define TERM_INITIAL_ROWS 24
#define PTY_BUF_SIZE      16384
#define CURSOR_BLINK_MS   600

/* ── Colors (Leaves OS palette — dark terminal) ── */

struct rgba { double r, g, b, a; };

/* 16-color ANSI palette + Leaves accents */
static const struct rgba palette[18] = {
	/* 0  black   */ { 0.11, 0.11, 0.13, 1.0 },
	/* 1  red     */ { 0.86, 0.15, 0.15, 1.0 },
	/* 2  green   */ { 0.09, 0.64, 0.29, 1.0 },
	/* 3  yellow  */ { 0.85, 0.47, 0.02, 1.0 },
	/* 4  blue    */ { 0.15, 0.39, 0.92, 1.0 },
	/* 5  magenta */ { 0.66, 0.27, 0.73, 1.0 },
	/* 6  cyan    */ { 0.17, 0.64, 0.76, 1.0 },
	/* 7  white   */ { 0.80, 0.80, 0.82, 1.0 },
	/* 8  bright black   */ { 0.40, 0.40, 0.42, 1.0 },
	/* 9  bright red     */ { 0.94, 0.33, 0.33, 1.0 },
	/* 10 bright green   */ { 0.30, 0.78, 0.47, 1.0 },
	/* 11 bright yellow  */ { 0.95, 0.67, 0.24, 1.0 },
	/* 12 bright blue    */ { 0.42, 0.58, 0.93, 1.0 },
	/* 13 bright magenta */ { 0.79, 0.46, 0.85, 1.0 },
	/* 14 bright cyan    */ { 0.38, 0.78, 0.87, 1.0 },
	/* 15 bright white   */ { 0.93, 0.93, 0.94, 1.0 },
	/* 16 foreground     */ { 0.88, 0.88, 0.89, 1.0 },
	/* 17 background     */ { 0.09, 0.09, 0.11, 1.0 },
};

#define FG_DEFAULT 16
#define BG_DEFAULT 17

/* ── Wayland globals ── */

static struct wl_display    *wl_display;
static struct wl_registry   *wl_registry;
static struct wl_compositor *wl_compositor;
static struct wl_shm        *wl_shm;
static struct wl_seat       *wl_seat;
static struct wl_keyboard   *wl_keyboard;
static struct xdg_wm_base   *xdg_wm_base;

/* ── Terminal state ── */

struct terminal {
	/* Wayland surfaces */
	struct wl_surface     *surface;
	struct xdg_surface    *xdg_surface;
	struct xdg_toplevel   *xdg_toplevel;
	struct wl_callback    *frame_cb;

	/* Shared memory buffer */
	struct wl_buffer *buffer;
	void   *shm_data;
	int     shm_size;
	int     buf_width;
	int     buf_height;

	/* Cairo rendering */
	cairo_surface_t       *cairo_surface;
	cairo_t               *cr;
	PangoFontDescription  *font_desc;
	int cell_w;   /* glyph cell width in px */
	int cell_h;   /* glyph cell height in px */

	/* Terminal state machine */
	struct tsm_screen *screen;
	struct tsm_vte    *vte;

	/* PTY */
	int   pty_master;
	pid_t child_pid;

	/* Keyboard */
	struct xkb_context *xkb_ctx;
	struct xkb_keymap  *xkb_keymap;
	struct xkb_state   *xkb_state;

	/* Geometry */
	int cols;
	int rows;
	int width;   /* pixel width */
	int height;  /* pixel height */
	int scale;

	/* State flags */
	bool configured;
	bool closed;
	bool needs_redraw;
	bool cursor_visible;
	uint64_t last_blink_ms;
};

static struct terminal term;

/* ── Time helper ── */

static uint64_t now_ms(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* ── Shared memory buffer ── */

static int create_shm_file(size_t size) {
	char name[] = "/leaves-term-XXXXXX";
	int fd = memfd_create(name, MFD_CLOEXEC);
	if (fd < 0) return -1;
	if (ftruncate(fd, size) < 0) {
		close(fd);
		return -1;
	}
	return fd;
}

static void buffer_release(void *data, struct wl_buffer *buffer) {
	(void)data;
	(void)buffer;
}

static const struct wl_buffer_listener buffer_listener = {
	.release = buffer_release,
};

static bool create_buffer(int width, int height) {
	int stride = width * 4;
	int size = stride * height;

	if (term.buffer) {
		wl_buffer_destroy(term.buffer);
		munmap(term.shm_data, term.shm_size);
		term.buffer = NULL;
	}

	int fd = create_shm_file(size);
	if (fd < 0) return false;

	term.shm_data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (term.shm_data == MAP_FAILED) {
		close(fd);
		return false;
	}
	term.shm_size = size;

	struct wl_shm_pool *pool = wl_shm_create_pool(wl_shm, fd, size);
	term.buffer = wl_shm_pool_create_buffer(pool, 0, width, height,
		stride, WL_SHM_FORMAT_ARGB8888);
	wl_shm_pool_destroy(pool);
	close(fd);

	wl_buffer_add_listener(term.buffer, &buffer_listener, NULL);

	term.buf_width = width;
	term.buf_height = height;

	/* Recreate Cairo surface */
	if (term.cr) cairo_destroy(term.cr);
	if (term.cairo_surface) cairo_surface_destroy(term.cairo_surface);

	term.cairo_surface = cairo_image_surface_create_for_data(
		term.shm_data, CAIRO_FORMAT_ARGB32, width, height, stride);
	term.cr = cairo_create(term.cairo_surface);

	return true;
}

/* ── Font metrics ── */

static void measure_cell(void) {
	cairo_surface_t *tmp = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, 1, 1);
	cairo_t *cr = cairo_create(tmp);

	PangoLayout *layout = pango_cairo_create_layout(cr);
	pango_layout_set_font_description(layout, term.font_desc);
	pango_layout_set_text(layout, "M", 1);

	PangoRectangle ink, logical;
	pango_layout_get_pixel_extents(layout, &ink, &logical);

	term.cell_w = logical.width;
	term.cell_h = logical.height;

	g_object_unref(layout);
	cairo_destroy(cr);
	cairo_surface_destroy(tmp);
}

/* ── TSM callbacks ── */

static void tsm_log_cb(void *data, const char *file, int line,
		const char *func, const char *subs, unsigned int sev,
		const char *format, va_list args) {
	(void)data; (void)file; (void)line; (void)func; (void)subs; (void)sev;
	(void)format; (void)args;
	/* Silence TSM logs */
}

static void tsm_write_cb(struct tsm_vte *vte, const char *u8, size_t len,
		void *data) {
	(void)vte; (void)data;
	/* Write VTE output to PTY master */
	const char *p = u8;
	size_t remaining = len;
	while (remaining > 0) {
		ssize_t n = write(term.pty_master, p, remaining);
		if (n < 0) {
			if (errno == EINTR || errno == EAGAIN) continue;
			break;
		}
		p += n;
		remaining -= n;
	}
}

/* Color conversion from TSM color code to our RGBA palette */
static struct rgba color_from_tsm(int8_t code, bool is_fg) {
	if (code >= 0 && code < 16) return palette[code];
	return palette[is_fg ? FG_DEFAULT : BG_DEFAULT];
}

static struct rgba color_from_rgb(uint8_t r, uint8_t g, uint8_t b) {
	return (struct rgba){ r / 255.0, g / 255.0, b / 255.0, 1.0 };
}

/* ── Rendering ── */

static int draw_cell_cb(struct tsm_screen *con, uint64_t id,
		const uint32_t *ch, size_t len, unsigned int cwidth,
		unsigned int posx, unsigned int posy,
		const struct tsm_screen_attr *attr, tsm_age_t age,
		void *data) {
	(void)con; (void)id; (void)age; (void)data;

	cairo_t *cr = term.cr;
	int x = TERM_PAD_X + posx * term.cell_w;
	int y = TERM_PAD_Y + posy * term.cell_h;

	/* Determine colors */
	struct rgba fg, bg;
	if (attr->fccode >= 0)
		fg = color_from_tsm(attr->fccode, true);
	else if (attr->fr || attr->fg || attr->fb)
		fg = color_from_rgb(attr->fr, attr->fg, attr->fb);
	else
		fg = palette[FG_DEFAULT];

	if (attr->bccode >= 0)
		bg = color_from_tsm(attr->bccode, false);
	else if (attr->br || attr->bg || attr->bb)
		bg = color_from_rgb(attr->br, attr->bg, attr->bb);
	else
		bg = palette[BG_DEFAULT];

	/* Inverse video */
	if (attr->inverse) {
		struct rgba tmp = fg;
		fg = bg;
		bg = tmp;
	}

	/* Draw background */
	cairo_set_source_rgba(cr, bg.r, bg.g, bg.b, bg.a);
	cairo_rectangle(cr, x, y, term.cell_w * cwidth, term.cell_h);
	cairo_fill(cr);

	/* Draw character */
	if (len > 0 && ch[0] != 0 && ch[0] != ' ') {
		/* Convert UCS-4 to UTF-8 */
		char utf8[16];
		int utf8_len = 0;
		for (size_t i = 0; i < len && ch[i] != 0; i++) {
			uint32_t c = ch[i];
			if (c < 0x80) {
				utf8[utf8_len++] = c;
			} else if (c < 0x800) {
				utf8[utf8_len++] = 0xC0 | (c >> 6);
				utf8[utf8_len++] = 0x80 | (c & 0x3F);
			} else if (c < 0x10000) {
				utf8[utf8_len++] = 0xE0 | (c >> 12);
				utf8[utf8_len++] = 0x80 | ((c >> 6) & 0x3F);
				utf8[utf8_len++] = 0x80 | (c & 0x3F);
			} else {
				utf8[utf8_len++] = 0xF0 | (c >> 18);
				utf8[utf8_len++] = 0x80 | ((c >> 12) & 0x3F);
				utf8[utf8_len++] = 0x80 | ((c >> 6) & 0x3F);
				utf8[utf8_len++] = 0x80 | (c & 0x3F);
			}
			if (utf8_len >= 12) break;
		}

		PangoLayout *layout = pango_cairo_create_layout(cr);
		PangoFontDescription *fd = pango_font_description_copy(term.font_desc);

		if (attr->bold)
			pango_font_description_set_weight(fd, PANGO_WEIGHT_BOLD);
		if (attr->italic)
			pango_font_description_set_style(fd, PANGO_STYLE_ITALIC);

		pango_layout_set_font_description(layout, fd);
		pango_layout_set_text(layout, utf8, utf8_len);

		cairo_set_source_rgba(cr, fg.r, fg.g, fg.b, fg.a);
		cairo_move_to(cr, x, y);
		pango_cairo_show_layout(cr, layout);

		pango_font_description_free(fd);
		g_object_unref(layout);
	}

	/* Underline */
	if (attr->underline) {
		cairo_set_source_rgba(cr, fg.r, fg.g, fg.b, fg.a);
		cairo_set_line_width(cr, 1.0);
		cairo_move_to(cr, x, y + term.cell_h - 1.5);
		cairo_line_to(cr, x + term.cell_w * cwidth, y + term.cell_h - 1.5);
		cairo_stroke(cr);
	}

	return 0;
}

static void draw_cursor(void) {
	if (!term.cursor_visible) return;

	unsigned int cx = tsm_screen_get_cursor_x(term.screen);
	unsigned int cy = tsm_screen_get_cursor_y(term.screen);
	int x = TERM_PAD_X + cx * term.cell_w;
	int y = TERM_PAD_Y + cy * term.cell_h;

	cairo_t *cr = term.cr;
	/* Block cursor with Leaves accent blue */
	cairo_set_source_rgba(cr, 0.15, 0.39, 0.92, 0.85);
	cairo_rectangle(cr, x, y, term.cell_w, term.cell_h);
	cairo_fill(cr);
}

static void render(void) {
	if (!term.cr || !term.buffer) return;

	/* Clear with background color */
	cairo_set_source_rgba(term.cr, palette[BG_DEFAULT].r,
		palette[BG_DEFAULT].g, palette[BG_DEFAULT].b, 1.0);
	cairo_paint(term.cr);

	/* Draw all cells */
	tsm_screen_draw(term.screen, draw_cell_cb, NULL);

	/* Draw cursor */
	draw_cursor();

	cairo_surface_flush(term.cairo_surface);

	/* Attach and commit */
	wl_surface_attach(term.surface, term.buffer, 0, 0);
	wl_surface_damage_buffer(term.surface, 0, 0, term.buf_width, term.buf_height);
	wl_surface_commit(term.surface);
}

/* ── Frame callback (render loop) ── */

static void frame_done(void *data, struct wl_callback *cb, uint32_t time);

static const struct wl_callback_listener frame_listener = {
	.done = frame_done,
};

static void request_frame(void) {
	if (term.frame_cb) return;
	term.frame_cb = wl_surface_frame(term.surface);
	wl_callback_add_listener(term.frame_cb, &frame_listener, NULL);
	wl_surface_commit(term.surface);
}

static void frame_done(void *data, struct wl_callback *cb, uint32_t time) {
	(void)data; (void)time;
	wl_callback_destroy(cb);
	term.frame_cb = NULL;

	/* Blink cursor */
	uint64_t now = now_ms();
	if (now - term.last_blink_ms >= CURSOR_BLINK_MS) {
		term.cursor_visible = !term.cursor_visible;
		term.last_blink_ms = now;
		term.needs_redraw = true;
	}

	if (term.needs_redraw) {
		render();
		term.needs_redraw = false;
	}

	request_frame();
}

/* ── Resize handling ── */

static void resize_terminal(int width, int height) {
	term.width = width;
	term.height = height;

	int new_cols = (width - 2 * TERM_PAD_X) / term.cell_w;
	int new_rows = (height - 2 * TERM_PAD_Y) / term.cell_h;
	if (new_cols < 2) new_cols = 2;
	if (new_rows < 2) new_rows = 2;

	if (new_cols != term.cols || new_rows != term.rows) {
		term.cols = new_cols;
		term.rows = new_rows;
		tsm_screen_resize(term.screen, new_cols, new_rows);

		/* Notify child of new size */
		struct winsize ws = {
			.ws_row = new_rows,
			.ws_col = new_cols,
			.ws_xpixel = width,
			.ws_ypixel = height,
		};
		ioctl(term.pty_master, TIOCSWINSZ, &ws);
	}

	create_buffer(width, height);
	term.needs_redraw = true;
}

/* ── XDG surface / toplevel ── */

static void xdg_surface_configure(void *data, struct xdg_surface *xdg_surface,
		uint32_t serial) {
	(void)data;
	xdg_surface_ack_configure(xdg_surface, serial);
	term.configured = true;

	if (term.width > 0 && term.height > 0) {
		resize_terminal(term.width, term.height);
	}
}

static const struct xdg_surface_listener xdg_surface_listener = {
	.configure = xdg_surface_configure,
};

static void xdg_toplevel_configure(void *data, struct xdg_toplevel *toplevel,
		int32_t width, int32_t height, struct wl_array *states) {
	(void)data; (void)toplevel; (void)states;
	if (width > 0 && height > 0) {
		term.width = width;
		term.height = height;
	}
}

static void xdg_toplevel_close(void *data, struct xdg_toplevel *toplevel) {
	(void)data; (void)toplevel;
	term.closed = true;
}

static void xdg_toplevel_configure_bounds(void *data,
		struct xdg_toplevel *toplevel, int32_t width, int32_t height) {
	(void)data; (void)toplevel; (void)width; (void)height;
}

static void xdg_toplevel_wm_capabilities(void *data,
		struct xdg_toplevel *toplevel, struct wl_array *capabilities) {
	(void)data; (void)toplevel; (void)capabilities;
}

static const struct xdg_toplevel_listener xdg_toplevel_listener = {
	.configure = xdg_toplevel_configure,
	.close = xdg_toplevel_close,
	.configure_bounds = xdg_toplevel_configure_bounds,
	.wm_capabilities = xdg_toplevel_wm_capabilities,
};

/* ── XDG WM base ── */

static void xdg_wm_base_ping(void *data, struct xdg_wm_base *base,
		uint32_t serial) {
	(void)data;
	xdg_wm_base_pong(base, serial);
}

static const struct xdg_wm_base_listener xdg_wm_base_listener = {
	.ping = xdg_wm_base_ping,
};

/* ── Keyboard handling ── */

static void kbd_keymap(void *data, struct wl_keyboard *kbd,
		uint32_t format, int fd, uint32_t size) {
	(void)data; (void)kbd;
	if (format != WL_KEYBOARD_KEYMAP_FORMAT_XKB_V1) {
		close(fd);
		return;
	}
	char *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (map == MAP_FAILED) {
		close(fd);
		return;
	}

	if (term.xkb_keymap) xkb_keymap_unref(term.xkb_keymap);
	if (term.xkb_state) xkb_state_unref(term.xkb_state);

	term.xkb_keymap = xkb_keymap_new_from_string(term.xkb_ctx, map,
		XKB_KEYMAP_FORMAT_TEXT_V1, XKB_KEYMAP_COMPILE_NO_FLAGS);
	munmap(map, size);
	close(fd);

	if (term.xkb_keymap)
		term.xkb_state = xkb_state_new(term.xkb_keymap);
}

static void kbd_enter(void *data, struct wl_keyboard *kbd, uint32_t serial,
		struct wl_surface *surface, struct wl_array *keys) {
	(void)data; (void)kbd; (void)serial; (void)surface; (void)keys;
	term.cursor_visible = true;
}

static void kbd_leave(void *data, struct wl_keyboard *kbd, uint32_t serial,
		struct wl_surface *surface) {
	(void)data; (void)kbd; (void)serial; (void)surface;
}

static void kbd_key(void *data, struct wl_keyboard *kbd, uint32_t serial,
		uint32_t time, uint32_t key, uint32_t state) {
	(void)data; (void)kbd; (void)serial; (void)time;

	if (state != WL_KEYBOARD_KEY_STATE_PRESSED || !term.xkb_state)
		return;

	uint32_t keycode = key + 8;
	xkb_keysym_t sym = xkb_state_key_get_one_sym(term.xkb_state, keycode);

	/* Check for Ctrl modifier */
	bool ctrl = xkb_state_mod_name_is_active(term.xkb_state,
		XKB_MOD_NAME_CTRL, XKB_STATE_MODS_EFFECTIVE);

	/* Pass to TSM VTE */
	uint32_t mods = 0;
	if (ctrl) mods |= TSM_CONTROL_MASK;
	if (xkb_state_mod_name_is_active(term.xkb_state,
			XKB_MOD_NAME_SHIFT, XKB_STATE_MODS_EFFECTIVE))
		mods |= TSM_SHIFT_MASK;
	if (xkb_state_mod_name_is_active(term.xkb_state,
			XKB_MOD_NAME_ALT, XKB_STATE_MODS_EFFECTIVE))
		mods |= TSM_ALT_MASK;

	/* Map xkb keysym to TSM — tsm_vte_handle_keyboard does the work */
	uint32_t ucs4 = xkb_state_key_get_utf32(term.xkb_state, keycode);

	if (tsm_vte_handle_keyboard(term.vte, sym, 0, mods, ucs4)) {
		/* VTE consumed the key → render */
		tsm_screen_sb_reset(term.screen);
		term.needs_redraw = true;
	}
}

static void kbd_modifiers(void *data, struct wl_keyboard *kbd, uint32_t serial,
		uint32_t mods_depressed, uint32_t mods_latched,
		uint32_t mods_locked, uint32_t group) {
	(void)data; (void)kbd; (void)serial;
	if (term.xkb_state) {
		xkb_state_update_mask(term.xkb_state, mods_depressed,
			mods_latched, mods_locked, 0, 0, group);
	}
}

static void kbd_repeat_info(void *data, struct wl_keyboard *kbd,
		int32_t rate, int32_t delay) {
	(void)data; (void)kbd; (void)rate; (void)delay;
	/* TODO: implement key repeat timer */
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

static void seat_capabilities(void *data, struct wl_seat *seat,
		uint32_t caps) {
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
	.capabilities = seat_capabilities,
	.name = seat_name,
};

/* ── Registry ── */

static void registry_global(void *data, struct wl_registry *reg,
		uint32_t name, const char *interface, uint32_t version) {
	(void)data; (void)version;
	if (strcmp(interface, wl_compositor_interface.name) == 0) {
		wl_compositor = wl_registry_bind(reg, name,
			&wl_compositor_interface, 4);
	} else if (strcmp(interface, wl_shm_interface.name) == 0) {
		wl_shm = wl_registry_bind(reg, name, &wl_shm_interface, 1);
	} else if (strcmp(interface, xdg_wm_base_interface.name) == 0) {
		xdg_wm_base = wl_registry_bind(reg, name,
			&xdg_wm_base_interface, 2);
		xdg_wm_base_add_listener(xdg_wm_base, &xdg_wm_base_listener, NULL);
	} else if (strcmp(interface, wl_seat_interface.name) == 0) {
		wl_seat = wl_registry_bind(reg, name, &wl_seat_interface, 5);
		wl_seat_add_listener(wl_seat, &seat_listener, NULL);
	}
}

static void registry_global_remove(void *data, struct wl_registry *reg,
		uint32_t name) {
	(void)data; (void)reg; (void)name;
}

static const struct wl_registry_listener registry_listener = {
	.global = registry_global,
	.global_remove = registry_global_remove,
};

/* ── PTY setup ── */

static bool setup_pty(void) {
	int master, slave;
	if (openpty(&master, &slave, NULL, NULL, NULL) < 0) {
		perror("openpty");
		return false;
	}

	/* Set non-blocking on master */
	int flags = fcntl(master, F_GETFL);
	fcntl(master, F_SETFL, flags | O_NONBLOCK);

	pid_t pid = fork();
	if (pid < 0) {
		perror("fork");
		close(master);
		close(slave);
		return false;
	}

	if (pid == 0) {
		/* Child — become session leader, set controlling terminal */
		close(master);
		setsid();
		ioctl(slave, TIOCSCTTY, 0);

		dup2(slave, STDIN_FILENO);
		dup2(slave, STDOUT_FILENO);
		dup2(slave, STDERR_FILENO);
		if (slave > 2) close(slave);

		/* Set initial window size */
		struct winsize ws = {
			.ws_row = term.rows,
			.ws_col = term.cols,
		};
		ioctl(STDIN_FILENO, TIOCSWINSZ, &ws);

		/* Set TERM */
		setenv("TERM", "xterm-256color", 1);
		setenv("COLORTERM", "truecolor", 1);

		/* Exec user's shell */
		const char *shell = getenv("SHELL");
		if (!shell) shell = "/bin/sh";
		execl(shell, shell, "-l", NULL);
		_exit(127);
	}

	/* Parent */
	close(slave);
	term.pty_master = master;
	term.child_pid = pid;
	return true;
}

/* ── Signal handling ── */

static volatile sig_atomic_t child_exited = 0;

static void sigchld_handler(int sig) {
	(void)sig;
	child_exited = 1;
}

/* ── Main ── */

int main(int argc, char *argv[]) {
	(void)argc; (void)argv;
	setlocale(LC_ALL, "");

	/* Set up SIGCHLD handler */
	struct sigaction sa = { .sa_handler = sigchld_handler };
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = SA_NOCLDSTOP;
	sigaction(SIGCHLD, &sa, NULL);

	/* Font setup */
	term.font_desc = pango_font_description_from_string(TERM_FONT);
	term.scale = 1;

	/* Measure cell size */
	measure_cell();

	/* Initial geometry */
	term.cols = TERM_INITIAL_COLS;
	term.rows = TERM_INITIAL_ROWS;
	term.width = term.cols * term.cell_w + 2 * TERM_PAD_X;
	term.height = term.rows * term.cell_h + 2 * TERM_PAD_Y;
	term.cursor_visible = true;
	term.last_blink_ms = now_ms();

	/* TSM screen + VTE */
	if (tsm_screen_new(&term.screen, tsm_log_cb, NULL) < 0) {
		fprintf(stderr, "Failed to create TSM screen\n");
		return 1;
	}
	tsm_screen_resize(term.screen, term.cols, term.rows);
	tsm_screen_set_max_sb(term.screen, TERM_SCROLLBACK);

	if (tsm_vte_new(&term.vte, term.screen, tsm_write_cb, NULL,
			tsm_log_cb, NULL) < 0) {
		fprintf(stderr, "Failed to create TSM VTE\n");
		return 1;
	}

	/* XKB context */
	term.xkb_ctx = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
	if (!term.xkb_ctx) {
		fprintf(stderr, "Failed to create XKB context\n");
		return 1;
	}

	/* Wayland connection */
	wl_display = wl_display_connect(NULL);
	if (!wl_display) {
		fprintf(stderr, "Failed to connect to Wayland display\n");
		return 1;
	}

	wl_registry = wl_display_get_registry(wl_display);
	wl_registry_add_listener(wl_registry, &registry_listener, NULL);
	wl_display_roundtrip(wl_display);

	if (!wl_compositor || !wl_shm || !xdg_wm_base) {
		fprintf(stderr, "Missing required Wayland globals\n");
		return 1;
	}

	/* Create surface and xdg toplevel */
	term.surface = wl_compositor_create_surface(wl_compositor);
	term.xdg_surface = xdg_wm_base_get_xdg_surface(xdg_wm_base, term.surface);
	xdg_surface_add_listener(term.xdg_surface, &xdg_surface_listener, NULL);

	term.xdg_toplevel = xdg_surface_get_toplevel(term.xdg_surface);
	xdg_toplevel_add_listener(term.xdg_toplevel, &xdg_toplevel_listener, NULL);
	xdg_toplevel_set_title(term.xdg_toplevel, "Leaves Terminal");
	xdg_toplevel_set_app_id(term.xdg_toplevel, "leaves-terminal");
	wl_surface_commit(term.surface);

	/* Wait for configure */
	while (!term.configured && !term.closed)
		wl_display_dispatch(wl_display);

	if (term.closed) goto cleanup;

	/* Create initial buffer */
	if (!create_buffer(term.width, term.height)) {
		fprintf(stderr, "Failed to create SHM buffer\n");
		goto cleanup;
	}

	/* Set up PTY */
	if (!setup_pty()) goto cleanup;

	/* Initial render */
	render();
	request_frame();

	/* Main event loop */
	int wl_fd = wl_display_get_fd(wl_display);
	struct pollfd fds[2] = {
		{ .fd = wl_fd, .events = POLLIN },
		{ .fd = term.pty_master, .events = POLLIN },
	};

	while (!term.closed && !child_exited) {
		/* Flush pending Wayland requests */
		while (wl_display_prepare_read(wl_display) != 0)
			wl_display_dispatch_pending(wl_display);
		wl_display_flush(wl_display);

		int ret = poll(fds, 2, CURSOR_BLINK_MS);
		if (ret < 0) {
			if (errno == EINTR) {
				wl_display_cancel_read(wl_display);
				continue;
			}
			wl_display_cancel_read(wl_display);
			break;
		}

		if (fds[0].revents & POLLIN) {
			wl_display_read_events(wl_display);
			wl_display_dispatch_pending(wl_display);
		} else {
			wl_display_cancel_read(wl_display);
		}

		/* Read from PTY */
		if (fds[1].revents & POLLIN) {
			char buf[PTY_BUF_SIZE];
			ssize_t n = read(term.pty_master, buf, sizeof(buf));
			if (n > 0) {
				tsm_vte_input(term.vte, buf, n);
				term.needs_redraw = true;
			} else if (n == 0 || (n < 0 && errno != EAGAIN && errno != EINTR)) {
				break;  /* PTY closed */
			}
		}

		/* Check for PTY hangup */
		if (fds[1].revents & (POLLHUP | POLLERR)) {
			break;
		}

		/* Redraw if needed (immediate, don't wait for frame callback) */
		if (term.needs_redraw && !term.frame_cb) {
			render();
			term.needs_redraw = false;
			request_frame();
		}
	}

cleanup:
	/* Clean up child */
	if (term.child_pid > 0) {
		kill(term.child_pid, SIGHUP);
		waitpid(term.child_pid, NULL, WNOHANG);
	}
	if (term.pty_master >= 0) close(term.pty_master);

	/* Clean up TSM */
	if (term.vte) tsm_vte_unref(term.vte);
	if (term.screen) tsm_screen_unref(term.screen);

	/* Clean up XKB */
	if (term.xkb_state) xkb_state_unref(term.xkb_state);
	if (term.xkb_keymap) xkb_keymap_unref(term.xkb_keymap);
	if (term.xkb_ctx) xkb_context_unref(term.xkb_ctx);

	/* Clean up Cairo */
	if (term.cr) cairo_destroy(term.cr);
	if (term.cairo_surface) cairo_surface_destroy(term.cairo_surface);
	if (term.font_desc) pango_font_description_free(term.font_desc);

	/* Clean up Wayland */
	if (term.buffer) wl_buffer_destroy(term.buffer);
	if (term.shm_data) munmap(term.shm_data, term.shm_size);
	if (term.xdg_toplevel) xdg_toplevel_destroy(term.xdg_toplevel);
	if (term.xdg_surface) xdg_surface_destroy(term.xdg_surface);
	if (term.surface) wl_surface_destroy(term.surface);
	if (wl_keyboard) wl_keyboard_destroy(wl_keyboard);
	if (wl_seat) wl_seat_destroy(wl_seat);
	if (xdg_wm_base) xdg_wm_base_destroy(xdg_wm_base);
	if (wl_shm) wl_shm_destroy(wl_shm);
	if (wl_compositor) wl_compositor_destroy(wl_compositor);
	if (wl_registry) wl_registry_destroy(wl_registry);
	if (wl_display) wl_display_disconnect(wl_display);

	return 0;
}
