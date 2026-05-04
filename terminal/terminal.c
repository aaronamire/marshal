/*
 * marshal-terminal — Minimal Wayland-native terminal emulator for Marshal.
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
#include <limits.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <sys/uio.h>
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

/* ── Colors (Marshal palette — dark terminal) ── */

struct rgba { double r, g, b, a; };

/* 16-color ANSI palette + Marshal accents */
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

/* ── Scrollback capture for proactive intents ──
 *
 * Captures the last SCROLLBACK_CAP bytes of raw PTY output in a ring buffer.
 * On non-zero child exit, the buffer is stripped of ANSI escapes and written
 * to ~/.marshal/terminal-scrollback.txt. The agentd compositor event watcher
 * reads this file to synthesize diagnostic GoalSpecs.
 */

#define SCROLLBACK_FILE_CAP  (32 * 1024)  /* 32KB ring buffer (~500 lines) */

static char scrollback_ring[SCROLLBACK_FILE_CAP];
static size_t scrollback_wpos = 0;     /* next write position in ring */
static size_t scrollback_total = 0;    /* total bytes ever written */

static void scrollback_write(const char *data, size_t len) {
	size_t orig = len;
	while (len > 0) {
		size_t space = SCROLLBACK_FILE_CAP - scrollback_wpos;
		size_t chunk = len < space ? len : space;
		memcpy(scrollback_ring + scrollback_wpos, data, chunk);
		scrollback_wpos = (scrollback_wpos + chunk) % SCROLLBACK_FILE_CAP;
		data += chunk;
		len -= chunk;
	}
	scrollback_total += orig;
}

/* Strip ANSI escape sequences and control chars for clean LLM-readable text.
 * Handles CSI sequences (ESC [ ... final_byte) and short escapes (ESC char). */
static size_t strip_ansi(const char *src, size_t src_len,
		char *dst, size_t dst_cap) {
	size_t di = 0;
	bool in_esc = false;
	bool in_csi = false;
	for (size_t si = 0; si < src_len && di < dst_cap - 1; si++) {
		unsigned char c = (unsigned char)src[si];
		if (in_esc) {
			if (c == '[') { in_csi = true; in_esc = false; continue; }
			in_esc = false;
			continue;
		}
		if (in_csi) {
			if (c >= 0x40 && c <= 0x7E) in_csi = false;
			continue;
		}
		if (c == 0x1B) { in_esc = true; continue; }
		/* Keep printable, newline, tab, CR, and UTF-8 continuation bytes */
		if (c >= 0x20 || c == '\n' || c == '\t' || c == '\r' || c >= 0x80)
			dst[di++] = (char)c;
	}
	dst[di] = '\0';
	return di;
}

static void dump_scrollback(int exit_code) {
	if (exit_code == 0) return;

	const char *home = getenv("HOME");
	if (!home) return;

	char dir_path[PATH_MAX];
	snprintf(dir_path, sizeof(dir_path), "%s/.marshal", home);
	mkdir(dir_path, 0700);  /* ignore EEXIST */

	/* Linearize ring buffer (oldest data first) */
	size_t used = scrollback_total < SCROLLBACK_FILE_CAP
		? scrollback_total : SCROLLBACK_FILE_CAP;
	if (used == 0) return;

	char *linear = malloc(used + 1);
	if (!linear) return;

	if (scrollback_total <= SCROLLBACK_FILE_CAP) {
		/* Buffer never wrapped */
		memcpy(linear, scrollback_ring, used);
	} else {
		/* Wrapped — oldest data starts at scrollback_wpos */
		size_t tail = SCROLLBACK_FILE_CAP - scrollback_wpos;
		memcpy(linear, scrollback_ring + scrollback_wpos, tail);
		memcpy(linear + tail, scrollback_ring, scrollback_wpos);
	}
	linear[used] = '\0';

	/* Strip ANSI escapes for clean text */
	char *clean = malloc(used + 1);
	if (!clean) { free(linear); return; }
	size_t clean_len = strip_ansi(linear, used, clean, used + 1);
	free(linear);

	char file_path[PATH_MAX];
	snprintf(file_path, sizeof(file_path),
		"%s/.marshal/terminal-scrollback.txt", home);

	int fd = open(file_path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
	if (fd >= 0) {
		/* Write may be partial on exotic FS — best-effort is fine */
		(void)write(fd, clean, clean_len);
		close(fd);
	}
	free(clean);
}

/* ── Wayland globals ── */

static struct wl_display    *wl_display;
static struct wl_registry   *wl_registry;
static struct wl_compositor *wl_compositor;
static struct wl_shm        *wl_shm;
static struct wl_seat       *wl_seat;
static struct wl_keyboard   *wl_keyboard;
static struct wl_pointer    *wl_pointer;
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

	/* Key repeat (driven by a timerfd polled in the main loop). The
	 * compositor advertises rate/delay via wl_keyboard.repeat_info; we
	 * fall back to sane defaults if no event has arrived yet. */
	int      repeat_timer_fd;
	int32_t  repeat_rate;     /* keys per second; 0 disables repeat */
	int32_t  repeat_delay;    /* ms before first repeat */
	xkb_keysym_t repeat_sym;
	uint32_t     repeat_mods;
	uint32_t     repeat_ucs4;
	bool         repeat_is_scroll;  /* repeat drives sb_up/sb_down instead of VTE */

	/* Mouse selection */
	double  pointer_x;
	double  pointer_y;
	bool    selecting;        /* left button held, dragging a selection */
	bool    has_selection;    /* selection committed (button released) */

	/* Bracketed paste mode (DECSET 2004). Toggled by the application via
	 * ESC[?2004h / ESC[?2004l on the PTY's output stream; we sniff for
	 * those sequences in the read loop. When enabled, paste handlers wrap
	 * clipboard contents in ESC[200~ … ESC[201~ so the shell can tell a
	 * paste apart from typed input and won't submit a multi-line paste
	 * one LF at a time. dec_match holds the in-progress prefix length so
	 * sequences split across read() boundaries still match. */
	bool     bracketed_paste;
	uint8_t  dec_match;
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
	char name[] = "/marshal-term-XXXXXX";
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
	/* Block cursor with Marshal accent blue */
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

/* ── Clipboard (Ctrl+Shift+C / Ctrl+Shift+V) ──
 *
 * The terminal has no mouse-driven selection yet, so Ctrl+Shift+C copies the
 * full visible screen to the Wayland clipboard. Ctrl+Shift+V reads the
 * clipboard and writes it straight to the PTY. We shell out to wl-copy /
 * wl-paste rather than implement wl_data_device_manager from scratch.
 */

struct copy_buf {
	char *data;
	size_t cap;
	size_t len;
	int last_y;   /* -1 = not started */
};

static void copy_buf_putc(struct copy_buf *cb, unsigned char c) {
	if (cb->len + 1 < cb->cap) cb->data[cb->len++] = (char)c;
}

static int copy_cell_cb(struct tsm_screen *con, uint64_t id,
		const uint32_t *ch, size_t len, unsigned int cwidth,
		unsigned int posx, unsigned int posy,
		const struct tsm_screen_attr *attr, tsm_age_t age,
		void *data) {
	(void)con; (void)id; (void)cwidth; (void)posx; (void)attr; (void)age;
	struct copy_buf *cb = data;

	if (cb->last_y >= 0 && (int)posy != cb->last_y) {
		/* rtrim trailing spaces of the previous row */
		while (cb->len > 0 && cb->data[cb->len - 1] == ' ') cb->len--;
		copy_buf_putc(cb, '\n');
	}
	cb->last_y = (int)posy;

	if (len == 0 || ch[0] == 0) {
		copy_buf_putc(cb, ' ');
		return 0;
	}
	for (size_t i = 0; i < len && ch[i] != 0; i++) {
		uint32_t c = ch[i];
		if (c < 0x80) {
			copy_buf_putc(cb, c);
		} else if (c < 0x800) {
			copy_buf_putc(cb, 0xC0 | (c >> 6));
			copy_buf_putc(cb, 0x80 | (c & 0x3F));
		} else if (c < 0x10000) {
			copy_buf_putc(cb, 0xE0 | (c >> 12));
			copy_buf_putc(cb, 0x80 | ((c >> 6) & 0x3F));
			copy_buf_putc(cb, 0x80 | (c & 0x3F));
		} else {
			copy_buf_putc(cb, 0xF0 | (c >> 18));
			copy_buf_putc(cb, 0x80 | ((c >> 12) & 0x3F));
			copy_buf_putc(cb, 0x80 | ((c >> 6) & 0x3F));
			copy_buf_putc(cb, 0x80 | (c & 0x3F));
		}
	}
	return 0;
}

/* Spawn wl-copy and pipe `data` to its stdin. If `primary` is true, copy to
 * the X11-style PRIMARY selection so middle-click paste works. */
static void wl_copy_send(const char *data, size_t len, bool primary) {
	if (!data || len == 0) return;

	int pipefd[2];
	if (pipe(pipefd) < 0) return;

	pid_t pid = fork();
	if (pid < 0) {
		close(pipefd[0]); close(pipefd[1]);
		return;
	}
	if (pid == 0) {
		close(pipefd[1]);
		dup2(pipefd[0], STDIN_FILENO);
		close(pipefd[0]);
		int devnull = open("/dev/null", O_WRONLY);
		if (devnull >= 0) {
			dup2(devnull, STDERR_FILENO);
			close(devnull);
		}
		if (primary)
			execlp("wl-copy", "wl-copy", "--primary", (char *)NULL);
		else
			execlp("wl-copy", "wl-copy", (char *)NULL);
		_exit(127);
	}

	close(pipefd[0]);
	const char *p = data;
	size_t left = len;
	while (left > 0) {
		ssize_t n = write(pipefd[1], p, left);
		if (n < 0) {
			if (errno == EINTR) continue;
			break;
		}
		p += n;
		left -= n;
	}
	close(pipefd[1]);
	/* wl-copy daemonizes; sigchld_handler reaps the foreground exit. */
}

static void copy_screen_to_clipboard(void) {
	/* If the user dragged out a selection, copy that — otherwise fall back
	 * to the entire visible screen. */
	char *sel = NULL;
	int sel_len = tsm_screen_selection_copy(term.screen, &sel);
	if (sel_len > 0 && sel) {
		wl_copy_send(sel, (size_t)sel_len, false);
		free(sel);
		return;
	}
	free(sel);

	size_t cap = (size_t)(term.cols + 1) * (size_t)term.rows * 4 + 16;
	struct copy_buf cb = {
		.data = malloc(cap), .cap = cap, .len = 0, .last_y = -1
	};
	if (!cb.data) return;

	tsm_screen_draw(term.screen, copy_cell_cb, &cb);
	/* rtrim final row, then drop trailing blank lines */
	while (cb.len > 0 && cb.data[cb.len - 1] == ' ') cb.len--;
	while (cb.len > 0 && cb.data[cb.len - 1] == '\n') cb.len--;

	wl_copy_send(cb.data, cb.len, false);
	free(cb.data);
}

/* Sniff PTY output for DECSET/DECRST 2004 (bracketed paste mode) and update
 * term.bracketed_paste. Sequences may be split across read() boundaries, so
 * dec_match holds the prefix length matched so far. We don't strip the bytes —
 * tsm_vte handles them downstream (it just doesn't expose the mode). */
static void sniff_dec_modes(const char *buf, size_t n) {
	/* The enable ('h') and disable ('l') sequences share the same 7-byte
	 * prefix; we only branch on the final byte. */
	static const char prefix[] = "\x1b[?2004";
	const size_t plen = sizeof(prefix) - 1 + 1;  /* +1 for the final byte */

	for (size_t i = 0; i < n; i++) {
		uint8_t m = term.dec_match;
		char c = buf[i];
		if (m < plen - 1) {
			/* Matching ESC [ ? 2 0 0 4 */
			if (c == prefix[m]) {
				term.dec_match = m + 1;
			} else {
				/* Restart: if we just saw ESC, start a fresh match */
				term.dec_match = (c == 0x1b) ? 1 : 0;
			}
		} else {
			/* At the final byte — 'h' enables, 'l' disables, anything else
			 * resets (and may itself start a new ESC match). */
			if (c == 'h') {
				term.bracketed_paste = true;
				term.dec_match = 0;
			} else if (c == 'l') {
				term.bracketed_paste = false;
				term.dec_match = 0;
			} else {
				term.dec_match = (c == 0x1b) ? 1 : 0;
			}
		}
	}
}

/* Send pasted clipboard bytes to the PTY, wrapping in bracketed-paste markers
 * when the application has enabled DECSET 2004. We also strip any embedded
 * ESC[201~ from the payload so a malicious clipboard can't close the bracket
 * early and inject post-paste commands. */
static void write_paste_to_pty(char *data, size_t len) {
	if (!data || len == 0) return;

	/* Defang any embedded "end-paste" marker. Replace the ESC byte with '?'
	 * so the rest of the sequence becomes harmless literal text. */
	if (term.bracketed_paste && len >= 6) {
		for (size_t i = 0; i + 5 < len; i++) {
			if (data[i] == 0x1b && data[i+1] == '[' &&
			    data[i+2] == '2' && data[i+3] == '0' &&
			    data[i+4] == '1' && data[i+5] == '~') {
				data[i] = '?';
			}
		}
	}

	static const char start[] = "\x1b[200~";
	static const char end[]   = "\x1b[201~";

	struct iovec parts[3];
	int nparts = 0;
	if (term.bracketed_paste) {
		parts[nparts].iov_base = (void *)start;
		parts[nparts].iov_len  = sizeof(start) - 1;
		nparts++;
	}
	parts[nparts].iov_base = data;
	parts[nparts].iov_len  = len;
	nparts++;
	if (term.bracketed_paste) {
		parts[nparts].iov_base = (void *)end;
		parts[nparts].iov_len  = sizeof(end) - 1;
		nparts++;
	}

	for (int i = 0; i < nparts; i++) {
		const char *p = parts[i].iov_base;
		size_t left = parts[i].iov_len;
		while (left > 0) {
			ssize_t w = write(term.pty_master, p, left);
			if (w < 0) {
				if (errno == EINTR) continue;
				if (errno == EAGAIN) {
					struct timespec ts = { 0, 1 * 1000 * 1000 };
					nanosleep(&ts, NULL);
					continue;
				}
				return;
			}
			p += w;
			left -= (size_t)w;
		}
	}
}

static void paste_from_clipboard(void) {
	int pipefd[2];
	if (pipe(pipefd) < 0) return;

	pid_t pid = fork();
	if (pid < 0) {
		close(pipefd[0]); close(pipefd[1]);
		return;
	}
	if (pid == 0) {
		close(pipefd[0]);
		dup2(pipefd[1], STDOUT_FILENO);
		close(pipefd[1]);
		int devnull = open("/dev/null", O_WRONLY);
		if (devnull >= 0) {
			dup2(devnull, STDERR_FILENO);
			close(devnull);
		}
		/* --no-newline: don't append a newline that wasn't on the clipboard */
		execlp("wl-paste", "wl-paste", "--no-newline", (char *)NULL);
		_exit(127);
	}

	close(pipefd[1]);
	char rbuf[4096];
	char *full = NULL;
	size_t total = 0, cap = 0;
	for (;;) {
		ssize_t n = read(pipefd[0], rbuf, sizeof(rbuf));
		if (n < 0) {
			if (errno == EINTR) continue;
			break;
		}
		if (n == 0) break;
		if (total + (size_t)n > cap) {
			size_t new_cap = cap == 0 ? 8192 : cap * 2;
			while (new_cap < total + (size_t)n) new_cap *= 2;
			char *new_full = realloc(full, new_cap);
			if (!new_full) { free(full); close(pipefd[0]); return; }
			full = new_full;
			cap = new_cap;
		}
		memcpy(full + total, rbuf, (size_t)n);
		total += (size_t)n;
	}
	close(pipefd[0]);
	if (!full || total == 0) { free(full); return; }

	/* Normalize CRLF / lone CR to LF — typical for clipboard contents
	 * pasted from web pages or other apps. */
	size_t out = 0;
	for (size_t i = 0; i < total; i++) {
		char c = full[i];
		if (c == '\r') {
			full[out++] = '\n';
			if (i + 1 < total && full[i + 1] == '\n') i++;
		} else {
			full[out++] = c;
		}
	}
	total = out;

	write_paste_to_pty(full, total);
	free(full);
}

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
	fprintf(stderr, "[term] kbd_keymap: keymap=%p state=%p\n",
		(void *)term.xkb_keymap, (void *)term.xkb_state);
}

static void kbd_enter(void *data, struct wl_keyboard *kbd, uint32_t serial,
		struct wl_surface *surface, struct wl_array *keys) {
	(void)data; (void)kbd; (void)serial; (void)surface; (void)keys;
	term.cursor_visible = true;
	fprintf(stderr, "[term] kbd_enter: xkb_state=%p\n", (void *)term.xkb_state);
}

static void kbd_leave(void *data, struct wl_keyboard *kbd, uint32_t serial,
		struct wl_surface *surface) {
	(void)data; (void)kbd; (void)serial; (void)surface;
}

/* Send a single keysym/mods pair through the VTE. Returns true if the VTE
 * actually emitted bytes (so the caller can decide whether the key is
 * worth repeating). */
static bool dispatch_sym(xkb_keysym_t sym, uint32_t mods, uint32_t ucs4) {
	if (tsm_vte_handle_keyboard(term.vte, sym, 0, mods, ucs4)) {
		tsm_screen_sb_reset(term.screen);
		term.needs_redraw = true;
		return true;
	}
	return false;
}

static void disarm_key_repeat(void) {
	if (term.repeat_timer_fd < 0) return;
	struct itimerspec its = {{0,0},{0,0}};
	timerfd_settime(term.repeat_timer_fd, 0, &its, NULL);
	term.repeat_sym = XKB_KEY_NoSymbol;
	term.repeat_is_scroll = false;
}

static void arm_key_repeat_initial(xkb_keysym_t sym, uint32_t mods,
		uint32_t ucs4) {
	if (term.repeat_timer_fd < 0 || term.repeat_rate <= 0 ||
			term.repeat_delay <= 0) return;
	term.repeat_sym  = sym;
	term.repeat_mods = mods;
	term.repeat_ucs4 = ucs4;
	term.repeat_is_scroll = false;
	struct itimerspec its = {0};
	its.it_value.tv_sec  = term.repeat_delay / 1000;
	its.it_value.tv_nsec = (long)(term.repeat_delay % 1000) * 1000000L;
	/* Periodic interval for repeats after the initial delay */
	long period_ns = 1000000000L / term.repeat_rate;
	its.it_interval.tv_sec  = period_ns / 1000000000L;
	its.it_interval.tv_nsec = period_ns % 1000000000L;
	timerfd_settime(term.repeat_timer_fd, 0, &its, NULL);
}

static void key_repeat_fire(void) {
	uint64_t expirations;
	if (read(term.repeat_timer_fd, &expirations, sizeof(expirations)) < 0)
		return;
	if (term.repeat_sym == XKB_KEY_NoSymbol) return;

	/* Scrollback keys (Shift+PageUp / Shift+PageDown) repeat the local
	 * scrollback move rather than firing through the VTE. */
	if (term.repeat_is_scroll) {
		if (term.repeat_sym == XKB_KEY_Page_Up)
			tsm_screen_sb_up(term.screen, term.rows / 2);
		else if (term.repeat_sym == XKB_KEY_Page_Down)
			tsm_screen_sb_down(term.screen, term.rows / 2);
		term.needs_redraw = true;
		return;
	}

	dispatch_sym(term.repeat_sym, term.repeat_mods, term.repeat_ucs4);
}

static void kbd_key(void *data, struct wl_keyboard *kbd, uint32_t serial,
		uint32_t time, uint32_t key, uint32_t state) {
	(void)data; (void)kbd; (void)serial; (void)time;

	if (!term.xkb_state) {
		fprintf(stderr, "[term] kbd_key: NO xkb_state, dropping key=%u\n", key);
		return;
	}

	/* On release, only stop repeating if the released key is the one we
	 * are currently repeating. (A modifier release while another key is
	 * held should not break the held key's repeat.) */
	if (state == WL_KEYBOARD_KEY_STATE_RELEASED) {
		uint32_t keycode = key + 8;
		xkb_keysym_t rel_sym = xkb_state_key_get_one_sym(term.xkb_state, keycode);
		if (rel_sym == term.repeat_sym)
			disarm_key_repeat();
		return;
	}

	uint32_t keycode = key + 8;
	xkb_keysym_t sym = xkb_state_key_get_one_sym(term.xkb_state, keycode);
	fprintf(stderr, "[term] kbd_key: key=%u sym=0x%x\n", key, sym);

	/* Check for Ctrl modifier */
	bool ctrl = xkb_state_mod_name_is_active(term.xkb_state,
		XKB_MOD_NAME_CTRL, XKB_STATE_MODS_EFFECTIVE);
	bool shift = xkb_state_mod_name_is_active(term.xkb_state,
		XKB_MOD_NAME_SHIFT, XKB_STATE_MODS_EFFECTIVE);

	/* Marshal clipboard shortcuts — Ctrl+Shift+C / Ctrl+Shift+V. We have
	 * to intercept BEFORE tsm_vte_handle_keyboard, otherwise Ctrl+Shift+C
	 * gets sent to the PTY as ETX and kills the foreground job. */
	if (ctrl && shift) {
		if (sym == XKB_KEY_C || sym == XKB_KEY_c) {
			disarm_key_repeat();
			copy_screen_to_clipboard();
			return;
		}
		if (sym == XKB_KEY_V || sym == XKB_KEY_v) {
			disarm_key_repeat();
			paste_from_clipboard();
			return;
		}
	}

	/* Scrollback navigation (Shift+PageUp / Shift+PageDown / Shift+Home /
	 * Shift+End). Intercepted because we want them to move the local
	 * scrollback view, not be sent to the shell. The repeat timer makes
	 * holding PageUp keep scrolling. */
	if (shift && !ctrl) {
		bool scroll_handled = false;
		if (sym == XKB_KEY_Page_Up) {
			tsm_screen_sb_up(term.screen, term.rows / 2);
			scroll_handled = true;
		} else if (sym == XKB_KEY_Page_Down) {
			tsm_screen_sb_down(term.screen, term.rows / 2);
			scroll_handled = true;
		} else if (sym == XKB_KEY_Home) {
			tsm_screen_sb_up(term.screen, TERM_SCROLLBACK);
			scroll_handled = true;
		} else if (sym == XKB_KEY_End) {
			tsm_screen_sb_reset(term.screen);
			scroll_handled = true;
		}
		if (scroll_handled) {
			term.needs_redraw = true;
			/* Only PageUp/PageDown repeat — Home/End are one-shot */
			if (sym == XKB_KEY_Page_Up || sym == XKB_KEY_Page_Down) {
				arm_key_repeat_initial(sym, 0, 0);
				term.repeat_is_scroll = true;
			}
			return;
		}
	}

	/* A new keypress always cancels any pending repeat from a previous
	 * key, even if dispatch_sym returns false (modifier-only press). */
	disarm_key_repeat();

	uint32_t mods = 0;
	if (ctrl) mods |= TSM_CONTROL_MASK;
	if (shift) mods |= TSM_SHIFT_MASK;
	if (xkb_state_mod_name_is_active(term.xkb_state,
			XKB_MOD_NAME_ALT, XKB_STATE_MODS_EFFECTIVE))
		mods |= TSM_ALT_MASK;

	uint32_t ucs4 = xkb_state_key_get_utf32(term.xkb_state, keycode);

	if (dispatch_sym(sym, mods, ucs4))
		arm_key_repeat_initial(sym, mods, ucs4);
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
	(void)data; (void)kbd;
	term.repeat_rate = rate;
	term.repeat_delay = delay;
	fprintf(stderr, "[term] kbd_repeat_info: rate=%d delay=%d\n", rate, delay);
}

static const struct wl_keyboard_listener keyboard_listener = {
	.keymap = kbd_keymap,
	.enter = kbd_enter,
	.leave = kbd_leave,
	.key = kbd_key,
	.modifiers = kbd_modifiers,
	.repeat_info = kbd_repeat_info,
};

/* ── Pointer (mouse selection + middle-click paste) ── */

static void pointer_to_cell(int *cx, int *cy) {
	int x = (int)term.pointer_x - TERM_PAD_X;
	int y = (int)term.pointer_y - TERM_PAD_Y;
	if (term.cell_w <= 0 || term.cell_h <= 0) {
		*cx = 0; *cy = 0; return;
	}
	int col = x / term.cell_w;
	int row = y / term.cell_h;
	if (col < 0) col = 0;
	if (col >= term.cols) col = term.cols - 1;
	if (row < 0) row = 0;
	if (row >= term.rows) row = term.rows - 1;
	*cx = col;
	*cy = row;
}

static void paste_primary_to_pty(void) {
	int pipefd[2];
	if (pipe(pipefd) < 0) return;
	pid_t pid = fork();
	if (pid < 0) { close(pipefd[0]); close(pipefd[1]); return; }
	if (pid == 0) {
		close(pipefd[0]);
		dup2(pipefd[1], STDOUT_FILENO);
		close(pipefd[1]);
		int devnull = open("/dev/null", O_WRONLY);
		if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
		execlp("wl-paste", "wl-paste", "--primary", "--no-newline", (char *)NULL);
		_exit(127);
	}
	close(pipefd[1]);
	char rbuf[4096];
	char *full = NULL;
	size_t total = 0, cap = 0;
	for (;;) {
		ssize_t n = read(pipefd[0], rbuf, sizeof(rbuf));
		if (n < 0) { if (errno == EINTR) continue; break; }
		if (n == 0) break;
		if (total + (size_t)n > cap) {
			size_t nc = cap == 0 ? 8192 : cap * 2;
			while (nc < total + (size_t)n) nc *= 2;
			char *nf = realloc(full, nc);
			if (!nf) { free(full); close(pipefd[0]); return; }
			full = nf; cap = nc;
		}
		memcpy(full + total, rbuf, (size_t)n);
		total += (size_t)n;
	}
	close(pipefd[0]);
	if (!full || total == 0) { free(full); return; }

	/* Normalize CRLF / lone CR to LF (PRIMARY usually arrives clean, but
	 * cross-app pastes can include them just like CLIPBOARD). */
	size_t out = 0;
	for (size_t i = 0; i < total; i++) {
		char c = full[i];
		if (c == '\r') {
			full[out++] = '\n';
			if (i + 1 < total && full[i + 1] == '\n') i++;
		} else {
			full[out++] = c;
		}
	}
	total = out;

	write_paste_to_pty(full, total);
	free(full);
}

static void ptr_enter(void *data, struct wl_pointer *p, uint32_t serial,
		struct wl_surface *surf, wl_fixed_t sx, wl_fixed_t sy) {
	(void)data; (void)p; (void)serial; (void)surf;
	term.pointer_x = wl_fixed_to_double(sx);
	term.pointer_y = wl_fixed_to_double(sy);
}

static void ptr_leave(void *data, struct wl_pointer *p, uint32_t serial,
		struct wl_surface *surf) {
	(void)data; (void)p; (void)serial; (void)surf;
}

static void ptr_motion(void *data, struct wl_pointer *p, uint32_t time,
		wl_fixed_t sx, wl_fixed_t sy) {
	(void)data; (void)p; (void)time;
	term.pointer_x = wl_fixed_to_double(sx);
	term.pointer_y = wl_fixed_to_double(sy);
	if (term.selecting) {
		int cx, cy;
		pointer_to_cell(&cx, &cy);
		tsm_screen_selection_target(term.screen, cx, cy);
		term.needs_redraw = true;
	}
}

#define BTN_LEFT   0x110
#define BTN_MIDDLE 0x112

static void ptr_button(void *data, struct wl_pointer *p, uint32_t serial,
		uint32_t time, uint32_t button, uint32_t state) {
	(void)data; (void)p; (void)serial; (void)time;

	bool pressed = (state == WL_POINTER_BUTTON_STATE_PRESSED);

	if (button == BTN_LEFT) {
		if (pressed) {
			/* Starting a new selection — clear the old one first */
			if (term.has_selection) {
				tsm_screen_selection_reset(term.screen);
				term.has_selection = false;
				term.needs_redraw = true;
			}
			int cx, cy;
			pointer_to_cell(&cx, &cy);
			tsm_screen_selection_start(term.screen, cx, cy);
			term.selecting = true;
			term.needs_redraw = true;
		} else if (term.selecting) {
			term.selecting = false;
			term.has_selection = true;
			/* Push selection to PRIMARY so middle-click paste works
			 * across apps. CLIPBOARD waits for explicit Ctrl+Shift+C. */
			char *sel = NULL;
			int n = tsm_screen_selection_copy(term.screen, &sel);
			if (n > 0 && sel) {
				wl_copy_send(sel, (size_t)n, true);
			}
			free(sel);
		}
	} else if (button == BTN_MIDDLE && pressed) {
		paste_primary_to_pty();
	}
}

/* Scrollwheel-driven scrollback. wl_pointer.axis delivers a continuous
 * `value` in 1/256 px units; libinput's wheel detents come through as
 * multiples of ~10. We accumulate fractional ticks across events and emit
 * one scrollback line per WL_AXIS_DETENT, so a slow precision wheel still
 * scrolls smoothly and a fast flick scrolls many lines at once. */
#define WL_AXIS_DETENT 10.0
static double axis_accum_v = 0.0;

static void ptr_axis(void *d, struct wl_pointer *p, uint32_t t, uint32_t axis,
		wl_fixed_t v) {
	(void)d; (void)p; (void)t;
	if (axis != WL_POINTER_AXIS_VERTICAL_SCROLL) return;
	axis_accum_v += wl_fixed_to_double(v);
	while (axis_accum_v >= WL_AXIS_DETENT) {
		tsm_screen_sb_down(term.screen, 3);
		axis_accum_v -= WL_AXIS_DETENT;
		term.needs_redraw = true;
	}
	while (axis_accum_v <= -WL_AXIS_DETENT) {
		tsm_screen_sb_up(term.screen, 3);
		axis_accum_v += WL_AXIS_DETENT;
		term.needs_redraw = true;
	}
}

static void ptr_frame(void *d, struct wl_pointer *p) { (void)d;(void)p; }
static void ptr_axis_source(void *d, struct wl_pointer *p, uint32_t s) {
	(void)d;(void)p;(void)s; }
static void ptr_axis_stop(void *d, struct wl_pointer *p, uint32_t t,
		uint32_t axis) {
	(void)d; (void)p; (void)t;
	/* Reset accumulator at end of a scroll gesture so a stopped flick
	 * doesn't leak fractional ticks into the next one. */
	if (axis == WL_POINTER_AXIS_VERTICAL_SCROLL) axis_accum_v = 0.0;
}
static void ptr_axis_discrete(void *d, struct wl_pointer *p, uint32_t axis,
		int32_t v) {
	/* Discrete (notched) wheels also send wl_pointer.axis above, so we
	 * already counted the scroll there; nothing to do here. */
	(void)d;(void)p;(void)axis;(void)v;
}
static void ptr_axis_value120(void *d, struct wl_pointer *p, uint32_t a,
		int32_t v) { (void)d;(void)p;(void)a;(void)v; }
static void ptr_axis_relative_direction(void *d, struct wl_pointer *p,
		uint32_t a, uint32_t dir) { (void)d;(void)p;(void)a;(void)dir; }

static const struct wl_pointer_listener pointer_listener = {
	.enter = ptr_enter,
	.leave = ptr_leave,
	.motion = ptr_motion,
	.button = ptr_button,
	.axis = ptr_axis,
	.frame = ptr_frame,
	.axis_source = ptr_axis_source,
	.axis_stop = ptr_axis_stop,
	.axis_discrete = ptr_axis_discrete,
	.axis_value120 = ptr_axis_value120,
	.axis_relative_direction = ptr_axis_relative_direction,
};

/* ── Seat ── */

static void seat_capabilities(void *data, struct wl_seat *seat,
		uint32_t caps) {
	(void)data;
	fprintf(stderr, "[term] seat_capabilities: caps=0x%x keyboard=%d pointer=%d\n",
		caps, !!(caps & WL_SEAT_CAPABILITY_KEYBOARD),
		!!(caps & WL_SEAT_CAPABILITY_POINTER));
	if (caps & WL_SEAT_CAPABILITY_KEYBOARD) {
		if (wl_keyboard) wl_keyboard_destroy(wl_keyboard);
		wl_keyboard = wl_seat_get_keyboard(seat);
		wl_keyboard_add_listener(wl_keyboard, &keyboard_listener, NULL);
		fprintf(stderr, "[term] keyboard listener attached\n");
	}
	if (caps & WL_SEAT_CAPABILITY_POINTER) {
		if (wl_pointer) wl_pointer_destroy(wl_pointer);
		wl_pointer = wl_seat_get_pointer(seat);
		wl_pointer_add_listener(wl_pointer, &pointer_listener, NULL);
		fprintf(stderr, "[term] pointer listener attached\n");
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
static volatile sig_atomic_t child_exit_code = 0;

static void sigchld_handler(int sig) {
	(void)sig;
	int status;
	pid_t pid;
	/* Reap every dead child so wl-copy / wl-paste helpers don't zombify.
	 * Only the shell child (term.child_pid) should signal terminal exit. */
	while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
		if (pid == term.child_pid) {
			child_exited = 1;
			if (WIFEXITED(status))
				child_exit_code = WEXITSTATUS(status);
			else if (WIFSIGNALED(status))
				child_exit_code = 128 + WTERMSIG(status);
		}
	}
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

	/* Key repeat — sane fallbacks; compositor overrides via repeat_info */
	term.repeat_timer_fd = timerfd_create(CLOCK_MONOTONIC,
		TFD_NONBLOCK | TFD_CLOEXEC);
	term.repeat_rate  = 25;
	term.repeat_delay = 600;
	term.repeat_sym   = XKB_KEY_NoSymbol;

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
	wl_display_roundtrip(wl_display);  /* globals: wl_seat, wl_compositor, … */
	wl_display_roundtrip(wl_display);  /* initial events: seat capabilities, ��� */

	fprintf(stderr, "[term] after roundtrips: seat=%p keyboard=%p\n",
		(void *)wl_seat, (void *)wl_keyboard);

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
	xdg_toplevel_set_title(term.xdg_toplevel, "Marshal Terminal");
	xdg_toplevel_set_app_id(term.xdg_toplevel, "marshal-terminal");
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
	struct pollfd fds[3] = {
		{ .fd = wl_fd, .events = POLLIN },
		{ .fd = term.pty_master, .events = POLLIN },
		{ .fd = term.repeat_timer_fd, .events = POLLIN },
	};
	nfds_t nfds = (term.repeat_timer_fd >= 0) ? 3 : 2;

	while (!term.closed && !child_exited) {
		/* Flush pending Wayland requests */
		while (wl_display_prepare_read(wl_display) != 0)
			wl_display_dispatch_pending(wl_display);
		wl_display_flush(wl_display);

		int ret = poll(fds, nfds, CURSOR_BLINK_MS);
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
				sniff_dec_modes(buf, (size_t)n);
				scrollback_write(buf, n);
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

		/* Key repeat — fire each pending repeat tick */
		if (nfds >= 3 && (fds[2].revents & POLLIN)) {
			key_repeat_fire();
		}

		/* Redraw if needed (immediate, don't wait for frame callback) */
		if (term.needs_redraw && !term.frame_cb) {
			render();
			term.needs_redraw = false;
			request_frame();
		}
	}

cleanup:
	/* Dump scrollback for proactive intents before tearing down */
	dump_scrollback(child_exit_code);

	/* Clean up child */
	if (term.child_pid > 0) {
		if (!child_exited) {
			/* Child still running (user closed window) — signal and reap */
			kill(term.child_pid, SIGHUP);
			waitpid(term.child_pid, NULL, WNOHANG);
		}
		/* If child_exited, SIGCHLD handler already reaped */
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
	if (term.repeat_timer_fd >= 0) close(term.repeat_timer_fd);
	if (wl_pointer) wl_pointer_destroy(wl_pointer);
	if (wl_keyboard) wl_keyboard_destroy(wl_keyboard);
	if (wl_seat) wl_seat_destroy(wl_seat);
	if (xdg_wm_base) xdg_wm_base_destroy(xdg_wm_base);
	if (wl_shm) wl_shm_destroy(wl_shm);
	if (wl_compositor) wl_compositor_destroy(wl_compositor);
	if (wl_registry) wl_registry_destroy(wl_registry);
	if (wl_display) wl_display_disconnect(wl_display);

	/* Propagate child's exit code so the compositor sees non-zero exits
	 * and emits child_exited events that trigger proactive intents. */
	return child_exit_code;
}
