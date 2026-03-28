#ifndef LEAVES_RENDERER_H
#define LEAVES_RENDERER_H

#include <cairo.h>
#include <pango/pangocairo.h>
#include <stdbool.h>

struct wlr_renderer;
struct leaves_feed;
struct leaves_input;
struct leaves_status;

/* ── Card text selection ── */

/* Hit-test info populated during render for each visible card.
 * Tracks title and result regions separately (different fonts). */
struct card_text_hit {
	int card_idx;
	int text_x;             /* left edge (shared by both regions) */
	int text_w;             /* available width */

	/* Title region */
	int title_y, title_h;
	char title[512];
	int title_len;

	/* Result region (may be empty) */
	int result_y, result_h;
	char result[3584];
	int result_len;

	/* Combined text for copy: title + \n + result */
	char text[4096];
	int text_len;
};

struct card_text_sel {
	int card_idx;           /* which card is selected, or -1 */
	int anchor;             /* byte offset of selection start */
	int focus;              /* byte offset of selection end */
};

struct leaves_renderer {
	int width;
	int height;
	cairo_surface_t *surface;
	cairo_t *cr;

	/* Pango font descriptions — Geist family */
	PangoFontDescription *font_intent_title;
	PangoFontDescription *font_body;
	PangoFontDescription *font_action_chain;
	PangoFontDescription *font_capability;
	PangoFontDescription *font_timing;
	PangoFontDescription *font_input;
	PangoFontDescription *font_keyboard_hint;
	PangoFontDescription *font_empty_heading;
	PangoFontDescription *font_empty_sub;
	PangoFontDescription *font_status;

	/* Overlay panel fonts */
	PangoFontDescription *font_overlay_title;
	PangoFontDescription *font_overlay_body;
	PangoFontDescription *font_overlay_mono;
	PangoFontDescription *font_overlay_button;

	bool inference_online;

	/* Status bar data (set by compositor before rendering) */
	struct leaves_status *status;

	/* Wallpaper (JPEG decoded to Cairo surface) */
	cairo_surface_t *wallpaper;

	/* Hit-test geometry (populated each frame) */
	bool overlay_buttons_valid;
	int  overlay_cancel_x,  overlay_cancel_y;
	int  overlay_cancel_w,  overlay_cancel_h;
	int  overlay_confirm_x, overlay_confirm_y;
	int  overlay_confirm_w, overlay_confirm_h;

	int  dropdown_x, dropdown_y, dropdown_w, dropdown_h;

	int  history_icon_x, history_icon_y;    /* top-left of icon zone */
	int  history_icon_w, history_icon_h;

	int  input_field_right_x;   /* right boundary of the input field */
	int  status_zone_left_x;    /* left edge of status indicator zone */

	/* Card text selection */
	struct card_text_hit card_hits[200]; /* matches MAX_INTENTS */
	int card_hit_count;
	struct card_text_sel card_sel;
};

struct leaves_renderer *renderer_create(void);
void renderer_destroy(struct leaves_renderer *r);
void renderer_resize(struct leaves_renderer *r, int width, int height);

/* Load a JPEG file as the desktop wallpaper (cover mode). */
void renderer_load_wallpaper(struct leaves_renderer *r, const char *path);

/* Render full frame to cairo surface. Returns pixel data + stride. */
unsigned char *renderer_draw_frame(struct leaves_renderer *r,
	struct leaves_feed *feed, struct leaves_input *input, int *stride);

/* Map a panel-relative x coordinate to a byte offset in input->buf. */
int renderer_input_hit_test(struct leaves_renderer *r,
	struct leaves_input *input, double panel_x);

/* Return the card index at the given panel y-coordinate, or -1. */
int renderer_card_hit_test(struct leaves_renderer *r,
	struct leaves_feed *feed, double panel_y);

/* Build a copyable text string from a card.  Returns bytes written (excl NUL). */
int renderer_card_copy_text(struct leaves_feed *feed, int card_idx,
	char *buf, int buf_size);

/* Hit-test card text at (panel_x, panel_y).
 * Returns card index and sets *byte_offset, or returns -1 if miss. */
int renderer_card_text_at(struct leaves_renderer *r,
	double panel_x, double panel_y, int *byte_offset);

/* Get the selected text range.  Returns length or 0. */
int renderer_card_sel_text(struct leaves_renderer *r,
	char *buf, int buf_size);

#endif
