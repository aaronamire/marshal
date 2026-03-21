#ifndef LEAVES_RENDERER_H
#define LEAVES_RENDERER_H

#include <cairo.h>
#include <pango/pangocairo.h>
#include <stdbool.h>

struct wlr_renderer;
struct leaves_feed;
struct leaves_input;

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
};

struct leaves_renderer *renderer_create(void);
void renderer_destroy(struct leaves_renderer *r);
void renderer_resize(struct leaves_renderer *r, int width, int height);

/* Render full frame to cairo surface. Returns pixel data + stride. */
unsigned char *renderer_draw_frame(struct leaves_renderer *r,
	struct leaves_feed *feed, struct leaves_input *input, int *stride);

#endif
