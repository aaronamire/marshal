#define _GNU_SOURCE
#include "renderer.h"
#include "colors.h"
#include "typography.h"
#include "geometry.h"
#include "feed.h"
#include "input.h"
#include "status.h"
#include <math.h>
#include <jpeglib.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* ── Rounded rectangle path ── */

static void rounded_rect(cairo_t *cr, double x, double y,
		double w, double h, double r) {
	cairo_new_sub_path(cr);
	cairo_arc(cr, x + w - r, y + r,     r, -M_PI / 2, 0);
	cairo_arc(cr, x + w - r, y + h - r, r,  0,         M_PI / 2);
	cairo_arc(cr, x + r,     y + h - r, r,  M_PI / 2,  M_PI);
	cairo_arc(cr, x + r,     y + r,     r,  M_PI,      3 * M_PI / 2);
	cairo_close_path(cr);
}

/* ── Leaf glyph for empty state (32x32 bezier) ── */

static void draw_leaf_glyph(cairo_t *cr, double cx, double cy) {
	cairo_save(cr);
	cairo_translate(cr, cx, cy);
	cairo_rotate(cr, 15.0 * M_PI / 180.0);

	cairo_move_to(cr, 0, -14);
	cairo_curve_to(cr, 10, -10, 14, 0, 0, 14);
	cairo_curve_to(cr, -14, 0, -10, -10, 0, -14);

	cairo_move_to(cr, 0, -10);
	cairo_line_to(cr, 0, 10);

	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.75);
	cairo_set_line_width(cr, 1.5);
	cairo_stroke(cr);

	cairo_restore(cr);
}

/* ── Taskbar logo (simple circle, ~12px radius) ── */

static void draw_logo_glyph(cairo_t *cr, double cx, double cy) {
	cairo_save(cr);
	cairo_arc(cr, cx, cy, 11.0, 0.0, 2.0 * M_PI);
	cairo_set_source_rgba(cr, 0.0, 0.0, 0.0, 0.75);
	cairo_set_line_width(cr, 1.5);
	cairo_stroke(cr);
	cairo_restore(cr);
}

/* ── PangoLayout helper with letter spacing ── */

static PangoLayout *create_layout(cairo_t *cr, PangoFontDescription *fd,
		int tracking) {
	PangoLayout *layout = pango_cairo_create_layout(cr);
	pango_layout_set_font_description(layout, fd);
	if (tracking > 0) {
		PangoAttrList *attrs = pango_attr_list_new();
		pango_attr_list_insert(attrs,
			pango_attr_letter_spacing_new(tracking));
		pango_layout_set_attributes(layout, attrs);
		pango_attr_list_unref(attrs);
	}
	return layout;
}

/* ── Time helper ── */

static double monotonic_time_s(void) {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* ── Card state helpers ── */

static bool state_is_pending(MarshalCardState s) {
	return s == CARD_STATE_PENDING || s == CARD_STATE_EXECUTING;
}

/* ── Renderer lifecycle ── */

struct marshal_renderer *renderer_create(void) {
	struct marshal_renderer *r = calloc(1, sizeof(*r));
	if (!r) return NULL;

	r->font_intent_title = pango_font_description_from_string(FONT_DESC_INTENT_TITLE);
	r->font_body         = pango_font_description_from_string(FONT_DESC_BODY);
	r->font_action_chain = pango_font_description_from_string(FONT_DESC_ACTION_CHAIN);
	r->font_capability   = pango_font_description_from_string(FONT_DESC_CAPABILITY);
	r->font_timing       = pango_font_description_from_string(FONT_DESC_TIMING);
	r->font_input        = pango_font_description_from_string(FONT_DESC_INPUT);
	r->font_keyboard_hint = pango_font_description_from_string(FONT_DESC_KEYBOARD_HINT);
	r->font_empty_heading = pango_font_description_from_string(FONT_DESC_EMPTY_HEADING);
	r->font_empty_sub    = pango_font_description_from_string(FONT_DESC_EMPTY_SUB);
	r->font_status       = pango_font_description_from_string(FONT_DESC_STATUS);

	r->font_overlay_title  = pango_font_description_from_string(FONT_DESC_OVERLAY_TITLE);
	r->font_overlay_body   = pango_font_description_from_string(FONT_DESC_OVERLAY_BODY);
	r->font_overlay_mono   = pango_font_description_from_string(FONT_DESC_OVERLAY_MONO);
	r->font_overlay_button = pango_font_description_from_string(FONT_DESC_OVERLAY_BUTTON);

	r->inference_online = true;
	r->card_sel.card_idx = -1;
	r->card_hit_count = 0;

	return r;
}

void renderer_destroy(struct marshal_renderer *r) {
	if (!r) return;
	if (r->cr) cairo_destroy(r->cr);
	if (r->surface) cairo_surface_destroy(r->surface);
	pango_font_description_free(r->font_intent_title);
	pango_font_description_free(r->font_body);
	pango_font_description_free(r->font_action_chain);
	pango_font_description_free(r->font_capability);
	pango_font_description_free(r->font_timing);
	pango_font_description_free(r->font_input);
	pango_font_description_free(r->font_keyboard_hint);
	pango_font_description_free(r->font_empty_heading);
	pango_font_description_free(r->font_empty_sub);
	pango_font_description_free(r->font_status);
	pango_font_description_free(r->font_overlay_title);
	pango_font_description_free(r->font_overlay_body);
	pango_font_description_free(r->font_overlay_mono);
	pango_font_description_free(r->font_overlay_button);
	if (r->wallpaper) cairo_surface_destroy(r->wallpaper);
	for (int i = 0; i < r->icon_cache_count; i++) {
		if (r->icon_cache[i].surface)
			cairo_surface_destroy(r->icon_cache[i].surface);
	}
	free(r);
}

void renderer_resize(struct marshal_renderer *r, int width, int height) {
	if (r->cr) cairo_destroy(r->cr);
	if (r->surface) cairo_surface_destroy(r->surface);
	r->width = width;
	r->height = height;
	r->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
	r->cr = cairo_create(r->surface);
}

/* ── Measure card height ── */

static int measure_card_height(struct marshal_renderer *r,
		MarshalIntent *intent, bool expanded) {
	cairo_t *cr = r->cr;
	int inner_w = r->width - 2 * CARD_MARGIN_H - CARD_INDICATOR_W
		- CARD_PADDING_H * 2;

	/* Line 1: intent text */
	PangoLayout *layout = create_layout(cr, r->font_intent_title, 0);
	pango_layout_set_text(layout, intent->natural_text, -1);
	pango_layout_set_width(layout, inner_w * PANGO_SCALE);
	pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
	pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
	pango_layout_set_height(layout, -2);
	int line1_w, line1_h;
	pango_layout_get_pixel_size(layout, &line1_w, &line1_h);
	g_object_unref(layout);

	int h = CARD_PADDING_V + line1_h;

	if (state_is_pending(intent->state) ||
			intent->state == CARD_STATE_AWAITING_CONFIRM) {
		/* Thinking / waiting text below */
		h += SPACE_S;
		PangoLayout *think = create_layout(cr, r->font_capability, 0);
		pango_layout_set_text(think, "thinking · · ·", -1);
		int tw, th;
		pango_layout_get_pixel_size(think, &tw, &th);
		g_object_unref(think);
		h += th;
		h += CARD_PADDING_V;
	} else if (intent->state == CARD_STATE_CANCELLED) {
		h += SPACE_S;
		PangoLayout *cl = create_layout(cr, r->font_capability, 0);
		pango_layout_set_text(cl, "cancelled", -1);
		int cw, ch;
		pango_layout_get_pixel_size(cl, &cw, &ch);
		g_object_unref(cl);
		h += ch;
		h += CARD_PADDING_V;
	} else if (intent->state == CARD_STATE_SEARCH_RESULT) {
		/* Search results: count line + hits */
		h += SPACE_S + 1 + SPACE_S; /* separator */
		PangoLayout *cnt = create_layout(cr, r->font_capability, 0);
		pango_layout_set_text(cnt, intent->action_chain, -1);
		int cw2, ch2;
		pango_layout_get_pixel_size(cnt, &cw2, &ch2);
		g_object_unref(cnt);
		h += ch2 + SPACE_XS;
		/* Each hit: title line + path line */
		int max_hits = intent->search_hit_count;
		if (max_hits > 5) max_hits = 5;
		h += max_hits * 36; /* ~18px title + 18px path */
		h += CARD_PADDING_V;
	} else {
		/* Completed / failed / history — full card */
		h += SPACE_S + 1 + SPACE_S; /* separator */

		if (intent->action_chain[0]) {
			PangoLayout *l2 = create_layout(cr, r->font_action_chain, 0);
			pango_layout_set_text(l2, intent->action_chain, -1);
			int l2w, l2h;
			pango_layout_get_pixel_size(l2, &l2w, &l2h);
			g_object_unref(l2);
			h += l2h + SPACE_XS;
		}

		/* Result summary */
		if (intent->result_summary[0]) {
			PangoLayout *rs = create_layout(cr, r->font_body, 0);
			pango_layout_set_text(rs, intent->result_summary, -1);
			pango_layout_set_width(rs, inner_w * PANGO_SCALE);
			pango_layout_set_wrap(rs, PANGO_WRAP_WORD_CHAR);
			if (!expanded)
				pango_layout_set_height(rs, -4);
			/* expanded: no height limit — show all content */
			int rsw, rsh;
			pango_layout_get_pixel_size(rs, &rsw, &rsh);
			g_object_unref(rs);
			h += rsh + SPACE_S;
		}

		/* Injection block height */
		if (intent->injection_detected && intent->injection_content[0]) {
			int inj_block_h = SPACE_S + 40 + SPACE_S + 18 + SPACE_S;
			if (intent->sandbox_active)
				inj_block_h += 16 + SPACE_XS;
			h += SPACE_XS + inj_block_h + SPACE_S;
		}

		int bottom_h = 0;
		if (intent->capability_scope[0]) {
			PangoLayout *l3 = create_layout(cr, r->font_capability, 0);
			pango_layout_set_text(l3, "x test", -1);
			int l3w, l3h;
			pango_layout_get_pixel_size(l3, &l3w, &l3h);
			g_object_unref(l3);
			if (l3h > bottom_h) bottom_h = l3h;
		}
		{
			PangoLayout *l4 = create_layout(cr, r->font_timing,
				TRACKING_CAPTION2);
			pango_layout_set_text(l4, "34.2s  ·  done", -1);
			int l4w, l4h;
			pango_layout_get_pixel_size(l4, &l4w, &l4h);
			g_object_unref(l4);
			if (l4h > bottom_h) bottom_h = l4h;
		}
		h += bottom_h;
		h += CARD_PADDING_V;
	}

	return h;
}

/* ── Indicator color per state ── */

static struct color indicator_for_state(MarshalCardState state, float opacity) {
	struct color c;
	switch (state) {
	case CARD_STATE_PENDING:
	case CARD_STATE_EXECUTING: {
		/* Pulsing blue */
		double t = monotonic_time_s();
		double pulse = 0.45 + 0.55 * (sin(t * M_PI / 0.8) * 0.5 + 0.5);
		c = ACCENT_BLUE;
		c.a = (uint8_t)(pulse * 255.0 * opacity);
		return c;
	}
	case CARD_STATE_AWAITING_CONFIRM:
		c = ACCENT_AMBER;
		c.a = (uint8_t)(c.a * opacity);
		return c;
	case CARD_STATE_DONE:
		c = ACCENT_GREEN;
		c.a = (uint8_t)(c.a * opacity);
		return c;
	case CARD_STATE_SEARCH_RESULT:
		c = ACCENT_AMBER;
		c.a = (uint8_t)(c.a * opacity);
		return c;
	case CARD_STATE_FAILED:
		c = ACCENT_RED;
		c.a = (uint8_t)(c.a * opacity);
		return c;
	case CARD_STATE_CANCELLED:
	case CARD_STATE_HISTORY:
	default:
		c = INDICATOR_HISTORY;
		c.a = (uint8_t)(c.a * opacity);
		return c;
	}
}

/* ── Draw a single intent card ── */

static int draw_card(struct marshal_renderer *r, MarshalIntent *intent,
		int y_base, bool selected, int card_index, bool expanded) {
	cairo_t *cr = r->cr;
	int card_w = r->width - 2 * CARD_MARGIN_H;
	int card_x = CARD_MARGIN_H;

	float opacity = intent->anim_opacity.pos;
	if (opacity < 0) opacity = 0;
	if (opacity > 1) opacity = 1;
	float y_off = intent->anim_y.pos;
	int card_y = y_base + (int)y_off;

	int card_h = measure_card_height(r, intent, expanded);

	/* Dim cancelled cards */
	float text_dim = (intent->state == CARD_STATE_CANCELLED) ? 0.4f : 1.0f;

	/* Card background: BG_ELEVATED */
	rounded_rect(cr, card_x, card_y, card_w, card_h, CARD_RADIUS);
	{
		struct color c = BG_ELEVATED;
		cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
			c.b / 255.0, opacity);
	}
	cairo_fill(cr);

	/* Card border */
	cairo_save(cr);
	rounded_rect(cr, card_x, card_y, card_w, card_h, CARD_RADIUS);
	if (selected) {
		struct color c = ACCENT_BLUE;
		cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
			c.b / 255.0, 0.6 * opacity);
		cairo_set_line_width(cr, 1.5);
	} else if (intent->proactive) {
		struct color c = ACCENT_GREEN;
		cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
			c.b / 255.0, 0.55 * opacity);
		cairo_set_line_width(cr, 1.25);
	} else {
		struct color c = BORDER_CARD;
		cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
			c.b / 255.0, (c.a / 255.0) * opacity);
		cairo_set_line_width(cr, 1.0);
	}
	cairo_stroke(cr);
	cairo_restore(cr);

	/* Indicator bar. Proactive cards (pushed by the OS, not typed by the
	 * user) override the state-based color with the accent green so it's
	 * visually obvious the OS wrote this card on its own. */
	struct color indicator_color = intent->proactive
		? (struct color){ACCENT_GREEN.r, ACCENT_GREEN.g, ACCENT_GREEN.b,
			(uint8_t)(ACCENT_GREEN.a * opacity)}
		: indicator_for_state(intent->state, opacity);
	cairo_save(cr);
	{
		double ix = card_x;
		double iy = card_y;
		double iw = CARD_INDICATOR_W;
		double ih = card_h;
		double ir = CARD_RADIUS;
		cairo_new_sub_path(cr);
		cairo_arc(cr, ix + ir, iy + ir,     ir, M_PI, 3 * M_PI / 2);
		cairo_line_to(cr, ix + iw, iy);
		cairo_line_to(cr, ix + iw, iy + ih);
		cairo_arc(cr, ix + ir, iy + ih - ir, ir, M_PI / 2, M_PI);
		cairo_close_path(cr);
	}
	set_color(cr, indicator_color);
	cairo_fill(cr);
	cairo_restore(cr);

	int text_x = card_x + CARD_INDICATOR_W + CARD_PADDING_H;
	int text_y = card_y + CARD_PADDING_V;
	int inner_w = card_w - CARD_INDICATOR_W - CARD_PADDING_H * 2;

	/* Track combined selectable text area for hit-testing */
	int sel_text_start_y = text_y;  /* top of selectable region */
	int result_text_y = 0;          /* y of result summary (set later) */

	/* LINE 1: intent natural text */
	{
		PangoLayout *layout = create_layout(cr, r->font_intent_title, 0);
		pango_layout_set_text(layout, intent->natural_text, -1);
		pango_layout_set_width(layout, inner_w * PANGO_SCALE);
		pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
		pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
		pango_layout_set_height(layout, -2);

		/* Draw selection highlight behind title text */
		if (r->card_sel.card_idx == card_index &&
				r->card_sel.anchor != r->card_sel.focus) {
			int sel_s = r->card_sel.anchor < r->card_sel.focus
				? r->card_sel.anchor : r->card_sel.focus;
			int sel_e = r->card_sel.anchor < r->card_sel.focus
				? r->card_sel.focus : r->card_sel.anchor;
			int title_len = (int)strlen(intent->natural_text);
			if (sel_s < title_len && sel_e > 0) {
				int s = sel_s < 0 ? 0 : sel_s;
				int e = sel_e > title_len ? title_len : sel_e;
				/* Use Pango's line-based index_to_x for each line */
				PangoLayoutIter *iter = pango_layout_get_iter(layout);
				do {
					PangoRectangle logical;
					pango_layout_iter_get_line_extents(iter, NULL, &logical);
					PangoLayoutLine *line = pango_layout_iter_get_line_readonly(iter);
					int ls = line->start_index;
					int le = ls + line->length;
					int cs = s > ls ? s : ls;
					int ce = e < le ? e : le;
					if (cs < ce) {
						int x1, x2;
						pango_layout_line_index_to_x(line, cs, FALSE, &x1);
						pango_layout_line_index_to_x(line, ce, FALSE, &x2);
						if (x1 > x2) { int tmp = x1; x1 = x2; x2 = tmp; }
						cairo_save(cr);
						cairo_set_source_rgba(cr, 0.14, 0.39, 0.92, 0.3);
						cairo_rectangle(cr,
							text_x + (double)x1 / PANGO_SCALE,
							text_y + (double)logical.y / PANGO_SCALE,
							(double)(x2 - x1) / PANGO_SCALE,
							(double)logical.height / PANGO_SCALE);
						cairo_fill(cr);
						cairo_restore(cr);
					}
				} while (pango_layout_iter_next_line(iter));
				pango_layout_iter_free(iter);
			}
		}

		cairo_move_to(cr, text_x, text_y);
		if (state_is_pending(intent->state) ||
				intent->state == CARD_STATE_AWAITING_CONFIRM) {
			struct color c = TEXT_SECONDARY;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, (c.a / 255.0) * opacity);
		} else {
			struct color c = TEXT_PRIMARY;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, (c.a / 255.0) * opacity * text_dim);
		}
		pango_cairo_show_layout(cr, layout);

		int lw, lh;
		pango_layout_get_pixel_size(layout, &lw, &lh);
		text_y += lh;
		g_object_unref(layout);
	}

	/* State-dependent content below title */
	if (state_is_pending(intent->state)) {
		/* Thinking animation */
		text_y += SPACE_S;
		double t = monotonic_time_s();
		int phase = (int)((t * 1000) / 500) % 4;
		const char *states[] = {"thinking", "thinking ·",
			"thinking · ·", "thinking · · ·"};

		PangoLayout *layout = create_layout(cr, r->font_capability, 0);
		pango_layout_set_text(layout, states[phase], -1);
		cairo_move_to(cr, text_x, text_y);
		{
			struct color c = ACCENT_BLUE;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, 0.70 * opacity);
		}
		pango_cairo_show_layout(cr, layout);
		g_object_unref(layout);
	} else if (intent->state == CARD_STATE_AWAITING_CONFIRM) {
		/* Waiting for confirmation */
		text_y += SPACE_S;
		double t = monotonic_time_s();
		int phase = (int)((t * 1000) / 500) % 4;
		const char *states[] = {"waiting for confirmation",
			"waiting for confirmation ·",
			"waiting for confirmation · ·",
			"waiting for confirmation · · ·"};

		PangoLayout *layout = create_layout(cr, r->font_capability, 0);
		pango_layout_set_text(layout, states[phase], -1);
		cairo_move_to(cr, text_x, text_y);
		{
			struct color c = ACCENT_AMBER;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, 0.70 * opacity);
		}
		pango_cairo_show_layout(cr, layout);
		g_object_unref(layout);
	} else if (intent->state == CARD_STATE_CANCELLED) {
		/* Cancelled text */
		text_y += SPACE_S;
		PangoLayout *layout = create_layout(cr, r->font_capability, 0);
		pango_layout_set_text(layout, "cancelled", -1);
		cairo_move_to(cr, text_x, text_y);
		{
			struct color c = TEXT_TERTIARY;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, (c.a / 255.0) * opacity);
		}
		pango_cairo_show_layout(cr, layout);
		g_object_unref(layout);
	} else if (intent->state == CARD_STATE_SEARCH_RESULT) {
		/* Search result card */

		/* Separator */
		text_y += SPACE_S;
		{
			struct color c = BORDER_SEPARATOR;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, (c.a / 255.0) * opacity);
		}
		cairo_set_line_width(cr, 0.5);
		cairo_move_to(cr, text_x, text_y + 0.5);
		cairo_line_to(cr, card_x + card_w - CARD_PADDING_H,
			text_y + 0.5);
		cairo_stroke(cr);
		text_y += 1 + SPACE_S;

		/* Result count */
		{
			PangoLayout *cnt_layout = create_layout(cr,
				r->font_capability, 0);
			pango_layout_set_text(cnt_layout,
				intent->action_chain, -1);
			cairo_move_to(cr, text_x, text_y);
			{
				struct color c = ACCENT_AMBER;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					opacity);
			}
			pango_cairo_show_layout(cr, cnt_layout);
			int cw3, ch3;
			pango_layout_get_pixel_size(cnt_layout, &cw3, &ch3);
			text_y += ch3 + SPACE_XS;
			g_object_unref(cnt_layout);
		}

		/* Individual hits */
		int max_hits = intent->search_hit_count;
		if (max_hits > 5) max_hits = 5;
		for (int h = 0; h < max_hits; h++) {
			MarshalSearchHit *hit = &intent->search_hits[h];

			/* Title */
			PangoLayout *tl = create_layout(cr,
				r->font_action_chain, 0);
			pango_layout_set_text(tl, hit->title, -1);
			pango_layout_set_width(tl, inner_w * PANGO_SCALE);
			pango_layout_set_ellipsize(tl, PANGO_ELLIPSIZE_END);
			pango_layout_set_single_paragraph_mode(tl, TRUE);
			cairo_move_to(cr, text_x, text_y);
			{
				struct color c = TEXT_PRIMARY;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					(c.a / 255.0) * opacity);
			}
			pango_cairo_show_layout(cr, tl);
			int tw2, th2;
			pango_layout_get_pixel_size(tl, &tw2, &th2);
			text_y += th2;
			g_object_unref(tl);

			/* Path */
			PangoLayout *pl = create_layout(cr,
				r->font_capability, 0);
			pango_layout_set_text(pl, hit->path, -1);
			pango_layout_set_width(pl, inner_w * PANGO_SCALE);
			pango_layout_set_ellipsize(pl, PANGO_ELLIPSIZE_END);
			pango_layout_set_single_paragraph_mode(pl, TRUE);
			cairo_move_to(cr, text_x, text_y);
			{
				struct color c = TEXT_TERTIARY;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					(c.a / 255.0) * opacity);
			}
			pango_cairo_show_layout(cr, pl);
			int pw2, ph2;
			pango_layout_get_pixel_size(pl, &pw2, &ph2);
			text_y += ph2 + SPACE_XS;
			g_object_unref(pl);
		}
	} else {
		/* DONE / FAILED / HISTORY — full card content */

		/* Separator line */
		text_y += SPACE_S;
		{
			struct color c = BORDER_SEPARATOR;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, (c.a / 255.0) * opacity);
		}
		cairo_set_line_width(cr, 0.5);
		cairo_move_to(cr, text_x, text_y + 0.5);
		cairo_line_to(cr, card_x + card_w - CARD_PADDING_H, text_y + 0.5);
		cairo_stroke(cr);
		text_y += 1 + SPACE_S;

		/* LINE 2: action chain */
		if (intent->action_chain[0]) {
			PangoLayout *layout = create_layout(cr,
				r->font_action_chain, 0);

			/* Build markup: default TEXT_SECONDARY, arrows ACCENT_BLUE */
			char markup[1024] = {0};
			int off = 0;
			off += snprintf(markup + off, sizeof(markup) - off,
				"<span foreground=\"#%02X%02X%02X\">",
				TEXT_SECONDARY.r, TEXT_SECONDARY.g, TEXT_SECONDARY.b);
			const char *src = intent->action_chain;
			while (*src && off < (int)sizeof(markup) - 80) {
				if (strncmp(src, "\xe2\x86\x92", 3) == 0) {
					off += snprintf(markup + off, sizeof(markup) - off,
						"</span><span foreground=\"#2563EB\">  \xe2\x86\x92  </span>"
						"<span foreground=\"#%02X%02X%02X\">",
						TEXT_SECONDARY.r, TEXT_SECONDARY.g,
						TEXT_SECONDARY.b);
					src += 3;
					while (*src == ' ') src++;
				} else if (*src == '&') {
					off += snprintf(markup + off, sizeof(markup) - off,
						"&amp;");
					src++;
				} else if (*src == '<') {
					off += snprintf(markup + off, sizeof(markup) - off,
						"&lt;");
					src++;
				} else if (*src == '>') {
					off += snprintf(markup + off, sizeof(markup) - off,
						"&gt;");
					src++;
				} else {
					markup[off++] = *src++;
				}
			}
			off += snprintf(markup + off, sizeof(markup) - off,
				"</span>");

			pango_layout_set_markup(layout, markup, -1);
			pango_layout_set_width(layout, inner_w * PANGO_SCALE);
			pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
			pango_layout_set_single_paragraph_mode(layout, TRUE);

			cairo_move_to(cr, text_x, text_y);
			cairo_set_source_rgba(cr, 0, 0, 0, opacity);
			pango_cairo_show_layout(cr, layout);

			int lw, lh;
			pango_layout_get_pixel_size(layout, &lw, &lh);
			text_y += lh + SPACE_XS;
			g_object_unref(layout);
		}

		/* Result summary text */
		result_text_y = text_y;  /* track for hit-test */
		if (intent->result_summary[0]) {
			PangoLayout *layout = create_layout(cr,
				r->font_body, 0);
			pango_layout_set_text(layout,
				intent->result_summary, -1);
			pango_layout_set_width(layout,
				inner_w * PANGO_SCALE);
			pango_layout_set_wrap(layout,
				PANGO_WRAP_WORD_CHAR);
			if (!expanded)
				pango_layout_set_height(layout, -4);
			/* expanded: no height limit — show full content */
			if (!expanded)
				pango_layout_set_ellipsize(layout,
					PANGO_ELLIPSIZE_END);

			/* Draw selection highlight for result text */
			if (r->card_sel.card_idx == card_index &&
					r->card_sel.anchor != r->card_sel.focus) {
				int title_len = (int)strlen(intent->natural_text);
				int base_off = title_len + 1; /* +1 for \n separator */
				int rlen = (int)strlen(intent->result_summary);
				int sel_s = r->card_sel.anchor < r->card_sel.focus
					? r->card_sel.anchor : r->card_sel.focus;
				int sel_e = r->card_sel.anchor < r->card_sel.focus
					? r->card_sel.focus : r->card_sel.anchor;
				int rs = sel_s - base_off;
				int re = sel_e - base_off;
				if (rs < rlen && re > 0) {
					if (rs < 0) rs = 0;
					if (re > rlen) re = rlen;
					PangoLayoutIter *iter = pango_layout_get_iter(layout);
					do {
						PangoRectangle logical;
						pango_layout_iter_get_line_extents(iter, NULL, &logical);
						PangoLayoutLine *line = pango_layout_iter_get_line_readonly(iter);
						int ls = line->start_index;
						int le = ls + line->length;
						int cs = rs > ls ? rs : ls;
						int ce = re < le ? re : le;
						if (cs < ce) {
							int x1, x2;
							pango_layout_line_index_to_x(line, cs, FALSE, &x1);
							pango_layout_line_index_to_x(line, ce, FALSE, &x2);
							if (x1 > x2) { int tmp = x1; x1 = x2; x2 = tmp; }
							cairo_save(cr);
							cairo_set_source_rgba(cr, 0.14, 0.39, 0.92, 0.3);
							cairo_rectangle(cr,
								text_x + (double)x1 / PANGO_SCALE,
								text_y + (double)logical.y / PANGO_SCALE,
								(double)(x2 - x1) / PANGO_SCALE,
								(double)logical.height / PANGO_SCALE);
							cairo_fill(cr);
							cairo_restore(cr);
						}
					} while (pango_layout_iter_next_line(iter));
					pango_layout_iter_free(iter);
				}
			}

			cairo_move_to(cr, text_x, text_y);
			{
				struct color c = TEXT_PRIMARY;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					(c.a / 255.0) * opacity * text_dim);
			}
			pango_cairo_show_layout(cr, layout);
			int rsw, rsh;
			pango_layout_get_pixel_size(layout, &rsw, &rsh);
			text_y += rsh + SPACE_S;
			g_object_unref(layout);
		}

		/* Injection detection block */
		if (intent->injection_detected &&
				intent->injection_content[0]) {
			text_y += SPACE_XS;

			/* Red background block */
			int inj_x = text_x;
			int inj_w = inner_w;

			/* Measure injection content */
			PangoLayout *inj_layout = create_layout(cr,
				r->font_capability, 0);
			pango_layout_set_text(inj_layout,
				intent->injection_content, -1);
			pango_layout_set_width(inj_layout,
				(inj_w - 2 * SPACE_S) * PANGO_SCALE);
			pango_layout_set_wrap(inj_layout, PANGO_WRAP_WORD_CHAR);
			int iw, ih;
			pango_layout_get_pixel_size(inj_layout, &iw, &ih);

			/* Red tinted background — strong enough to read as a block */
			int block_h = SPACE_S + ih + SPACE_S + 18 + SPACE_S;
			if (intent->sandbox_active)
				block_h += 16 + SPACE_XS;
			rounded_rect(cr, inj_x, text_y, inj_w, block_h,
				CARD_RADIUS);
			{
				struct color c = ACCENT_RED;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					0.10 * opacity);
			}
			cairo_fill(cr);

			/* Solid left accent bar */
			cairo_rectangle(cr, inj_x, text_y, 4, block_h);
			{
				struct color c = ACCENT_RED;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					opacity);
			}
			cairo_fill(cr);

			/* Injection snippet — rendered as normal dark text so
			 * benign lines don't read as alarming; the red tint +
			 * left bar + label below carry the warning. */
			cairo_move_to(cr, inj_x + SPACE_S + 4,
				text_y + SPACE_S);
			{
				struct color c = TEXT_PRIMARY;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					(c.a / 255.0) * opacity);
			}
			pango_cairo_show_layout(cr, inj_layout);
			g_object_unref(inj_layout);

			/* INJECTION DETECTED label — bold, full-opacity red */
			int label_y = text_y + SPACE_S + ih + SPACE_S;
			PangoLayout *bl = create_layout(cr,
				r->font_action_chain, 0);
			pango_layout_set_markup(bl,
				"<span weight=\"heavy\" letter_spacing=\"600\">"
				"INJECTION DETECTED IN CONTENT</span>", -1);
			cairo_move_to(cr, inj_x + SPACE_S + 4, label_y);
			{
				struct color c = ACCENT_RED;
				cairo_set_source_rgba(cr, c.r / 255.0,
					c.g / 255.0, c.b / 255.0,
					opacity);
			}
			pango_cairo_show_layout(cr, bl);
			g_object_unref(bl);
			label_y += 18;

			/* Sandbox info */
			if (intent->sandbox_active) {
				label_y += SPACE_XS;
				char sandbox_text[600];
				snprintf(sandbox_text, sizeof(sandbox_text),
					"\xe2\x9c\x93 Landlock sandbox active — "
					"authorized: [%s]",
					intent->authorized_paths);
				PangoLayout *sl = create_layout(cr,
					r->font_capability, 0);
				pango_layout_set_text(sl,
					sandbox_text, -1);
				pango_layout_set_width(sl,
					(inj_w - 2 * SPACE_S) * PANGO_SCALE);
				pango_layout_set_ellipsize(sl,
					PANGO_ELLIPSIZE_END);
				pango_layout_set_single_paragraph_mode(sl,
					TRUE);
				cairo_move_to(cr, inj_x + SPACE_S, label_y);
				{
					struct color c = ACCENT_GREEN;
					cairo_set_source_rgba(cr,
						c.r / 255.0, c.g / 255.0,
						c.b / 255.0, opacity);
				}
				pango_cairo_show_layout(cr, sl);
				g_object_unref(sl);
			}

			text_y += block_h + SPACE_S;
		}

		/* LINE 3: capability scope (left) + LINE 4: timing (right) */
		int bottom_y = text_y;

		if (intent->capability_scope[0]) {
			PangoLayout *layout = create_layout(cr,
				r->font_capability, 0);
			char scope_text[600];
			snprintf(scope_text, sizeof(scope_text), "\xe2\x9a\x91 %s",
				intent->capability_scope);
			pango_layout_set_text(layout, scope_text, -1);
			pango_layout_set_width(layout,
				(inner_w * 2 / 3) * PANGO_SCALE);
			pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
			pango_layout_set_single_paragraph_mode(layout, TRUE);

			cairo_move_to(cr, text_x, bottom_y);
			{
				struct color c = ACCENT_GREEN;
				cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
					c.b / 255.0, opacity);
			}
			pango_cairo_show_layout(cr, layout);
			g_object_unref(layout);
		}

		/* Timing (right-aligned) */
		{
			PangoLayout *layout = create_layout(cr, r->font_timing,
				TRACKING_CAPTION2);
			char timing[256];
			const char *status_str;
			const char *status_hex;
			if (intent->state == CARD_STATE_FAILED) {
				status_str = "failed";
				status_hex = "#DC2626";
			} else {
				status_str = "done";
				status_hex = "#16A34A";
			}
			snprintf(timing, sizeof(timing),
				"<span foreground=\"#%02X%02X%02X\">%.1fs  \xc2\xb7  </span>"
				"<span foreground=\"%s\">%s</span>",
				TEXT_TERTIARY.r, TEXT_TERTIARY.g, TEXT_TERTIARY.b,
				intent->duration_ms / 1000.0,
				status_hex, status_str);
			pango_layout_set_markup(layout, timing, -1);

			int tw, th;
			pango_layout_get_pixel_size(layout, &tw, &th);
			int timing_x = card_x + card_w - CARD_PADDING_H - tw;
			cairo_move_to(cr, timing_x, bottom_y);
			cairo_set_source_rgba(cr, 0, 0, 0, opacity);
			pango_cairo_show_layout(cr, layout);
			g_object_unref(layout);
		}
	}

	/* Record card text hit-test geometry for mouse selection */
	if (card_index >= 0 && r->card_hit_count < 200) {
		struct card_text_hit *hit = &r->card_hits[r->card_hit_count];
		memset(hit, 0, sizeof(*hit));
		hit->card_idx = card_index;
		hit->text_x = text_x;
		hit->text_w = inner_w;

		/* Title region */
		hit->title_y = sel_text_start_y;
		int tl = (int)strlen(intent->natural_text);
		if (tl > (int)sizeof(hit->title) - 1)
			tl = (int)sizeof(hit->title) - 1;
		memcpy(hit->title, intent->natural_text, tl);
		hit->title[tl] = '\0';
		hit->title_len = tl;

		/* Measure title height with the correct font */
		{
			PangoLayout *tmp = create_layout(r->cr,
				r->font_intent_title, 0);
			pango_layout_set_text(tmp, hit->title, hit->title_len);
			pango_layout_set_width(tmp, inner_w * PANGO_SCALE);
			pango_layout_set_wrap(tmp, PANGO_WRAP_WORD_CHAR);
			pango_layout_set_height(tmp, -2);
			int tw, th;
			pango_layout_get_pixel_size(tmp, &tw, &th);
			hit->title_h = th;
			g_object_unref(tmp);
		}

		/* Result region */
		if (intent->result_summary[0]) {
			hit->result_y = result_text_y;
			int rl = (int)strlen(intent->result_summary);
			if (rl > (int)sizeof(hit->result) - 1)
				rl = (int)sizeof(hit->result) - 1;
			memcpy(hit->result, intent->result_summary, rl);
			hit->result[rl] = '\0';
			hit->result_len = rl;

			/* Measure result height */
			PangoLayout *tmp = create_layout(r->cr,
				r->font_body, 0);
			pango_layout_set_text(tmp, hit->result, hit->result_len);
			pango_layout_set_width(tmp, inner_w * PANGO_SCALE);
			pango_layout_set_wrap(tmp, PANGO_WRAP_WORD_CHAR);
			if (!expanded)
				pango_layout_set_height(tmp, -4);
			int rw, rh;
			pango_layout_get_pixel_size(tmp, &rw, &rh);
			hit->result_h = rh;
			g_object_unref(tmp);
		}

		/* Build combined text for copy: title + \n + result */
		int off = 0;
		memcpy(hit->text, hit->title, hit->title_len);
		off = hit->title_len;
		if (hit->result_len > 0) {
			hit->text[off++] = '\n';
			int rl = hit->result_len;
			if (off + rl > (int)sizeof(hit->text) - 1)
				rl = (int)sizeof(hit->text) - 1 - off;
			memcpy(hit->text + off, hit->result, rl);
			off += rl;
		}
		hit->text[off] = '\0';
		hit->text_len = off;
		r->card_hit_count++;
	}

	return card_h;
}

/* ── Briefing card ── */

static int draw_briefing_card(struct marshal_renderer *r,
		MarshalBriefing *briefing, int y) {
	if (!briefing->loaded) return 0;

	cairo_t *cr = r->cr;
	int card_x = CARD_MARGIN_H;
	int card_w = r->width - 2 * CARD_MARGIN_H;
	int inner_w = card_w - 2 * CARD_PADDING_H - CARD_INDICATOR_W - SPACE_S;
	float alpha = briefing->anim_opacity.pos;
	if (alpha < 0.01f) return 0;

	/* Measure content height */
	int content_h = 0;

	/* Headline */
	PangoLayout *headline = create_layout(cr, r->font_intent_title, 0);
	pango_layout_set_text(headline, briefing->headline, -1);
	pango_layout_set_width(headline, inner_w * PANGO_SCALE);
	pango_layout_set_wrap(headline, PANGO_WRAP_WORD);
	int hw, hh;
	pango_layout_get_pixel_size(headline, &hw, &hh);
	content_h += hh + SPACE_S;

	/* Group lines */
	int group_lines = 0;
	for (int s = 0; s < briefing->section_count; s++) {
		MarshalBriefingSection *sec = &briefing->sections[s];
		for (int g = 0; g < sec->group_count; g++) {
			group_lines++;
			MarshalBriefingGroup *grp = &sec->groups[g];
			group_lines += (grp->item_count > 3) ? 3 : grp->item_count;
		}
	}
	int line_h = 18;
	content_h += group_lines * line_h;

	int card_h = CARD_PADDING_V + content_h + CARD_PADDING_V;

	/* Card background */
	cairo_save(cr);
	rounded_rect(cr, card_x, y, card_w, card_h, CARD_RADIUS);
	set_color(cr, color_with_alpha(FILL_CARD,
		(uint8_t)(FILL_CARD.a * alpha)));
	cairo_fill_preserve(cr);
	set_color(cr, color_with_alpha(BORDER_CARD,
		(uint8_t)(BORDER_CARD.a * alpha)));
	cairo_set_line_width(cr, 1);
	cairo_stroke(cr);

	/* Blue indicator bar (briefing accent) */
	rounded_rect(cr, card_x + SPACE_XS, y + CARD_PADDING_V,
		CARD_INDICATOR_W, card_h - 2 * CARD_PADDING_V,
		CARD_INDICATOR_W / 2.0);
	set_color(cr, color_with_alpha(ACCENT_BLUE,
		(uint8_t)(ACCENT_BLUE.a * alpha)));
	cairo_fill(cr);

	/* Headline text */
	int text_x = card_x + CARD_PADDING_H + CARD_INDICATOR_W + SPACE_S;
	int text_y = y + CARD_PADDING_V;
	cairo_move_to(cr, text_x, text_y);
	set_color(cr, color_with_alpha(TEXT_PRIMARY,
		(uint8_t)(TEXT_PRIMARY.a * alpha)));
	pango_cairo_show_layout(cr, headline);
	g_object_unref(headline);
	text_y += hh + SPACE_S;

	/* Groups */
	for (int s = 0; s < briefing->section_count; s++) {
		MarshalBriefingSection *sec = &briefing->sections[s];
		for (int g = 0; g < sec->group_count; g++) {
			MarshalBriefingGroup *grp = &sec->groups[g];
			char dir_line[300];
			snprintf(dir_line, sizeof(dir_line), "%s (%d)",
				grp->directory, grp->count);

			PangoLayout *dir_layout = create_layout(cr,
				r->font_action_chain, 0);
			pango_layout_set_text(dir_layout, dir_line, -1);
			cairo_move_to(cr, text_x, text_y);
			set_color(cr, color_with_alpha(TEXT_SECONDARY,
				(uint8_t)(TEXT_SECONDARY.a * alpha)));
			pango_cairo_show_layout(cr, dir_layout);
			g_object_unref(dir_layout);
			text_y += line_h;

			int show = (grp->item_count > 3) ? 3 : grp->item_count;
			for (int i = 0; i < show; i++) {
				char item_line[300];
				snprintf(item_line, sizeof(item_line),
					"  · %s", grp->items[i].title);
				PangoLayout *il = create_layout(cr,
					r->font_capability, 0);
				pango_layout_set_text(il, item_line, -1);
				cairo_move_to(cr, text_x, text_y);
				set_color(cr, color_with_alpha(TEXT_TERTIARY,
					(uint8_t)(TEXT_TERTIARY.a * alpha)));
				pango_cairo_show_layout(cr, il);
				g_object_unref(il);
				text_y += line_h;
			}
		}
	}

	cairo_restore(cr);
	return card_h;
}

/* ── Watcher cards ── */

static int draw_watcher_cards(struct marshal_renderer *r,
		struct marshal_feed *feed, int y) {
	if (feed->watcher_count == 0) return 0;

	cairo_t *cr = r->cr;
	int card_x = CARD_MARGIN_H;
	int card_w = r->width - 2 * CARD_MARGIN_H;
	int inner_w = card_w - 2 * CARD_PADDING_H - CARD_INDICATOR_W - SPACE_S;
	int total_h = 0;

	for (int w = 0; w < feed->watcher_count; w++) {
		MarshalWatcher *watcher = &feed->watchers[w];
		float alpha = watcher->anim_opacity.pos;
		if (alpha < 0.01f) continue;

		/* Compact card: name + path on one card, ~48px tall */
		int card_h = CARD_PADDING_V + 18 + SPACE_XS + 16 + CARD_PADDING_V;
		int card_y = y + total_h;

		/* Background */
		cairo_save(cr);
		rounded_rect(cr, card_x, card_y, card_w, card_h, CARD_RADIUS);
		set_color(cr, color_with_alpha(FILL_CARD,
			(uint8_t)(FILL_CARD.a * alpha)));
		cairo_fill_preserve(cr);
		set_color(cr, color_with_alpha(BORDER_CARD,
			(uint8_t)(BORDER_CARD.a * alpha)));
		cairo_set_line_width(cr, 1);
		cairo_stroke(cr);

		/* Green indicator bar */
		{
			double ix = card_x;
			double iy = card_y;
			double iw = CARD_INDICATOR_W;
			double ih = card_h;
			double ir = CARD_RADIUS;
			cairo_new_sub_path(cr);
			cairo_arc(cr, ix + ir, iy + ir, ir,
				M_PI, 3 * M_PI / 2);
			cairo_line_to(cr, ix + iw, iy);
			cairo_line_to(cr, ix + iw, iy + ih);
			cairo_arc(cr, ix + ir, iy + ih - ir, ir,
				M_PI / 2, M_PI);
			cairo_close_path(cr);
		}
		set_color(cr, color_with_alpha(ACCENT_GREEN,
			(uint8_t)(ACCENT_GREEN.a * alpha)));
		cairo_fill(cr);

		int text_x = card_x + CARD_INDICATOR_W + CARD_PADDING_H;
		int text_y = card_y + CARD_PADDING_V;

		/* Pulsing green dot */
		{
			double t = monotonic_time_s();
			double pulse = 0.4 + 0.6 *
				(sin(t * M_PI / 1.2) * 0.5 + 0.5);
			cairo_arc(cr, text_x + 5, text_y + 8, 4, 0,
				2 * M_PI);
			struct color gc = ACCENT_GREEN;
			cairo_set_source_rgba(cr, gc.r / 255.0,
				gc.g / 255.0, gc.b / 255.0,
				pulse * alpha);
			cairo_fill(cr);
		}

		/* Watcher name */
		{
			PangoLayout *nl = create_layout(cr,
				r->font_action_chain, 0);
			pango_layout_set_text(nl, watcher->name, -1);
			pango_layout_set_width(nl,
				(inner_w - 20) * PANGO_SCALE);
			pango_layout_set_ellipsize(nl,
				PANGO_ELLIPSIZE_END);
			pango_layout_set_single_paragraph_mode(nl, TRUE);
			cairo_move_to(cr, text_x + 16, text_y);
			set_color(cr, color_with_alpha(TEXT_PRIMARY,
				(uint8_t)(TEXT_PRIMARY.a * alpha)));
			pango_cairo_show_layout(cr, nl);
			int nw, nh;
			pango_layout_get_pixel_size(nl, &nw, &nh);
			text_y += nh + SPACE_XS;
			g_object_unref(nl);
		}

		/* Path + pattern + fire count */
		{
			char detail[512];
			if (watcher->pattern[0])
				snprintf(detail, sizeof(detail),
					"watching %s for %s  ·  %d fired",
					watcher->watched_path,
					watcher->pattern,
					watcher->fire_count);
			else
				snprintf(detail, sizeof(detail),
					"watching %s  ·  %d fired",
					watcher->watched_path,
					watcher->fire_count);

			PangoLayout *dl = create_layout(cr,
				r->font_capability, 0);
			pango_layout_set_text(dl, detail, -1);
			pango_layout_set_width(dl, inner_w * PANGO_SCALE);
			pango_layout_set_ellipsize(dl,
				PANGO_ELLIPSIZE_END);
			pango_layout_set_single_paragraph_mode(dl, TRUE);
			cairo_move_to(cr, text_x + 16, text_y);
			set_color(cr, color_with_alpha(TEXT_TERTIARY,
				(uint8_t)(TEXT_TERTIARY.a * alpha)));
			pango_cairo_show_layout(cr, dl);
			g_object_unref(dl);
		}

		cairo_restore(cr);
		total_h += card_h + CARD_GAP;
	}

	return total_h;
}

/* ── Empty state ── */

/* ── JPEG wallpaper loader ── */

void renderer_load_wallpaper(struct marshal_renderer *r, const char *path) {
	if (r->wallpaper) {
		cairo_surface_destroy(r->wallpaper);
		r->wallpaper = NULL;
	}
	FILE *f = fopen(path, "rb");
	if (!f) return;

	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, f);
	jpeg_read_header(&cinfo, TRUE);
	cinfo.out_color_space = JCS_RGB;
	jpeg_start_decompress(&cinfo);

	int w = (int)cinfo.output_width;
	int h = (int)cinfo.output_height;
	int stride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, w);
	unsigned char *data = calloc(1, (size_t)stride * h);
	unsigned char *row  = malloc((size_t)w * 3);

	while ((int)cinfo.output_scanline < h) {
		JSAMPROW rp = row;
		jpeg_read_scanlines(&cinfo, &rp, 1);
		int y = (int)cinfo.output_scanline - 1;
		uint32_t *px = (uint32_t *)(data + y * stride);
		for (int x = 0; x < w; x++)
			px[x] = 0xFF000000u |
				((uint32_t)row[x * 3]     << 16) |
				((uint32_t)row[x * 3 + 1] << 8)  |
				 (uint32_t)row[x * 3 + 2];
	}
	free(row);
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(f);

	/* If image is portrait (taller than wide), rotate 90° CW for landscape */
	if (h > w) {
		int rw = h, rh = w;
		int rstride = cairo_format_stride_for_width(CAIRO_FORMAT_ARGB32, rw);
		unsigned char *rdata = calloc(1, (size_t)rstride * rh);
		for (int sy = 0; sy < h; sy++) {
			uint32_t *src_row = (uint32_t *)(data + sy * stride);
			for (int sx = 0; sx < w; sx++) {
				/* (sx, sy) → (h-1-sy, sx) rotated 90° CW */
				int dx = h - 1 - sy;
				int dy = sx;
				uint32_t *dst_row = (uint32_t *)(rdata + dy * rstride);
				dst_row[dx] = src_row[sx];
			}
		}
		free(data);
		data = rdata;
		w = rw;
		h = rh;
		stride = rstride;
	}

	r->wallpaper = cairo_image_surface_create_for_data(
		data, CAIRO_FORMAT_ARGB32, w, h, stride);

	/* Tie pixel buffer lifetime to the surface */
	static const cairo_user_data_key_t key;
	cairo_surface_set_user_data(r->wallpaper, &key, data, free);
}

/* ── Greeting (no-wallpaper desktop) ──
 *
 * "Hi, <Name>" centred big and quiet, in the spirit of Claude / ChatGPT's
 * empty-conversation state. We avoid loading a real-name PNG portrait or
 * accent colour: this is a *content-free desktop* — the greeting should
 * read, not perform.
 *
 * Name resolution priority: $USER → getlogin() → getpwuid()->pw_gecos
 * (full name; first comma-delimited field) → getpwuid()->pw_name. We
 * Title-Case it because login names are usually lowercase and "hi, xan"
 * looks like a chat sender, not a salutation. */
static void draw_greeting(struct marshal_renderer *r, int y, int h) {
	cairo_t *cr = r->cr;

	/* Resolve display name once per frame — cheap (env + libc).
	 *
	 * Priority:
	 *   1. ~/.marshal/greeting-name  ← user-set override (single line)
	 *   2. GECOS full-name field      ← real name from /etc/passwd
	 *   3. $USER                      ← login name
	 *   4. getlogin()                 ← fallback
	 *   5. pw_name                    ← last resort
	 *   6. "there"                    ← extreme fallback
	 *
	 * The override file is intentional: changing GECOS via chfn requires
	 * PAM and a password prompt — overkill for "what name shows on the
	 * desktop." A plain text file the user owns is the right ergonomics. */
	char raw[128] = {0};
	const char *home = getenv("HOME");
	if (home) {
		char path[256];
		snprintf(path, sizeof(path), "%s/.marshal/greeting-name", home);
		FILE *f = fopen(path, "r");
		if (f) {
			if (fgets(raw, sizeof(raw), f)) {
				size_t len = strlen(raw);
				while (len && (raw[len-1] == '\n' ||
						raw[len-1] == '\r' ||
						raw[len-1] == ' '))
					raw[--len] = '\0';
			}
			fclose(f);
		}
	}
	if (!raw[0]) {
		struct passwd *pw = getpwuid(getuid());
		if (pw && pw->pw_gecos && pw->pw_gecos[0] &&
				pw->pw_gecos[0] != ',') {
			size_t n = 0;
			const char *p = pw->pw_gecos;
			while (*p && *p != ',' && n < sizeof(raw) - 1)
				raw[n++] = *p++;
			raw[n] = '\0';
		}
	}
	if (!raw[0]) {
		const char *envu = getenv("USER");
		if (envu && envu[0])
			snprintf(raw, sizeof(raw), "%s", envu);
	}
	if (!raw[0]) {
		const char *lg = getlogin();
		if (lg && lg[0])
			snprintf(raw, sizeof(raw), "%s", lg);
	}
	if (!raw[0]) {
		struct passwd *pw = getpwuid(getuid());
		if (pw && pw->pw_name)
			snprintf(raw, sizeof(raw), "%s", pw->pw_name);
	}
	if (!raw[0]) snprintf(raw, sizeof(raw), "there");

	/* Title-case the first letter, leave the rest as-is. Login names are
	 * usually lowercase ("xan" → "Xan"), real names from GECOS already
	 * capitalised. */
	if (raw[0] >= 'a' && raw[0] <= 'z') raw[0] = raw[0] - ('a' - 'A');

	char greeting[160];
	/* Period reads warmer / quieter than "!", which feels presentational. */
	snprintf(greeting, sizeof(greeting), "Hi, %s.", raw);

	/* Pick a font size proportional to the canvas. */
	int px = h / 13;
	if (px < 38) px = 38;
	if (px > 110) px = 110;

	/* Editorial serif — Source Serif 4 Display, italic, light. Geist at
	 * this size reads like a slide title; a serif italic reads like a
	 * page header (Claude.ai, Medium, NYT). Source Serif 4 ships an
	 * optical-size "Display" cut tuned for headlines. We list real
	 * fallbacks so machines without Adobe Source Serif still render
	 * sanely — every fontconfig setup has a "serif" default. */
	PangoFontDescription *fd = pango_font_description_from_string(
		"Source Serif 4 Display, Source Serif 4, Source Serif Pro, "
		"Iowan Old Style, Charter, Georgia, serif");
	pango_font_description_set_style(fd, PANGO_STYLE_ITALIC);
	pango_font_description_set_weight(fd, PANGO_WEIGHT_LIGHT);
	pango_font_description_set_absolute_size(fd, px * PANGO_SCALE);

	/* Slight negative tracking so the display-size serif feels tight
	 * rather than airy. */
	PangoLayout *layout = create_layout(cr, fd, -10);
	pango_layout_set_text(layout, greeting, -1);
	int tw, th;
	pango_layout_get_pixel_size(layout, &tw, &th);

	/* Centre vertically with a 6% upward lift so the input bar's
	 * visual weight at the bottom doesn't pull the optical centre. */
	int cx_x = (r->width - tw) / 2;
	int cy_y = y + (h - th) / 2 - h * 6 / 100;

	cairo_move_to(cr, cx_x, cy_y);
	/* Lower contrast than before — softer, less "title-card" feel. */
	cairo_set_source_rgba(cr, 0.13, 0.13, 0.15, 0.85);
	pango_cairo_show_layout(cr, layout);
	g_object_unref(layout);

	/* Subtitle in upright Geist for a clear typographic hierarchy
	 * (italic serif headline + upright sans subhead = editorial pair).
	 * Smaller and quieter so the eye flows past it. */
	int sub_px = px * 2 / 5;
	if (sub_px < 17) sub_px = 17;
	PangoFontDescription *sub_fd = pango_font_description_copy(
		r->font_empty_sub);
	pango_font_description_set_weight(sub_fd, PANGO_WEIGHT_NORMAL);
	pango_font_description_set_absolute_size(sub_fd,
		sub_px * PANGO_SCALE);

	PangoLayout *sub = create_layout(cr, sub_fd, 20);
	pango_layout_set_text(sub, "What would you like to do?", -1);
	int sw, sh;
	pango_layout_get_pixel_size(sub, &sw, &sh);
	cairo_move_to(cr, (r->width - sw) / 2, cy_y + th + sub_px / 2);
	cairo_set_source_rgba(cr, 0.45, 0.45, 0.50, 0.78);
	pango_cairo_show_layout(cr, sub);
	g_object_unref(sub);

	pango_font_description_free(sub_fd);
	pango_font_description_free(fd);
}

/* Draw wallpaper in "cover" mode within [0, y .. y+h] */
static void draw_wallpaper(struct marshal_renderer *r, int y, int h) {
	if (!r->wallpaper) return;
	cairo_t *cr = r->cr;
	int iw = cairo_image_surface_get_width(r->wallpaper);
	int ih = cairo_image_surface_get_height(r->wallpaper);

	double sx = (double)r->width / iw;
	double sy = (double)h / ih;
	double sc = sx > sy ? sx : sy;           /* cover: use larger */
	double ox = (r->width - iw * sc) / 2.0;
	double oy = y + (h - ih * sc) / 2.0;

	cairo_save(cr);
	cairo_rectangle(cr, 0, y, r->width, h);
	cairo_clip(cr);
	cairo_translate(cr, ox, oy);
	cairo_scale(cr, sc, sc);
	cairo_set_source_surface(cr, r->wallpaper, 0, 0);
	cairo_paint(cr);
	cairo_restore(cr);
}

/* ── History-toggle icon (3-line list) ── */

static void draw_history_icon(cairo_t *cr, int cx, int cy, bool active) {
	struct color c = active ? ACCENT_BLUE : TEXT_SECONDARY;
	set_color(cr, c);

	int lw = 12;       /* line width */
	int gap = 5;       /* vertical gap between lines */
	int lx = cx - lw / 2;
	int top = cy - gap;

	for (int i = 0; i < 3; i++) {
		int y = top + i * gap;
		/* Bullet */
		cairo_arc(cr, lx - 3, y, 1.3, 0, 2 * M_PI);
		cairo_fill(cr);
		/* Line */
		rounded_rect(cr, lx, y - 0.8, lw, 1.6, 0.8);
		cairo_fill(cr);
	}
}

static void draw_empty_state(struct marshal_renderer *r, int top, int bottom) {
	cairo_t *cr = r->cr;
	int center_x = r->width / 2;

	PangoLayout *heading = create_layout(cr, r->font_empty_heading,
		TRACKING_EMPTY_H);
	pango_layout_set_text(heading, "Marshal", -1);
	int hw, hh;
	pango_layout_get_pixel_size(heading, &hw, &hh);

	PangoLayout *sub = create_layout(cr, r->font_empty_sub, 0);
	pango_layout_set_text(sub, "Intent-native computing", -1);
	int sw, sh;
	pango_layout_get_pixel_size(sub, &sw, &sh);

	int total_h = 32 + SPACE_L + hh + SPACE_S + sh;
	int start_y = top + (bottom - top - total_h) / 2;

	draw_leaf_glyph(cr, center_x, start_y + 16);

	int heading_y = start_y + 32 + SPACE_L;
	cairo_move_to(cr, center_x - hw / 2, heading_y);
	set_color(cr, TEXT_PRIMARY);
	pango_cairo_show_layout(cr, heading);

	int sub_y = heading_y + hh + SPACE_S;
	cairo_move_to(cr, center_x - sw / 2, sub_y);
	set_color(cr, TEXT_TERTIARY);
	pango_cairo_show_layout(cr, sub);

	g_object_unref(heading);
	g_object_unref(sub);
}

/* ── Status banner ── */

static void draw_status_banner(struct marshal_renderer *r) {
	cairo_t *cr = r->cr;

	cairo_set_source_rgba(cr, 0xD9 / 255.0, 0x77 / 255.0,
		0x06 / 255.0, 0.10);
	cairo_rectangle(cr, 0, 0, r->width, STATUS_BANNER_H);
	cairo_fill(cr);

	cairo_set_source_rgba(cr, 0xD9 / 255.0, 0x77 / 255.0,
		0x06 / 255.0, 0.25);
	cairo_rectangle(cr, 0, STATUS_BANNER_H - 1, r->width, 1);
	cairo_fill(cr);

	PangoLayout *layout = create_layout(cr, r->font_status, 0);
	pango_layout_set_text(layout,
		"\xe2\x9a\xa0  Inference server offline  \xc2\xb7  "
		"bash scripts/start-inference.sh", -1);
	int tw, th;
	pango_layout_get_pixel_size(layout, &tw, &th);
	cairo_move_to(cr, CARD_MARGIN_H,
		(STATUS_BANNER_H - th) / 2);
	set_color(cr, ACCENT_AMBER);
	pango_cairo_show_layout(cr, layout);
	g_object_unref(layout);
}

/* ── Taskbar (input bar + OS icon + running apps) ── */

/* ── Status bar icons (Cairo-drawn, no icon font dependency) ── */

static void draw_battery_icon(cairo_t *cr, int x, int cy, int pct,
		bool charging) {
	int bw = 18, bh = 10;
	int by = cy - bh / 2;
	rounded_rect(cr, x, by, bw, bh, 2);
	set_color(cr, TEXT_SECONDARY);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);
	set_color(cr, TEXT_SECONDARY);
	cairo_rectangle(cr, x + bw, cy - 2, 2, 4);
	cairo_fill(cr);
	int fill_w = (bw - 4) * pct / 100;
	if (fill_w > 0) {
		struct color fc;
		if (charging)        fc = ACCENT_BLUE;
		else if (pct <= 20)  fc = ACCENT_RED;
		else if (pct <= 40)  fc = ACCENT_AMBER;
		else                 fc = ACCENT_GREEN;
		set_color(cr, fc);
		cairo_rectangle(cr, x + 2, by + 2, fill_w, bh - 4);
		cairo_fill(cr);
	}
}

static void draw_wifi_icon(cairo_t *cr, double cx, double cy,
		bool connected, int signal) {
	struct color c = connected ? TEXT_SECONDARY : TEXT_TERTIARY;
	set_color(cr, c);
	cairo_arc(cr, cx, cy + 3, 1.5, 0, 2 * M_PI);
	cairo_fill(cr);
	for (int i = 0; i < 3; i++) {
		double r = 4.0 + i * 3.0;
		bool active = connected && signal >= (i + 1) * 25;
		double a = active ? (c.a / 255.0) : 0.12;
		cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
			c.b / 255.0, a);
		cairo_set_line_width(cr, 1.5);
		cairo_arc(cr, cx, cy + 3, r, -M_PI * 0.75, -M_PI * 0.25);
		cairo_stroke(cr);
	}
}

static void draw_bt_icon(cairo_t *cr, double cx, double cy, bool enabled) {
	struct color c = enabled ? ACCENT_BLUE : TEXT_TERTIARY;
	set_color(cr, c);
	cairo_set_line_width(cr, 1.3);
	double h = 12, hw = 4;
	double top = cy - h / 2, bot = cy + h / 2;
	cairo_move_to(cr, cx, top);
	cairo_line_to(cr, cx, bot);
	cairo_stroke(cr);
	cairo_move_to(cr, cx, top);
	cairo_line_to(cr, cx + hw, cy - h / 6);
	cairo_line_to(cr, cx - hw, cy + h / 6);
	cairo_stroke(cr);
	cairo_move_to(cr, cx, bot);
	cairo_line_to(cr, cx + hw, cy + h / 6);
	cairo_line_to(cr, cx - hw, cy - h / 6);
	cairo_stroke(cr);
}

static void draw_volume_icon(cairo_t *cr, int x, int cy, int pct, bool muted) {
	struct color c = muted ? TEXT_TERTIARY : TEXT_SECONDARY;
	set_color(cr, c);
	cairo_set_line_width(cr, 1.3);

	/* Speaker body: small rectangle */
	int bw = 5, bh = 6;
	int by = cy - bh / 2;
	cairo_rectangle(cr, x, by, bw, bh);
	cairo_fill(cr);

	/* Speaker cone: triangle */
	cairo_move_to(cr, x + bw, by - 2);
	cairo_line_to(cr, x + bw + 5, by - 5);
	cairo_line_to(cr, x + bw + 5, by + bh + 5);
	cairo_line_to(cr, x + bw, by + bh + 2);
	cairo_close_path(cr);
	cairo_fill(cr);

	if (muted) {
		/* X mark */
		set_color(cr, ACCENT_RED);
		cairo_set_line_width(cr, 1.5);
		int mx = x + bw + 8;
		cairo_move_to(cr, mx, cy - 3);
		cairo_line_to(cr, mx + 6, cy + 3);
		cairo_stroke(cr);
		cairo_move_to(cr, mx + 6, cy - 3);
		cairo_line_to(cr, mx, cy + 3);
		cairo_stroke(cr);
	} else {
		/* Sound waves — 1 to 3 arcs depending on volume */
		int waves = pct < 33 ? 1 : pct < 66 ? 2 : 3;
		double wx = x + bw + 7;
		for (int i = 0; i < waves; i++) {
			double r = 3.0 + i * 3.0;
			cairo_arc(cr, wx, cy, r, -M_PI * 0.35, M_PI * 0.35);
			cairo_stroke(cr);
		}
	}
}

/* ────────────────────────────────────────────────────────────────────────
 * App icon resolution + B/W conversion
 *
 * The bottom-bar app indicators are real freedesktop icons desaturated to
 * B/W. Pipeline:
 *
 *   1. app_id ("org.mozilla.firefox", "google-chrome", "marshal-terminal")
 *      → icon name. We try the .desktop file's `Icon=` field first, then
 *      fall back to the app_id itself, with a cheap reverse-DNS strip
 *      (last "."-component) so app_ids like "org.mozilla.firefox" still
 *      hit "firefox".
 *
 *   2. icon name → file path. Search:
 *        $XDG_DATA_DIRS/icons/hicolor/{scalable,128x128,96x96,64x64,48x48}/apps/<name>.{svg,png}
 *        /usr/share/pixmaps/<name>.{png,svg,xpm}
 *      Larger sizes preferred so downscaling stays clean. SVG only when
 *      librsvg was found at build time.
 *
 *   3. file → cairo_image_surface (ARGB32). PNG: cairo native loader.
 *      SVG: rsvg_handle_render_document into a fresh ARGB32 surface.
 *
 *   4. Desaturate. Per-pixel: luma = 0.299R + 0.587G + 0.114B; rewrite
 *      RGB to (luma, luma, luma), keep alpha. Cairo uses pre-multiplied
 *      ARGB32 so we un-premul before the math and re-premul after.
 *
 *   5. Cache by icon name on the renderer. FIFO eviction at 32 entries.
 *
 * Result: launching Firefox shows the actual Firefox flame in monochrome.
 * Only when icon resolution genuinely fails does the renderer fall back
 * to the monogram circle. */

#ifdef MARSHAL_HAS_RSVG
#include <librsvg/rsvg.h>
#endif

static void desaturate_argb32(cairo_surface_t *surface) {
	if (!surface) return;
	if (cairo_image_surface_get_format(surface) != CAIRO_FORMAT_ARGB32) return;

	cairo_surface_flush(surface);
	int w = cairo_image_surface_get_width(surface);
	int h = cairo_image_surface_get_height(surface);
	int stride = cairo_image_surface_get_stride(surface);
	unsigned char *data = cairo_image_surface_get_data(surface);
	if (!data) return;

	for (int y = 0; y < h; y++) {
		uint32_t *row = (uint32_t *)(data + y * stride);
		for (int x = 0; x < w; x++) {
			uint32_t px = row[x];
			uint8_t a = (px >> 24) & 0xff;
			if (a == 0) { continue; }
			uint8_t r_ = (px >> 16) & 0xff;
			uint8_t g_ = (px >> 8)  & 0xff;
			uint8_t b_ = (px)       & 0xff;
			/* Un-premultiply, compute luma, re-premultiply. */
			if (a < 0xff) {
				r_ = (uint8_t)((r_ * 255 + a / 2) / a);
				g_ = (uint8_t)((g_ * 255 + a / 2) / a);
				b_ = (uint8_t)((b_ * 255 + a / 2) / a);
			}
			uint32_t luma = (299u * r_ + 587u * g_ + 114u * b_ + 500u)
				/ 1000u;
			if (luma > 255) luma = 255;
			uint8_t l8 = (uint8_t)luma;
			uint8_t lr = (uint8_t)((l8 * a + 127) / 255);
			row[x] = ((uint32_t)a  << 24) |
				 ((uint32_t)lr << 16) |
				 ((uint32_t)lr <<  8) |
				 ((uint32_t)lr);
		}
	}
	cairo_surface_mark_dirty(surface);
}

static char *read_icon_field_from_desktop(const char *path) {
	FILE *f = fopen(path, "r");
	if (!f) return NULL;
	char line[1024];
	bool in_main = false;
	while (fgets(line, sizeof(line), f)) {
		size_t n = strlen(line);
		while (n && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = '\0';
		if (line[0] == '[') {
			in_main = (strcmp(line, "[Desktop Entry]") == 0);
			continue;
		}
		if (!in_main) continue;
		if (strncmp(line, "Icon=", 5) == 0 && line[5]) {
			char *out = strdup(line + 5);
			fclose(f);
			return out;
		}
	}
	fclose(f);
	return NULL;
}

/* Walk XDG_DATA_DIRS looking for `<dir>/applications/<app_id>.desktop`.
 * Returns malloc'd icon name (caller frees), or NULL. */
static char *resolve_icon_name_via_desktop(const char *app_id) {
	if (!app_id || !app_id[0]) return NULL;

	const char *xdg = getenv("XDG_DATA_DIRS");
	if (!xdg || !xdg[0]) xdg = "/usr/local/share:/usr/share";

	const char *home = getenv("HOME");
	char user_data[512] = {0};
	if (home) snprintf(user_data, sizeof(user_data),
		"%s/.local/share", home);

	/* Build candidate list: ~/.local/share first, then XDG_DATA_DIRS. */
	const char *roots[8];
	int nroot = 0;
	if (user_data[0]) roots[nroot++] = user_data;

	char xdg_buf[1024];
	snprintf(xdg_buf, sizeof(xdg_buf), "%s", xdg);
	char *save = NULL;
	for (char *tok = strtok_r(xdg_buf, ":", &save);
			tok && nroot < 8;
			tok = strtok_r(NULL, ":", &save)) {
		roots[nroot++] = tok;
	}

	char path[1024];
	for (int i = 0; i < nroot; i++) {
		snprintf(path, sizeof(path), "%s/applications/%s.desktop",
			roots[i], app_id);
		char *icon = read_icon_field_from_desktop(path);
		if (icon) return icon;

		/* Also try lower-cased app_id — wlroots gives raw class names
		 * which can be mixed-case (e.g. "Firefox"). */
		char lower[128];
		size_t alen = strlen(app_id);
		if (alen >= sizeof(lower)) continue;
		for (size_t j = 0; j < alen; j++)
			lower[j] = (app_id[j] >= 'A' && app_id[j] <= 'Z')
				? app_id[j] + 32 : app_id[j];
		lower[alen] = '\0';
		snprintf(path, sizeof(path), "%s/applications/%s.desktop",
			roots[i], lower);
		icon = read_icon_field_from_desktop(path);
		if (icon) return icon;
	}
	return NULL;
}

/* Find an actual icon file on disk for a given icon name. Returns malloc'd
 * absolute path (caller frees), or NULL. Tries scalable SVG first (only
 * if librsvg was linked), then large-then-small PNGs. */
static char *find_icon_file(const char *icon_name) {
	if (!icon_name || !icon_name[0]) return NULL;

	/* If the name is already an absolute path (some .desktop files use
	 * an absolute Icon= line), use it directly. */
	if (icon_name[0] == '/') {
		if (access(icon_name, R_OK) == 0) return strdup(icon_name);
	}

	const char *xdg = getenv("XDG_DATA_DIRS");
	if (!xdg || !xdg[0]) xdg = "/usr/local/share:/usr/share";
	const char *home = getenv("HOME");

	const char *icon_roots[8];
	int nroot = 0;
	char user_icons[512] = {0};
	if (home) {
		snprintf(user_icons, sizeof(user_icons), "%s/.local/share",
			home);
		icon_roots[nroot++] = user_icons;
	}
	char xdg_buf[1024];
	snprintf(xdg_buf, sizeof(xdg_buf), "%s", xdg);
	char *save = NULL;
	for (char *tok = strtok_r(xdg_buf, ":", &save);
			tok && nroot < 8;
			tok = strtok_r(NULL, ":", &save)) {
		icon_roots[nroot++] = tok;
	}

	const char *sizes[] = {
#ifdef MARSHAL_HAS_RSVG
		"scalable",
#endif
		"512x512", "256x256", "192x192", "128x128", "96x96",
		"64x64", "48x48", "32x32", "24x24", "16x16",
	};
	const char *exts_svg[] = { "svg" };
	const char *exts_png[] = { "png" };

	char path[1024];
	for (int i = 0; i < nroot; i++) {
		for (size_t s = 0; s < sizeof(sizes)/sizeof(sizes[0]); s++) {
			const char **exts = (strcmp(sizes[s], "scalable") == 0)
				? exts_svg : exts_png;
			int n_exts = (strcmp(sizes[s], "scalable") == 0) ? 1 : 1;
			for (int e = 0; e < n_exts; e++) {
				snprintf(path, sizeof(path),
					"%s/icons/hicolor/%s/apps/%s.%s",
					icon_roots[i], sizes[s], icon_name,
					exts[e]);
				if (access(path, R_OK) == 0) return strdup(path);
			}
		}
		/* Also try /usr/share/pixmaps/<name>.{png,svg} */
		const char *pm_exts[] = {
			"png",
#ifdef MARSHAL_HAS_RSVG
			"svg",
#endif
			"xpm",
		};
		for (size_t e = 0; e < sizeof(pm_exts)/sizeof(pm_exts[0]); e++) {
			snprintf(path, sizeof(path), "%s/pixmaps/%s.%s",
				icon_roots[i], icon_name, pm_exts[e]);
			if (access(path, R_OK) == 0) return strdup(path);
		}
	}
	return NULL;
}

#define ICON_RENDER_PX 96  /* pre-rasterised cache size; cairo scales when drawn */

static cairo_surface_t *load_icon_surface(const char *path) {
	if (!path) return NULL;
	const char *dot = strrchr(path, '.');
	if (!dot) return NULL;

	if (strcasecmp(dot, ".png") == 0) {
		cairo_surface_t *s = cairo_image_surface_create_from_png(path);
		if (!s || cairo_surface_status(s) != CAIRO_STATUS_SUCCESS) {
			if (s) cairo_surface_destroy(s);
			return NULL;
		}
		return s;
	}
#ifdef MARSHAL_HAS_RSVG
	if (strcasecmp(dot, ".svg") == 0) {
		GError *err = NULL;
		RsvgHandle *handle = rsvg_handle_new_from_file(path, &err);
		if (!handle) {
			if (err) g_error_free(err);
			return NULL;
		}
		cairo_surface_t *s = cairo_image_surface_create(
			CAIRO_FORMAT_ARGB32, ICON_RENDER_PX, ICON_RENDER_PX);
		cairo_t *cr = cairo_create(s);
		RsvgRectangle vp = { 0, 0, ICON_RENDER_PX, ICON_RENDER_PX };
		gboolean ok = rsvg_handle_render_document(handle, cr, &vp, &err);
		cairo_destroy(cr);
		g_object_unref(handle);
		if (err) g_error_free(err);
		if (!ok) {
			cairo_surface_destroy(s);
			return NULL;
		}
		return s;
	}
#endif
	return NULL;
}

/* Look up `key` in the renderer's cache. The key is whatever the caller
 * has — usually app_id like "org.mozilla.firefox" or "Firefox". On miss
 * we try in order:
 *   1. find an icon file under hicolor/pixmaps for the literal key,
 *   2. read `.desktop` for that key, follow Icon= field,
 *   3. strip reverse-DNS prefix (last "." component) and try again.
 * First hit wins, gets desaturated and cached.
 *
 * Returns NULL if no icon could be found — caller falls back to monogram. */
static cairo_surface_t *icon_cache_get(struct marshal_renderer *r,
		const char *key) {
	if (!key || !key[0]) return NULL;
	for (int i = 0; i < r->icon_cache_count; i++) {
		if (strcmp(r->icon_cache[i].name, key) == 0)
			return r->icon_cache[i].surface;
	}

	/* 1. literal key as icon name */
	char *path = find_icon_file(key);

	/* 2. .desktop's Icon= field */
	if (!path) {
		char *via_desktop = resolve_icon_name_via_desktop(key);
		if (via_desktop) {
			path = find_icon_file(via_desktop);
			free(via_desktop);
		}
	}

	/* 3. last reverse-DNS component */
	if (!path) {
		const char *dot = strrchr(key, '.');
		if (dot && dot[1]) {
			path = find_icon_file(dot + 1);
			if (!path) {
				char *via = resolve_icon_name_via_desktop(dot + 1);
				if (via) {
					path = find_icon_file(via);
					free(via);
				}
			}
		}
	}

	cairo_surface_t *s = load_icon_surface(path);
	free(path);
	if (s) desaturate_argb32(s);

	/* Insert (with FIFO eviction). Even on miss we cache NULL so we
	 * don't re-stat the filesystem on every frame. */
	int idx;
	if (r->icon_cache_count < MARSHAL_ICON_CACHE_MAX) {
		idx = r->icon_cache_count++;
	} else {
		idx = 0;
		if (r->icon_cache[0].surface)
			cairo_surface_destroy(r->icon_cache[0].surface);
		memmove(&r->icon_cache[0], &r->icon_cache[1],
			(MARSHAL_ICON_CACHE_MAX - 1) * sizeof(r->icon_cache[0]));
		idx = MARSHAL_ICON_CACHE_MAX - 1;
	}
	snprintf(r->icon_cache[idx].name, sizeof(r->icon_cache[idx].name),
		"%s", key);
	r->icon_cache[idx].surface = s;
	return s;
}

static void draw_taskbar(struct marshal_renderer *r,
		struct marshal_input *input, struct marshal_feed *feed) {
	cairo_t *cr = r->cr;
	int bar_y = r->height - INPUT_HEIGHT;

	/* Background */
	set_color(cr, BG_INPUT);
	cairo_rectangle(cr, 0, bar_y, r->width, INPUT_HEIGHT);
	cairo_fill(cr);

	/* Top border */
	set_color(cr, BORDER_CARD);
	cairo_rectangle(cr, 0, bar_y, r->width, 1);
	cairo_fill(cr);

	/* ── Left zone: OS icon ── */
	int icon_cx = TASKBAR_ICON_W / 2;
	int icon_cy = bar_y + INPUT_HEIGHT / 2;
	draw_logo_glyph(cr, icon_cx, icon_cy);

	/* Separator after icon */
	set_color(cr, BORDER_SEPARATOR);
	cairo_rectangle(cr, TASKBAR_ICON_W, bar_y + 10, 1,
		INPUT_HEIGHT - 20);
	cairo_fill(cr);

	/* ── Center zone: input field ── */
	int input_x = TASKBAR_ICON_W + INPUT_PADDING_L;
	int input_max_x = r->input_field_right_x > 0
		? r->input_field_right_x : (r->width - TASKBAR_APPS_W);

	PangoLayout *layout = create_layout(cr, r->font_input, 0);
	pango_layout_set_width(layout,
		(input_max_x - input_x - SPACE_L) * PANGO_SCALE);
	pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
	int lw, lh;

	if (input->len == 0) {
		pango_layout_set_text(layout,
			"What should we do? Try \"help\"", -1);
		pango_layout_get_pixel_size(layout, &lw, &lh);
		cairo_move_to(cr, input_x, bar_y + (INPUT_HEIGHT - lh) / 2);
		set_color(cr, TEXT_PLACEHOLDER);
		pango_cairo_show_layout(cr, layout);
	} else {
		pango_layout_set_text(layout, input->buf, input->len);
		pango_layout_get_pixel_size(layout, &lw, &lh);
		int text_y = bar_y + (INPUT_HEIGHT - lh) / 2;

		/* Selection highlight — drawn before text so text renders on top */
		if (input->sel_anchor != -1 &&
				input->sel_anchor != input->sel_focus) {
			int sel_s = input->sel_anchor < input->sel_focus
				? input->sel_anchor : input->sel_focus;
			int sel_e = input->sel_anchor < input->sel_focus
				? input->sel_focus : input->sel_anchor;

			PangoRectangle sr, er;
			pango_layout_get_cursor_pos(layout, sel_s, &sr, NULL);
			pango_layout_get_cursor_pos(layout, sel_e, &er, NULL);

			int hx  = input_x + sr.x / PANGO_SCALE;
			int hx2 = input_x + er.x / PANGO_SCALE;
			cairo_set_source_rgba(cr,
				ACCENT_BLUE.r / 255.0, ACCENT_BLUE.g / 255.0,
				ACCENT_BLUE.b / 255.0, 0.18);
			cairo_rectangle(cr, hx, text_y - 1, hx2 - hx, lh + 2);
			cairo_fill(cr);
		}

		cairo_move_to(cr, input_x, text_y);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, layout);
	}

	/* Cursor */
	if (input->cursor_visible) {
		PangoLayout *measure = create_layout(cr, r->font_input, 0);
		pango_layout_set_text(measure, input->buf, input->cursor_pos);
		int cursor_w, cursor_h;
		pango_layout_get_pixel_size(measure, &cursor_w, &cursor_h);
		g_object_unref(measure);

		int cx = input_x + cursor_w;
		int cy = bar_y + (INPUT_HEIGHT - 18) / 2;
		set_color(cr, ACCENT_BLUE);
		cairo_rectangle(cr, cx, cy, 1.5, 18);
		cairo_fill(cr);
	}

	if (input->len > 0) {
		PangoLayout *hint = create_layout(cr, r->font_keyboard_hint,
			TRACKING_CAPTION2);
		pango_layout_set_text(hint, "\xe2\x86\xb5 return", -1);
		int hw, hh;
		pango_layout_get_pixel_size(hint, &hw, &hh);
		int hint_x = input_max_x - SPACE_L - hw;
		if (hint_x > input_x + lw + SPACE_M) {
			cairo_move_to(cr, hint_x,
				bar_y + (INPUT_HEIGHT - hh) / 2);
			set_color(cr, TEXT_PLACEHOLDER);
			pango_cairo_show_layout(cr, hint);
		}
		g_object_unref(hint);
	}

	g_object_unref(layout);

	/* ── Right zone: history icon + status indicators ── */
	struct marshal_status *st = r->status;
	int bar_cy = bar_y + INPUT_HEIGHT / 2;

	/* Build the right zone right-to-left */
	int rx = r->width - STATUS_PAD_H;

	/* Battery: icon + percentage */
	if (st && st->battery_pct >= 0) {
		char pstr[8];
		snprintf(pstr, sizeof(pstr), "%d%%", st->battery_pct);
		PangoLayout *bpl = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(bpl, pstr, -1);
		int bpw, bph;
		pango_layout_get_pixel_size(bpl, &bpw, &bph);
		rx -= bpw;
		cairo_move_to(cr, rx, bar_cy - bph / 2);
		set_color(cr, TEXT_SECONDARY);
		pango_cairo_show_layout(cr, bpl);
		g_object_unref(bpl);
		rx -= 22 + SPACE_XS;
		draw_battery_icon(cr, rx, bar_cy,
			st->battery_pct, st->battery_charging);
		rx -= SPACE_S;
	}

	/* Volume icon */
	if (st && st->volume_pct >= 0) {
		char vstr[8];
		snprintf(vstr, sizeof(vstr), "%d%%", st->volume_muted ? 0 : st->volume_pct);
		PangoLayout *vpl = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(vpl, vstr, -1);
		int vpw, vph;
		pango_layout_get_pixel_size(vpl, &vpw, &vph);
		rx -= vpw;
		cairo_move_to(cr, rx, bar_cy - vph / 2);
		set_color(cr, st->volume_muted ? TEXT_TERTIARY : TEXT_SECONDARY);
		pango_cairo_show_layout(cr, vpl);
		g_object_unref(vpl);
		rx -= 20 + SPACE_XS;
		draw_volume_icon(cr, rx, bar_cy, st->volume_pct, st->volume_muted);
		rx -= SPACE_S;
	}

	/* Bluetooth icon */
	if (st && st->bt_available) {
		rx -= 8;
		draw_bt_icon(cr, rx, bar_cy, st->bt_enabled);
		rx -= 8 + SPACE_S;
	}

	/* WiFi icon */
	rx -= 8;
	draw_wifi_icon(cr, rx, bar_cy,
		st ? st->wifi_connected : false,
		st ? st->wifi_signal : 0);
	rx -= 8 + SPACE_S;

	/* Keyboard layout */
	if (st) {
		PangoLayout *kbl = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(kbl, st->kb_layout, -1);
		int kw2, kh2;
		pango_layout_get_pixel_size(kbl, &kw2, &kh2);
		rx -= kw2;
		cairo_move_to(cr, rx, bar_cy - kh2 / 2);
		set_color(cr, TEXT_SECONDARY);
		pango_cairo_show_layout(cr, kbl);
		g_object_unref(kbl);
		rx -= SPACE_S;
	}

	/* Time */
	if (st) {
		PangoLayout *tl = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(tl, st->time_str, -1);
		int tw2, th2;
		pango_layout_get_pixel_size(tl, &tw2, &th2);
		rx -= tw2;
		cairo_move_to(cr, rx, bar_cy - th2 / 2);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, tl);
		g_object_unref(tl);
		rx -= SPACE_S;
	}

	/* Store where the status zone starts */
	r->status_zone_left_x = rx;

	/* Separator */
	set_color(cr, BORDER_SEPARATOR);
	cairo_rectangle(cr, rx, bar_y + 10, 1, INPUT_HEIGHT - 20);
	cairo_fill(cr);
	rx -= SPACE_XS;

	/* History toggle icon */
	bool hist_on = st ? st->history_open : false;
	int hist_cx = rx - HISTORY_ICON_W / 2;
	draw_history_icon(cr, hist_cx, bar_cy, hist_on);

	/* Store history icon hit rect */
	r->history_icon_x = rx - HISTORY_ICON_W;
	r->history_icon_y = bar_y;
	r->history_icon_w = HISTORY_ICON_W;
	r->history_icon_h = INPUT_HEIGHT;

	rx -= HISTORY_ICON_W;

	/* ── Running-app icons ──
	 * Walk r->apps[] right-to-left. Each app gets the freedesktop icon
	 * for its app_id, desaturated to B/W and cached. If resolution fails
	 * we fall back to a thin-stroke monogram circle.
	 *
	 * Focused app is rendered at full opacity; background apps are
	 * dimmed to ~55% so the active window reads as primary. */
	int app_size = INPUT_HEIGHT - 22;       /* visual diameter */
	int app_gap  = 10;
	for (int i = r->app_count - 1; i >= 0; i--) {
		int icx = rx - app_size / 2 - SPACE_XS;
		int icy = bar_cy;

		struct marshal_bar_app *app = &r->apps[i];
		cairo_surface_t *icon = icon_cache_get(r, app->icon_name);

		if (icon) {
			int iw = cairo_image_surface_get_width(icon);
			int ih = cairo_image_surface_get_height(icon);
			double sc = (double)app_size / (iw > ih ? iw : ih);
			double draw_x = icx - iw * sc / 2.0;
			double draw_y = icy - ih * sc / 2.0;

			cairo_save(cr);
			cairo_translate(cr, draw_x, draw_y);
			cairo_scale(cr, sc, sc);
			cairo_set_source_surface(cr, icon, 0, 0);
			cairo_paint_with_alpha(cr, app->focused ? 0.97 : 0.55);
			cairo_restore(cr);
		} else {
			/* Fallback: monogram in a thin-stroke circle. */
			double radius = app_size / 2.0;
			if (app->focused) {
				cairo_arc(cr, icx, icy, radius, 0, 2 * M_PI);
				cairo_set_source_rgba(cr, 0.05, 0.05, 0.05, 0.95);
				cairo_fill(cr);
			} else {
				cairo_arc(cr, icx, icy, radius - 0.5, 0, 2 * M_PI);
				cairo_set_source_rgba(cr, 0.20, 0.20, 0.20, 0.85);
				cairo_set_line_width(cr, 1.2);
				cairo_stroke(cr);
			}
			PangoLayout *gl = create_layout(cr, r->font_input, 0);
			pango_layout_set_text(gl,
				app->glyph[0] ? app->glyph : "?", -1);
			int gw, gh;
			pango_layout_get_pixel_size(gl, &gw, &gh);
			cairo_move_to(cr, icx - gw / 2, icy - gh / 2);
			if (app->focused)
				cairo_set_source_rgba(cr, 1, 1, 1, 0.97);
			else
				cairo_set_source_rgba(cr, 0.20, 0.20, 0.20, 0.95);
			pango_cairo_show_layout(cr, gl);
			g_object_unref(gl);
		}

		/* Store hit rect — compositor uses these to focus on click */
		app->hit_x = icx - app_size / 2;
		app->hit_y = bar_y + (INPUT_HEIGHT - app_size) / 2;
		app->hit_w = app_size;
		app->hit_h = app_size;

		rx -= app_size + app_gap;
	}

	r->input_field_right_x = rx;
}

/* (Icon functions moved above draw_taskbar.) */

/* ── Quick-settings dropdown ── */

static void draw_dropdown(struct marshal_renderer *r,
		struct marshal_status *s) {
	if (!s || !s->dropdown_open) return;
	cairo_t *cr = r->cr;

	int dx = r->width - DROPDOWN_W - STATUS_PAD_H;
	if (dx < STATUS_PAD_H) dx = STATUS_PAD_H;
	/* Will be positioned above the taskbar once we know content_h */
	int dy = 0;  /* placeholder — computed below */

	/* Measure content height */
	int rows = 0;
	int row_h = 28;  /* height per info row */

	/* WiFi section: label + SSID */
	rows += 2;
	/* Bluetooth section: label + status */
	if (s->bt_available) rows += 2;
	/* Volume section: label + bar */
	if (s->volume_pct >= 0) rows += 2;
	/* Battery section: label + bar */
	if (s->battery_pct >= 0) rows += 2;

	int content_h = DROPDOWN_PAD + rows * row_h + DROPDOWN_PAD;
	int dw = DROPDOWN_W;
	int dh = content_h;
	/* Position above the taskbar with a 4px gap */
	dy = r->height - INPUT_HEIGHT - dh - 4;

	/* Store geometry for hit-testing */
	r->dropdown_x = dx;
	r->dropdown_y = dy;
	r->dropdown_w = dw;
	r->dropdown_h = dh;

	/* Reset section hit rects — populated below as each section renders */
	r->qs_wifi_x = r->qs_wifi_y = r->qs_wifi_w = r->qs_wifi_h = -1;
	r->qs_bt_x   = r->qs_bt_y   = r->qs_bt_w   = r->qs_bt_h   = -1;
	r->qs_vol_x  = r->qs_vol_y  = r->qs_vol_w  = r->qs_vol_h  = -1;

	/* Shadow */
	cairo_set_source_rgba(cr, 0, 0, 0, 0.08);
	rounded_rect(cr, dx + 2, dy + 2, dw, dh, DROPDOWN_RADIUS);
	cairo_fill(cr);

	/* Background */
	rounded_rect(cr, dx, dy, dw, dh, DROPDOWN_RADIUS);
	set_color(cr, BG_OVERLAY_PANEL);
	cairo_fill(cr);

	/* Border */
	rounded_rect(cr, dx, dy, dw, dh, DROPDOWN_RADIUS);
	set_color(cr, BORDER_OVERLAY);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);

	/* Content */
	int cx = dx + DROPDOWN_PAD;
	int cy = dy + DROPDOWN_PAD;
	int cw = dw - 2 * DROPDOWN_PAD;

	/* ── WiFi section ── */
	{
		PangoLayout *lbl = create_layout(cr, r->font_status, 0);
		pango_layout_set_text(lbl, "Wi-Fi", -1);
		int lw, lh;
		pango_layout_get_pixel_size(lbl, &lw, &lh);
		cairo_move_to(cr, cx, cy + (row_h - lh) / 2);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, lbl);

		/* Status dot — right-aligned */
		double dot_x = cx + cw - DROPDOWN_TOGGLE_R;
		double dot_cy = cy + row_h / 2.0;
		if (s->wifi_connected) {
			set_color(cr, ACCENT_GREEN);
			cairo_arc(cr, dot_x, dot_cy, DROPDOWN_TOGGLE_R, 0, 2*M_PI);
			cairo_fill(cr);
		} else {
			set_color(cr, TEXT_TERTIARY);
			cairo_arc(cr, dot_x, dot_cy, DROPDOWN_TOGGLE_R, 0, 2*M_PI);
			cairo_set_line_width(cr, 1.0);
			cairo_stroke(cr);
		}
		g_object_unref(lbl);
		/* Whole row is the click target — users aim at the label, not
		 * the 5px dot. */
		r->qs_wifi_x = cx;
		r->qs_wifi_y = cy;
		r->qs_wifi_w = cw;
		r->qs_wifi_h = row_h;
		cy += row_h;

		/* SSID / disconnected */
		PangoLayout *ssid = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(ssid,
			s->wifi_connected ? s->wifi_ssid : "Not connected", -1);
		pango_layout_get_pixel_size(ssid, &lw, &lh);
		cairo_move_to(cr, cx, cy + (row_h - lh) / 2);
		set_color(cr, TEXT_SECONDARY);
		pango_cairo_show_layout(cr, ssid);
		g_object_unref(ssid);
		cy += row_h;
	}

	/* ── Bluetooth section ── */
	if (s->bt_available) {
		/* Separator */
		set_color(cr, BORDER_SEPARATOR);
		cairo_rectangle(cr, cx, cy, cw, 1);
		cairo_fill(cr);

		PangoLayout *lbl = create_layout(cr, r->font_status, 0);
		pango_layout_set_text(lbl, "Bluetooth", -1);
		int lw, lh;
		pango_layout_get_pixel_size(lbl, &lw, &lh);
		cairo_move_to(cr, cx, cy + (row_h - lh) / 2);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, lbl);

		double dot_x = cx + cw - DROPDOWN_TOGGLE_R;
		double dot_cy = cy + row_h / 2.0;
		if (s->bt_enabled) {
			set_color(cr, ACCENT_BLUE);
			cairo_arc(cr, dot_x, dot_cy, DROPDOWN_TOGGLE_R, 0, 2*M_PI);
			cairo_fill(cr);
		} else {
			set_color(cr, TEXT_TERTIARY);
			cairo_arc(cr, dot_x, dot_cy, DROPDOWN_TOGGLE_R, 0, 2*M_PI);
			cairo_set_line_width(cr, 1.0);
			cairo_stroke(cr);
		}
		g_object_unref(lbl);
		r->qs_bt_x = cx;
		r->qs_bt_y = cy;
		r->qs_bt_w = cw;
		r->qs_bt_h = row_h;
		cy += row_h;

		PangoLayout *st = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(st,
			s->bt_enabled ? "Enabled" : "Disabled", -1);
		pango_layout_get_pixel_size(st, &lw, &lh);
		cairo_move_to(cr, cx, cy + (row_h - lh) / 2);
		set_color(cr, TEXT_SECONDARY);
		pango_cairo_show_layout(cr, st);
		g_object_unref(st);
		cy += row_h;
	}

	/* ── Volume section ── */
	if (s->volume_pct >= 0) {
		/* Separator */
		set_color(cr, BORDER_SEPARATOR);
		cairo_rectangle(cr, cx, cy, cw, 1);
		cairo_fill(cr);

		PangoLayout *lbl = create_layout(cr, r->font_status, 0);
		char vol_str[32];
		snprintf(vol_str, sizeof(vol_str), "Volume  %d%%",
			s->volume_muted ? 0 : s->volume_pct);
		pango_layout_set_text(lbl, vol_str, -1);
		int lw, lh;
		pango_layout_get_pixel_size(lbl, &lw, &lh);
		cairo_move_to(cr, cx, cy + (row_h - lh) / 2);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, lbl);
		g_object_unref(lbl);
		cy += row_h;

		/* Volume bar */
		int bar_w = cw;
		int bar_h = 6;
		int bar_y = cy + (row_h - bar_h) / 2;
		rounded_rect(cr, cx, bar_y, bar_w, bar_h, 3);
		set_color(cr, FILL_CARD);
		cairo_fill(cr);

		int fill = s->volume_muted ? 0 : bar_w * s->volume_pct / 100;
		if (fill > 0) {
			rounded_rect(cr, cx, bar_y, fill, bar_h, 3);
			set_color(cr, ACCENT_BLUE);
			cairo_fill(cr);
		}

		/* Click target is the entire bar row — a 6px-tall bar would be
		 * impossible to hit. */
		r->qs_vol_x = cx;
		r->qs_vol_y = cy;
		r->qs_vol_w = bar_w;
		r->qs_vol_h = row_h;

		/* Muted / active label */
		PangoLayout *vsl = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(vsl,
			s->volume_muted ? "Muted" : "Active", -1);
		pango_layout_get_pixel_size(vsl, &lw, &lh);
		cairo_move_to(cr, cx + bar_w - lw, bar_y + bar_h + 2);
		set_color(cr, TEXT_TERTIARY);
		pango_cairo_show_layout(cr, vsl);
		g_object_unref(vsl);
		cy += row_h;
	}

	/* ── Battery section ── */
	if (s->battery_pct >= 0) {
		/* Separator */
		set_color(cr, BORDER_SEPARATOR);
		cairo_rectangle(cr, cx, cy, cw, 1);
		cairo_fill(cr);

		PangoLayout *lbl = create_layout(cr, r->font_status, 0);
		char batt_str[32];
		snprintf(batt_str, sizeof(batt_str), "Battery  %d%%",
			s->battery_pct);
		pango_layout_set_text(lbl, batt_str, -1);
		int lw, lh;
		pango_layout_get_pixel_size(lbl, &lw, &lh);
		cairo_move_to(cr, cx, cy + (row_h - lh) / 2);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, lbl);
		g_object_unref(lbl);
		cy += row_h;

		/* Progress bar */
		int bar_w = cw;
		int bar_h = 6;
		int bar_y = cy + (row_h - bar_h) / 2;
		rounded_rect(cr, cx, bar_y, bar_w, bar_h, 3);
		set_color(cr, FILL_CARD);
		cairo_fill(cr);

		int fill = bar_w * s->battery_pct / 100;
		if (fill > 0) {
			struct color fc;
			if (s->battery_charging)      fc = ACCENT_BLUE;
			else if (s->battery_pct <= 20) fc = ACCENT_RED;
			else                           fc = ACCENT_GREEN;
			rounded_rect(cr, cx, bar_y, fill, bar_h, 3);
			set_color(cr, fc);
			cairo_fill(cr);
		}

		/* Charging / discharging label */
		PangoLayout *chl = create_layout(cr, r->font_timing, 0);
		pango_layout_set_text(chl,
			s->battery_charging ? "Charging" : "On battery", -1);
		pango_layout_get_pixel_size(chl, &lw, &lh);
		cairo_move_to(cr, cx + bar_w - lw, bar_y + bar_h + 2);
		set_color(cr, TEXT_TERTIARY);
		pango_cairo_show_layout(cr, chl);
		g_object_unref(chl);
		cy += row_h;
	}
}

/* ── Authorization overlay ── */

static void draw_auth_overlay(struct marshal_renderer *r,
		struct marshal_feed *feed) {
	cairo_t *cr = r->cr;

	if (!feed->awaiting_confirm || feed->confirm_card_idx < 0 ||
			feed->confirm_card_idx >= feed->count)
		return;

	MarshalIntent *intent = &feed->intents[feed->confirm_card_idx];

	/* Full-screen dimming layer: rgba(0,0,0,0.72) */
	set_color(cr, OVERLAY_DIM);
	cairo_rectangle(cr, 0, 0, r->width, r->height);
	cairo_fill(cr);

	/* Panel dimensions */
	int panel_w = r->width - 2 * OVERLAY_MARGIN_H;
	if (panel_w > OVERLAY_MAX_W) panel_w = OVERLAY_MAX_W;
	int panel_x = (r->width - panel_w) / 2;
	int content_w = panel_w - 2 * OVERLAY_PADDING;

	/* Measure content height to center panel */
	int content_h = 0;

	/* Title: "Authorization required" */
	PangoLayout *title = create_layout(cr, r->font_overlay_title, 0);
	pango_layout_set_text(title, "Authorization required", -1);
	int tw, th;
	pango_layout_get_pixel_size(title, &tw, &th);
	content_h += th + SPACE_L;

	/* Intent text */
	PangoLayout *intent_text = create_layout(cr, r->font_overlay_body, 0);
	pango_layout_set_text(intent_text, intent->natural_text, -1);
	pango_layout_set_width(intent_text, content_w * PANGO_SCALE);
	pango_layout_set_wrap(intent_text, PANGO_WRAP_WORD_CHAR);
	pango_layout_set_height(intent_text, -3);
	pango_layout_set_ellipsize(intent_text, PANGO_ELLIPSIZE_END);
	int itw, ith;
	pango_layout_get_pixel_size(intent_text, &itw, &ith);
	content_h += ith + SPACE_L;

	/* Actions summary */
	PangoLayout *actions_layout = NULL;
	int alw = 0, alh = 0;
	if (intent->actions_summary[0]) {
		actions_layout = create_layout(cr, r->font_overlay_mono, 0);
		pango_layout_set_text(actions_layout, intent->actions_summary, -1);
		pango_layout_set_width(actions_layout, content_w * PANGO_SCALE);
		pango_layout_set_wrap(actions_layout, PANGO_WRAP_WORD_CHAR);
		pango_layout_get_pixel_size(actions_layout, &alw, &alh);
		content_h += alh + SPACE_L;
	}

	/* Scope block */
	PangoLayout *scope_layout = NULL;
	int slw = 0, slh = 0;
	if (intent->resources[0]) {
		scope_layout = create_layout(cr, r->font_overlay_mono, 0);
		char scope_text[600];
		snprintf(scope_text, sizeof(scope_text),
			"\xe2\x9a\x91  Sandboxed to: %s", intent->resources);
		pango_layout_set_text(scope_layout, scope_text, -1);
		pango_layout_set_width(scope_layout, content_w * PANGO_SCALE);
		pango_layout_set_wrap(scope_layout, PANGO_WRAP_WORD_CHAR);
		pango_layout_get_pixel_size(scope_layout, &slw, &slh);
		content_h += slh + SPACE_L;
	}

	/* Reversibility hint */
	PangoLayout *rev_layout = create_layout(cr, r->font_overlay_body, 0);
	const char *rev_text = intent->reversible ?
		"This action is reversible." :
		"This action is not reversible.";
	pango_layout_set_text(rev_layout, rev_text, -1);
	int rlw, rlh;
	pango_layout_get_pixel_size(rev_layout, &rlw, &rlh);
	content_h += rlh + SPACE_XL;

	/* Button row */
	content_h += OVERLAY_BUTTON_H;

	int panel_h = content_h + 2 * OVERLAY_PADDING;
	int panel_y = (r->height - panel_h) / 2;

	/* Panel background */
	rounded_rect(cr, panel_x, panel_y, panel_w, panel_h, OVERLAY_RADIUS);
	set_color(cr, BG_OVERLAY_PANEL);
	cairo_fill(cr);

	/* Panel border */
	rounded_rect(cr, panel_x, panel_y, panel_w, panel_h, OVERLAY_RADIUS);
	set_color(cr, BORDER_OVERLAY);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);

	/* Draw content */
	int cx = panel_x + OVERLAY_PADDING;
	int cy = panel_y + OVERLAY_PADDING;

	/* Title */
	cairo_move_to(cr, cx, cy);
	set_color(cr, TEXT_PRIMARY);
	pango_cairo_show_layout(cr, title);
	cy += th + SPACE_L;

	/* Intent text */
	cairo_move_to(cr, cx, cy);
	set_color(cr, TEXT_SECONDARY);
	pango_cairo_show_layout(cr, intent_text);
	cy += ith + SPACE_L;

	/* Actions summary */
	if (actions_layout) {
		cairo_move_to(cr, cx, cy);
		set_color(cr, TEXT_SECONDARY);
		pango_cairo_show_layout(cr, actions_layout);
		cy += alh + SPACE_L;
	}

	/* Scope block */
	if (scope_layout) {
		cairo_move_to(cr, cx, cy);
		set_color(cr, ACCENT_GREEN);
		pango_cairo_show_layout(cr, scope_layout);
		cy += slh + SPACE_L;
	}

	/* Reversibility hint */
	cairo_move_to(cr, cx, cy);
	set_color(cr, intent->reversible ? ACCENT_GREEN : ACCENT_RED);
	pango_cairo_show_layout(cr, rev_layout);
	cy += rlh + SPACE_XL;

	/* Button row — right-aligned */
	int btn_cancel_w = 80;
	int btn_confirm_w = 100;
	int buttons_total = btn_cancel_w + OVERLAY_BUTTON_GAP + btn_confirm_w;
	int btn_x = panel_x + panel_w - OVERLAY_PADDING - buttons_total;

	/* Expose hit rects to compositor for mouse click handling */
	r->overlay_cancel_x  = btn_x;
	r->overlay_cancel_y  = cy;
	r->overlay_cancel_w  = btn_cancel_w;
	r->overlay_cancel_h  = OVERLAY_BUTTON_H;
	int btn_confirm_x_save = btn_x + btn_cancel_w + OVERLAY_BUTTON_GAP;
	r->overlay_confirm_x = btn_confirm_x_save;
	r->overlay_confirm_y = cy;
	r->overlay_confirm_w = btn_confirm_w;
	r->overlay_confirm_h = OVERLAY_BUTTON_H;
	r->overlay_buttons_valid = true;

	/* Cancel button — outlined */
	rounded_rect(cr, btn_x, cy, btn_cancel_w, OVERLAY_BUTTON_H,
		OVERLAY_BUTTON_R);
	set_color(cr, BORDER_OVERLAY);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);

	PangoLayout *cancel_text = create_layout(cr, r->font_overlay_button, 0);
	pango_layout_set_text(cancel_text, "Cancel", -1);
	int ctw, cth;
	pango_layout_get_pixel_size(cancel_text, &ctw, &cth);
	cairo_move_to(cr, btn_x + (btn_cancel_w - ctw) / 2,
		cy + (OVERLAY_BUTTON_H - cth) / 2);
	set_color(cr, TEXT_SECONDARY);
	pango_cairo_show_layout(cr, cancel_text);
	g_object_unref(cancel_text);

	/* Confirm button — filled blue */
	int btn_confirm_x = btn_x + btn_cancel_w + OVERLAY_BUTTON_GAP;
	rounded_rect(cr, btn_confirm_x, cy, btn_confirm_w, OVERLAY_BUTTON_H,
		OVERLAY_BUTTON_R);
	set_color(cr, ACCENT_BLUE);
	cairo_fill(cr);

	PangoLayout *confirm_text = create_layout(cr, r->font_overlay_button, 0);
	pango_layout_set_text(confirm_text, "Confirm", -1);
	int cfw, cfh;
	pango_layout_get_pixel_size(confirm_text, &cfw, &cfh);
	cairo_move_to(cr, btn_confirm_x + (btn_confirm_w - cfw) / 2,
		cy + (OVERLAY_BUTTON_H - cfh) / 2);
	set_color(cr, TEXT_PRIMARY);
	pango_cairo_show_layout(cr, confirm_text);
	g_object_unref(confirm_text);

	/* Keyboard hints below buttons */
	cy += OVERLAY_BUTTON_H + SPACE_S;
	PangoLayout *hints = create_layout(cr, r->font_keyboard_hint,
		TRACKING_CAPTION2);
	pango_layout_set_text(hints,
		"N / Esc = cancel    Y / Enter = confirm", -1);
	int hkw, hkh;
	pango_layout_get_pixel_size(hints, &hkw, &hkh);
	cairo_move_to(cr, panel_x + panel_w - OVERLAY_PADDING - hkw, cy);
	set_color(cr, TEXT_TERTIARY);
	pango_cairo_show_layout(cr, hints);
	g_object_unref(hints);

	/* Cleanup */
	g_object_unref(title);
	g_object_unref(intent_text);
	if (actions_layout) g_object_unref(actions_layout);
	if (scope_layout) g_object_unref(scope_layout);
	g_object_unref(rev_layout);
}

/* ── Expanded card full-window overlay ── */

static void draw_expanded_overlay(struct marshal_renderer *r,
		struct marshal_feed *feed) {
	cairo_t *cr = r->cr;

	int idx = feed->expanded_card;
	if (idx < 0 || idx >= feed->count)
		return;

	MarshalIntent *intent = &feed->intents[idx];

	/* Full-screen dimming */
	set_color(cr, OVERLAY_DIM);
	cairo_rectangle(cr, 0, 0, r->width, r->height);
	cairo_fill(cr);

	/* Panel: nearly full screen with margins */
	int margin_h = 32;
	int margin_v = 32;
	int panel_w = r->width - 2 * margin_h;
	int panel_h = r->height - 2 * margin_v;
	int panel_x = margin_h;
	int panel_y = margin_v;
	int content_w = panel_w - 2 * OVERLAY_PADDING;
	int cx = panel_x + OVERLAY_PADDING;
	int content_top = panel_y + OVERLAY_PADDING;

	/* Panel background */
	rounded_rect(cr, panel_x, panel_y, panel_w, panel_h, OVERLAY_RADIUS);
	set_color(cr, BG_OVERLAY_PANEL);
	cairo_fill(cr);

	/* Panel border */
	rounded_rect(cr, panel_x, panel_y, panel_w, panel_h, OVERLAY_RADIUS);
	set_color(cr, BORDER_OVERLAY);
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);

	int cy = content_top;

	/* Title: intent natural text */
	PangoLayout *title = create_layout(cr, r->font_intent_title, 0);
	pango_layout_set_text(title, intent->natural_text, -1);
	pango_layout_set_width(title, content_w * PANGO_SCALE);
	pango_layout_set_wrap(title, PANGO_WRAP_WORD_CHAR);
	int tw, th;
	pango_layout_get_pixel_size(title, &tw, &th);
	cairo_move_to(cr, cx, cy);
	set_color(cr, TEXT_PRIMARY);
	pango_cairo_show_layout(cr, title);
	g_object_unref(title);
	cy += th + SPACE_S;

	/* Action chain */
	if (intent->action_chain[0]) {
		PangoLayout *chain = create_layout(cr, r->font_action_chain, 0);
		pango_layout_set_text(chain, intent->action_chain, -1);
		int cw, ch;
		pango_layout_get_pixel_size(chain, &cw, &ch);
		cairo_move_to(cr, cx, cy);
		set_color(cr, TEXT_SECONDARY);
		pango_cairo_show_layout(cr, chain);
		g_object_unref(chain);
		cy += ch + SPACE_S;
	}

	/* Separator */
	cairo_set_source_rgba(cr, 0, 0, 0, 0.06);
	cairo_rectangle(cr, cx, cy, content_w, 1);
	cairo_fill(cr);
	cy += 1 + SPACE_S;

	/* Close button — top right "✕" */
	int close_sz = 28;
	int close_x = panel_x + panel_w - OVERLAY_PADDING - close_sz;
	int close_y = content_top;
	r->expanded_close_x = close_x;
	r->expanded_close_y = close_y;
	r->expanded_close_w = close_sz;
	r->expanded_close_h = close_sz;
	r->expanded_overlay_valid = true;
	{
		cairo_save(cr);
		double mx = close_x + close_sz / 2.0;
		double my = close_y + close_sz / 2.0;
		double d = 7;
		cairo_set_line_width(cr, 1.5);
		set_color(cr, TEXT_SECONDARY);
		cairo_move_to(cr, mx - d, my - d);
		cairo_line_to(cr, mx + d, my + d);
		cairo_move_to(cr, mx + d, my - d);
		cairo_line_to(cr, mx - d, my + d);
		cairo_stroke(cr);
		cairo_restore(cr);
	}

	/* Content area for result summary with scroll */
	int content_area_top = cy;
	int content_area_h = panel_y + panel_h - OVERLAY_PADDING - cy;

	r->expanded_content_x = cx;
	r->expanded_content_y = content_area_top;
	r->expanded_content_w = content_w;
	r->expanded_content_h = content_area_h;

	if (intent->result_summary[0]) {
		/* Measure full content height */
		PangoLayout *result = create_layout(cr, r->font_body, 0);
		pango_layout_set_text(result, intent->result_summary, -1);
		pango_layout_set_width(result, content_w * PANGO_SCALE);
		pango_layout_set_wrap(result, PANGO_WRAP_WORD_CHAR);
		int rw, rh;
		pango_layout_get_pixel_size(result, &rw, &rh);
		feed->expanded_content_h = rh;

		/* Clamp scroll */
		float max_scroll = (float)(rh - content_area_h);
		if (max_scroll < 0) max_scroll = 0;
		if (feed->expanded_scroll < 0)
			feed->expanded_scroll = 0;
		if (feed->expanded_scroll > max_scroll)
			feed->expanded_scroll = max_scroll;

		/* Clip to content area */
		cairo_save(cr);
		cairo_rectangle(cr, cx, content_area_top,
			content_w, content_area_h);
		cairo_clip(cr);

		/* Draw with scroll offset */
		cairo_move_to(cr, cx,
			content_area_top - (int)feed->expanded_scroll);
		set_color(cr, TEXT_PRIMARY);
		pango_cairo_show_layout(cr, result);
		g_object_unref(result);
		cairo_restore(cr);

		/* Scroll bar (if content overflows) */
		if (rh > content_area_h) {
			int sb_w = 4;
			int sb_x = panel_x + panel_w - OVERLAY_PADDING / 2 - sb_w;
			double ratio = (double)content_area_h / rh;
			int thumb_h = (int)(content_area_h * ratio);
			if (thumb_h < 20) thumb_h = 20;
			int track_h = content_area_h - thumb_h;
			int thumb_y = content_area_top +
				(int)(track_h * (feed->expanded_scroll / max_scroll));
			cairo_set_source_rgba(cr, 0, 0, 0, 0.15);
			rounded_rect(cr, sb_x, thumb_y, sb_w, thumb_h, 2);
			cairo_fill(cr);
		}
	}

	/* Keyboard hint */
	PangoLayout *hint = create_layout(cr, r->font_keyboard_hint,
		TRACKING_CAPTION2);
	pango_layout_set_text(hint, "Esc = close    Scroll = navigate", -1);
	int hkw, hkh;
	pango_layout_get_pixel_size(hint, &hkw, &hkh);
	cairo_move_to(cr,
		panel_x + panel_w - OVERLAY_PADDING - hkw,
		panel_y + panel_h - OVERLAY_PADDING + SPACE_XS);
	set_color(cr, TEXT_TERTIARY);
	/* Only show if it fits below content */
	if (panel_y + panel_h - OVERLAY_PADDING + SPACE_XS + hkh
			<= panel_y + panel_h)
		pango_cairo_show_layout(cr, hint);
	g_object_unref(hint);
}

/* ── Full frame render ── */

unsigned char *renderer_draw_frame(struct marshal_renderer *r,
		struct marshal_feed *feed, struct marshal_input *input, int *stride) {
	if (!r->surface || !r->cr) return NULL;
	cairo_t *cr = r->cr;

	/* Reset hit-test state */
	r->overlay_buttons_valid = false;
	r->expanded_overlay_valid = false;
	r->card_hit_count = 0;

	bool show_feed = r->status && r->status->history_open;
	int wall_h = r->height - INPUT_HEIGHT;

	/* 1. Background: wallpaper, or — when no wallpaper image is set —
	 * a Claude/ChatGPT-style centred "Hi, <Name>" greeting. The greeting
	 * lives on the panel cairo surface, so app windows (which are
	 * scene-graph siblings above the panel buffer) cleanly cover it the
	 * moment something is launched. No special hide-when-app-running
	 * logic needed. */
	if (!show_feed && r->wallpaper) {
		draw_wallpaper(r, 0, wall_h);
	} else {
		set_color(cr, BG_BASE);
		cairo_paint(cr);
		if (!show_feed) {
			draw_greeting(r, 0, wall_h);
		}
	}

	int top_offset = 0;

	/* 2. Inference-offline banner (floats over wallpaper) */
	if (!r->inference_online) {
		draw_status_banner(r);
		top_offset = STATUS_BANNER_H;
	}

	pthread_mutex_lock(&feed->mutex);

	/* 3. Feed cards (only when history is open) */
	if (show_feed) {
		/* Semi-transparent scrim over wallpaper when history is open
		 * so cards are readable */
		if (r->wallpaper) {
			cairo_set_source_rgba(cr, 1, 1, 1, 0.82);
			cairo_rectangle(cr, 0, 0, r->width, wall_h);
			cairo_fill(cr);
		}

		int feed_top = top_offset + FEED_PADDING_T;
		int feed_bottom = r->height - INPUT_HEIGHT - SPACE_M;

		/* Briefing */
		int briefing_h = 0;
		if (feed->briefing.loaded && !feed->briefing.empty) {
			briefing_h = draw_briefing_card(r,
				&feed->briefing, feed_top);
			if (briefing_h > 0)
				briefing_h += CARD_GAP;
		}

		/* Watchers */
		int watchers_h = draw_watcher_cards(r, feed,
			feed_top + briefing_h);

		if (feed->count == 0 && !feed->briefing.loaded
				&& feed->watcher_count == 0) {
			draw_empty_state(r,
				feed_top + briefing_h + watchers_h,
				feed_bottom);
		} else if (feed->count > 0) {
			cairo_save(cr);
			cairo_rectangle(cr, 0, top_offset,
				r->width, r->height - INPUT_HEIGHT - top_offset);
			cairo_clip(cr);

			int card_heights[MAX_INTENTS];
			for (int i = 0; i < feed->count; i++)
				card_heights[i] = measure_card_height(r,
					&feed->intents[i],
					i == feed->expanded_card);

			int y = feed_bottom + (int)feed->scroll_offset;
			for (int i = feed->count - 1; i >= 0; i--) {
				y -= card_heights[i];
				if (y < r->height + 50 &&
						y + card_heights[i] > top_offset - 50)
					draw_card(r, &feed->intents[i], y,
					i == feed->selected_card, i,
					i == feed->expanded_card);
				y -= CARD_GAP;
			}

			cairo_restore(cr);
		}
	}

	/* 4. Taskbar (always on top) */
	draw_taskbar(r, input, feed);

	/* 5. Quick-settings dropdown */
	draw_dropdown(r, r->status);

	/* 6. Authorization overlay (on top of everything) */
	if (feed->awaiting_confirm) {
		draw_auth_overlay(r, feed);
	}

	/* 7. Expanded card full-window overlay */
	if (feed->expanded_card >= 0 && !feed->awaiting_confirm) {
		draw_expanded_overlay(r, feed);
	}

	pthread_mutex_unlock(&feed->mutex);

	cairo_surface_flush(r->surface);
	*stride = cairo_image_surface_get_stride(r->surface);
	return cairo_image_surface_get_data(r->surface);
}

int renderer_input_hit_test(struct marshal_renderer *r,
		struct marshal_input *input, double panel_x) {
	if (input->len == 0) return 0;

	int input_x     = TASKBAR_ICON_W + INPUT_PADDING_L;
	int input_max_x = r->input_field_right_x > 0
		? r->input_field_right_x : (r->width - TASKBAR_APPS_W);

	/* Recreate the same layout parameters as draw_taskbar */
	PangoLayout *layout = create_layout(r->cr, r->font_input, 0);
	pango_layout_set_width(layout,
		(input_max_x - input_x - SPACE_L) * PANGO_SCALE);
	pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
	pango_layout_set_text(layout, input->buf, input->len);

	/* Convert screen x to layout-relative x (clamped to [0, layout width]) */
	double rel_x = panel_x - input_x;
	if (rel_x < 0) rel_x = 0;

	int index, trailing;
	pango_layout_xy_to_index(layout,
		(int)(rel_x * PANGO_SCALE), 0, &index, &trailing);

	/* trailing==1 means the click was in the right half of the glyph —
	 * advance index past this codepoint so the cursor lands after it. */
	if (trailing) {
		const char *p   = input->buf + index;
		const char *end = input->buf + input->len;
		if (p < end) {
			p++;
			while (p < end && ((unsigned char)*p & 0xC0) == 0x80)
				p++;
		}
		index = (int)(p - input->buf);
	}

	g_object_unref(layout);

	if (index < 0)           index = 0;
	if (index > input->len)  index = input->len;
	return index;
}

int renderer_card_hit_test(struct marshal_renderer *r,
		struct marshal_feed *feed, double panel_y) {
	if (!feed || feed->count == 0 || !r->cr) return -1;

	int feed_bottom = r->height - INPUT_HEIGHT - SPACE_M;
	int y = feed_bottom + (int)feed->scroll_offset;

	for (int i = feed->count - 1; i >= 0; i--) {
		int card_h = measure_card_height(r, &feed->intents[i],
			i == feed->expanded_card);
		y -= card_h;
		float y_off = feed->intents[i].anim_y.pos;
		int card_y = y + (int)y_off;

		if (panel_y >= card_y && panel_y < card_y + card_h)
			return i;
		y -= CARD_GAP;
	}
	return -1;
}

int renderer_card_copy_text(struct marshal_feed *feed, int card_idx,
		char *buf, int buf_size) {
	if (!feed || card_idx < 0 || card_idx >= feed->count || buf_size <= 0)
		return 0;

	MarshalIntent *intent = &feed->intents[card_idx];
	int off = 0;

	/* Natural text */
	if (intent->natural_text[0] && off < buf_size - 1)
		off += snprintf(buf + off, buf_size - off,
			"%s", intent->natural_text);

	/* Result summary */
	if (intent->result_summary[0] && off < buf_size - 2)
		off += snprintf(buf + off, buf_size - off,
			"\n%s", intent->result_summary);

	/* Action chain */
	if (intent->action_chain[0] && off < buf_size - 2)
		off += snprintf(buf + off, buf_size - off,
			"\n%s", intent->action_chain);

	return off;
}

int renderer_card_text_at(struct marshal_renderer *r,
		double panel_x, double panel_y, int *byte_offset) {
	if (!r || !r->cr) return -1;

	for (int i = 0; i < r->card_hit_count; i++) {
		struct card_text_hit *hit = &r->card_hits[i];
		if (panel_x < hit->text_x || panel_x >= hit->text_x + hit->text_w)
			continue;

		/* Check title region */
		if (panel_y >= hit->title_y &&
				panel_y < hit->title_y + hit->title_h) {
			PangoLayout *layout = create_layout(r->cr,
				r->font_intent_title, 0);
			pango_layout_set_text(layout, hit->title, hit->title_len);
			pango_layout_set_width(layout, hit->text_w * PANGO_SCALE);
			pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
			pango_layout_set_height(layout, -2);

			int index = 0, trailing = 0;
			pango_layout_xy_to_index(layout,
				(int)((panel_x - hit->text_x) * PANGO_SCALE),
				(int)((panel_y - hit->title_y) * PANGO_SCALE),
				&index, &trailing);

			if (trailing > 0 && index < hit->title_len) {
				const char *p = hit->title + index;
				while (trailing > 0 && *p) {
					unsigned char c = (unsigned char)*p;
					if (c < 0x80) p++; else if (c < 0xE0) p += 2;
					else if (c < 0xF0) p += 3; else p += 4;
					trailing--;
				}
				index = (int)(p - hit->title);
			}
			g_object_unref(layout);

			if (index < 0) index = 0;
			if (index > hit->title_len) index = hit->title_len;
			*byte_offset = index;  /* title bytes start at 0 in combined */
			return hit->card_idx;
		}

		/* Check result region */
		if (hit->result_len > 0 && panel_y >= hit->result_y &&
				panel_y < hit->result_y + hit->result_h) {
			PangoLayout *layout = create_layout(r->cr,
				r->font_body, 0);
			pango_layout_set_text(layout, hit->result, hit->result_len);
			pango_layout_set_width(layout, hit->text_w * PANGO_SCALE);
			pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
			/* No height limit — use stored hit->result_h for bounds */

			int index = 0, trailing = 0;
			pango_layout_xy_to_index(layout,
				(int)((panel_x - hit->text_x) * PANGO_SCALE),
				(int)((panel_y - hit->result_y) * PANGO_SCALE),
				&index, &trailing);

			if (trailing > 0 && index < hit->result_len) {
				const char *p = hit->result + index;
				while (trailing > 0 && *p) {
					unsigned char c = (unsigned char)*p;
					if (c < 0x80) p++; else if (c < 0xE0) p += 2;
					else if (c < 0xF0) p += 3; else p += 4;
					trailing--;
				}
				index = (int)(p - hit->result);
			}
			g_object_unref(layout);

			if (index < 0) index = 0;
			if (index > hit->result_len) index = hit->result_len;
			/* Offset into combined text: title_len + 1 (\n) + index */
			*byte_offset = hit->title_len + 1 + index;
			return hit->card_idx;
		}
	}

	return -1;
}

int renderer_card_sel_text(struct marshal_renderer *r,
		char *buf, int buf_size) {
	if (!r || r->card_sel.card_idx < 0 || buf_size <= 0)
		return 0;
	if (r->card_sel.anchor == r->card_sel.focus)
		return 0;

	/* Find the card_hit for this card */
	struct card_text_hit *hit = NULL;
	for (int i = 0; i < r->card_hit_count; i++) {
		if (r->card_hits[i].card_idx == r->card_sel.card_idx) {
			hit = &r->card_hits[i];
			break;
		}
	}
	if (!hit) return 0;

	int s = r->card_sel.anchor < r->card_sel.focus
		? r->card_sel.anchor : r->card_sel.focus;
	int e = r->card_sel.anchor < r->card_sel.focus
		? r->card_sel.focus : r->card_sel.anchor;
	if (s < 0) s = 0;
	if (e > hit->text_len) e = hit->text_len;
	if (s >= e) return 0;

	int len = e - s;
	if (len > buf_size - 1) len = buf_size - 1;
	memcpy(buf, hit->text + s, len);
	buf[len] = '\0';
	return len;
}
