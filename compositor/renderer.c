#define _GNU_SOURCE
#include "renderer.h"
#include "colors.h"
#include "typography.h"
#include "geometry.h"
#include "feed.h"
#include "input.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

	struct color c = ACCENT_GREEN;
	cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
		c.b / 255.0, 0.70);
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

static bool state_is_pending(LeavesCardState s) {
	return s == CARD_STATE_PENDING || s == CARD_STATE_EXECUTING;
}

/* ── Renderer lifecycle ── */

struct leaves_renderer *renderer_create(void) {
	struct leaves_renderer *r = calloc(1, sizeof(*r));
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

	return r;
}

void renderer_destroy(struct leaves_renderer *r) {
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
	free(r);
}

void renderer_resize(struct leaves_renderer *r, int width, int height) {
	if (r->cr) cairo_destroy(r->cr);
	if (r->surface) cairo_surface_destroy(r->surface);
	r->width = width;
	r->height = height;
	r->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
	r->cr = cairo_create(r->surface);
}

/* ── Measure card height ── */

static int measure_card_height(struct leaves_renderer *r,
		LeavesIntent *intent) {
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

static struct color indicator_for_state(LeavesCardState state, float opacity) {
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

static int draw_card(struct leaves_renderer *r, LeavesIntent *intent,
		int y_base) {
	cairo_t *cr = r->cr;
	int card_w = r->width - 2 * CARD_MARGIN_H;
	int card_x = CARD_MARGIN_H;

	float opacity = intent->anim_opacity.pos;
	if (opacity < 0) opacity = 0;
	if (opacity > 1) opacity = 1;
	float y_off = intent->anim_y.pos;
	int card_y = y_base + (int)y_off;

	int card_h = measure_card_height(r, intent);

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
	{
		struct color c = BORDER_CARD;
		cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
			c.b / 255.0, (c.a / 255.0) * opacity);
	}
	cairo_set_line_width(cr, 1.0);
	cairo_stroke(cr);
	cairo_restore(cr);

	/* Indicator bar */
	struct color indicator_color = indicator_for_state(intent->state, opacity);
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

	/* LINE 1: intent natural text */
	{
		PangoLayout *layout = create_layout(cr, r->font_intent_title, 0);
		pango_layout_set_text(layout, intent->natural_text, -1);
		pango_layout_set_width(layout, inner_w * PANGO_SCALE);
		pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
		pango_layout_set_wrap(layout, PANGO_WRAP_WORD_CHAR);
		pango_layout_set_height(layout, -2);

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

	return card_h;
}

/* ── Briefing card ── */

static int draw_briefing_card(struct leaves_renderer *r,
		LeavesBriefing *briefing, int y) {
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
		LeavesBriefingSection *sec = &briefing->sections[s];
		for (int g = 0; g < sec->group_count; g++) {
			group_lines++;
			LeavesBriefingGroup *grp = &sec->groups[g];
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
		LeavesBriefingSection *sec = &briefing->sections[s];
		for (int g = 0; g < sec->group_count; g++) {
			LeavesBriefingGroup *grp = &sec->groups[g];
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

/* ── Empty state ── */

static void draw_empty_state(struct leaves_renderer *r, int top, int bottom) {
	cairo_t *cr = r->cr;
	int center_x = r->width / 2;

	PangoLayout *heading = create_layout(cr, r->font_empty_heading,
		TRACKING_EMPTY_H);
	pango_layout_set_text(heading, "Leaves OS", -1);
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

static void draw_status_banner(struct leaves_renderer *r) {
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

static void draw_taskbar(struct leaves_renderer *r,
		struct leaves_input *input, struct leaves_feed *feed) {
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
	draw_leaf_glyph(cr, icon_cx, icon_cy);

	/* Separator after icon */
	set_color(cr, BORDER_SEPARATOR);
	cairo_rectangle(cr, TASKBAR_ICON_W, bar_y + 10, 1,
		INPUT_HEIGHT - 20);
	cairo_fill(cr);

	/* ── Center zone: input field ── */
	int input_x = TASKBAR_ICON_W + INPUT_PADDING_L;
	int input_max_x = r->width - TASKBAR_APPS_W;

	PangoLayout *layout = create_layout(cr, r->font_input, 0);
	pango_layout_set_width(layout,
		(input_max_x - input_x - SPACE_L) * PANGO_SCALE);
	pango_layout_set_ellipsize(layout, PANGO_ELLIPSIZE_END);
	int lw, lh;

	if (input->len == 0) {
		pango_layout_set_text(layout, "What do you want to do?", -1);
		pango_layout_get_pixel_size(layout, &lw, &lh);
		cairo_move_to(cr, input_x, bar_y + (INPUT_HEIGHT - lh) / 2);
		set_color(cr, TEXT_PLACEHOLDER);
		pango_cairo_show_layout(cr, layout);
	} else {
		pango_layout_set_text(layout, input->buf, input->len);
		pango_layout_get_pixel_size(layout, &lw, &lh);
		cairo_move_to(cr, input_x, bar_y + (INPUT_HEIGHT - lh) / 2);
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

	/* ── Right zone: running apps / active intents ── */
	/* Separator before apps zone */
	set_color(cr, BORDER_SEPARATOR);
	cairo_rectangle(cr, r->width - TASKBAR_APPS_W, bar_y + 10, 1,
		INPUT_HEIGHT - 20);
	cairo_fill(cr);

	/* Count active intents (pending, executing, awaiting confirm) */
	int active_count = 0;
	for (int i = 0; i < feed->count; i++) {
		LeavesCardState s = feed->intents[i].state;
		if (s == CARD_STATE_PENDING || s == CARD_STATE_EXECUTING ||
				s == CARD_STATE_AWAITING_CONFIRM)
			active_count++;
	}

	int apps_cx = r->width - TASKBAR_APPS_W / 2;
	int apps_cy = bar_y + INPUT_HEIGHT / 2;

	if (active_count > 0) {
		/* Draw activity dots (max 5 visible) */
		int dots = active_count > 5 ? 5 : active_count;
		int total_w = dots * (TASKBAR_DOT_R * 2) +
			(dots - 1) * TASKBAR_DOT_GAP;
		int dot_x = apps_cx - total_w / 2 + TASKBAR_DOT_R;

		for (int i = 0; i < dots; i++) {
			/* Pulse animation for active dots */
			double t = monotonic_time_s();
			double phase = fmod(t + i * 0.3, 2.0);
			double alpha_mult = phase < 1.0 ?
				0.5 + 0.5 * phase : 0.5 + 0.5 * (2.0 - phase);

			struct color c = ACCENT_BLUE;
			cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
				c.b / 255.0, alpha_mult * 0.9);
			cairo_arc(cr, dot_x, apps_cy,
				TASKBAR_DOT_R, 0, 2 * M_PI);
			cairo_fill(cr);

			dot_x += TASKBAR_DOT_R * 2 + TASKBAR_DOT_GAP;
		}

		/* Show count if more than 5 */
		if (active_count > 5) {
			PangoLayout *cnt = create_layout(cr,
				r->font_keyboard_hint, 0);
			char cnt_str[8];
			snprintf(cnt_str, sizeof(cnt_str), "+%d",
				active_count - 5);
			pango_layout_set_text(cnt, cnt_str, -1);
			int cw, ch;
			pango_layout_get_pixel_size(cnt, &cw, &ch);
			cairo_move_to(cr, dot_x + SPACE_XS,
				apps_cy - ch / 2);
			set_color(cr, TEXT_TERTIARY);
			pango_cairo_show_layout(cr, cnt);
			g_object_unref(cnt);
		}
	} else {
		/* No active intents — show subtle idle indicator */
		PangoLayout *idle = create_layout(cr,
			r->font_keyboard_hint, TRACKING_CAPTION2);
		pango_layout_set_text(idle, "idle", -1);
		int iw, ih;
		pango_layout_get_pixel_size(idle, &iw, &ih);
		cairo_move_to(cr, apps_cx - iw / 2, apps_cy - ih / 2);
		set_color(cr, TEXT_TERTIARY);
		pango_cairo_show_layout(cr, idle);
		g_object_unref(idle);
	}
}

/* ── Authorization overlay ── */

static void draw_auth_overlay(struct leaves_renderer *r,
		struct leaves_feed *feed) {
	cairo_t *cr = r->cr;

	if (!feed->awaiting_confirm || feed->confirm_card_idx < 0 ||
			feed->confirm_card_idx >= feed->count)
		return;

	LeavesIntent *intent = &feed->intents[feed->confirm_card_idx];

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

/* ── Full frame render ── */

unsigned char *renderer_draw_frame(struct leaves_renderer *r,
		struct leaves_feed *feed, struct leaves_input *input, int *stride) {
	if (!r->surface || !r->cr) return NULL;
	cairo_t *cr = r->cr;

	/* 1. Clear with BG_BASE */
	set_color(cr, BG_BASE);
	cairo_paint(cr);

	int top_offset = 0;

	/* 2. Status banner */
	if (!r->inference_online) {
		draw_status_banner(r);
		top_offset = STATUS_BANNER_H;
	}

	/* Feed area bounds */
	int feed_top = top_offset + FEED_PADDING_T;
	int feed_bottom = r->height - INPUT_HEIGHT - SPACE_M;

	pthread_mutex_lock(&feed->mutex);

	/* 3. Briefing card (always at top when loaded) */
	int briefing_h = 0;
	if (feed->briefing.loaded && !feed->briefing.empty) {
		briefing_h = draw_briefing_card(r, &feed->briefing, feed_top);
		if (briefing_h > 0)
			briefing_h += CARD_GAP;
	}

	if (feed->count == 0 && !feed->briefing.loaded) {
		draw_empty_state(r, feed_top + briefing_h, feed_bottom);
	} else if (feed->count > 0) {
		/* 4. Clip feed area */
		cairo_save(cr);
		cairo_rectangle(cr, 0, top_offset,
			r->width, r->height - INPUT_HEIGHT - top_offset);
		cairo_clip(cr);

		/* Calculate card heights */
		int card_heights[MAX_INTENTS];
		for (int i = 0; i < feed->count; i++) {
			card_heights[i] = measure_card_height(r, &feed->intents[i]);
		}

		/* Draw cards bottom-up (newest at bottom) */
		int y = feed_bottom - (int)feed->scroll_offset;
		for (int i = feed->count - 1; i >= 0; i--) {
			y -= card_heights[i];
			if (y < r->height + 50 &&
					y + card_heights[i] > top_offset - 50) {
				draw_card(r, &feed->intents[i], y);
			}
			y -= CARD_GAP;
		}

		cairo_restore(cr);
	}

	/* 5. Taskbar (always on top) */
	draw_taskbar(r, input, feed);

	/* 6. Authorization overlay (on top of everything) */
	if (feed->awaiting_confirm) {
		draw_auth_overlay(r, feed);
	}

	pthread_mutex_unlock(&feed->mutex);

	cairo_surface_flush(r->surface);
	*stride = cairo_image_surface_get_stride(r->surface);
	return cairo_image_surface_get_data(r->surface);
}
