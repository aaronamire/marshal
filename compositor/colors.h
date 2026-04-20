#ifndef MARSHAL_COLORS_H
#define MARSHAL_COLORS_H

#include <stdint.h>

struct color {
	uint8_t r, g, b, a;
};

/* ── Backgrounds (day mode) ── */
#define BG_BASE          ((struct color){0xFF, 0xFF, 0xFF, 0xFF})  /* #FFFFFF */
#define BG_ELEVATED      ((struct color){0xF5, 0xF5, 0xF5, 0xFF})  /* #F5F5F5 */
#define BG_INPUT         ((struct color){0xFA, 0xFA, 0xFA, 0xFF})  /* #FAFAFA */
#define BG_OVERLAY       ((struct color){0xF0, 0xF0, 0xF0, 0xFF})  /* #F0F0F0 */

/* ── Text ── */
#define TEXT_PRIMARY     ((struct color){0x00, 0x00, 0x00, 0xE8})  /* rgba(0,0,0,0.91) */
#define TEXT_SECONDARY   ((struct color){0x00, 0x00, 0x00, 0x99})  /* rgba(0,0,0,0.60) */
#define TEXT_TERTIARY    ((struct color){0x00, 0x00, 0x00, 0x52})  /* rgba(0,0,0,0.32) */
#define TEXT_PLACEHOLDER ((struct color){0x00, 0x00, 0x00, 0x33})  /* rgba(0,0,0,0.20) */

/* ── Accents ── */
#define ACCENT_BLUE      ((struct color){0x25, 0x63, 0xEB, 0xFF})  /* #2563EB */
#define ACCENT_GREEN     ((struct color){0x16, 0xA3, 0x4A, 0xFF})  /* #16A34A */
#define ACCENT_AMBER     ((struct color){0xD9, 0x77, 0x06, 0xFF})  /* #D97706 */
#define ACCENT_RED       ((struct color){0xDC, 0x26, 0x26, 0xFF})  /* #DC2626 */

/* ── Structural ── */
#define BORDER_CARD      ((struct color){0x00, 0x00, 0x00, 0x1A})  /* rgba(0,0,0,0.10) */
#define BORDER_SEPARATOR ((struct color){0x00, 0x00, 0x00, 0x0F})  /* rgba(0,0,0,0.06) */
#define FILL_CARD        ((struct color){0x00, 0x00, 0x00, 0x08})  /* rgba(0,0,0,0.03) */

/* ── Indicator bar ── */
#define INDICATOR_HISTORY ((struct color){0x00, 0x00, 0x00, 0x14}) /* rgba(0,0,0,0.08) */

/* ── Overlay ── */
#define BG_OVERLAY_PANEL  ((struct color){0xFB, 0xFB, 0xFB, 0xFF})  /* #FBFBFB */
#define BORDER_OVERLAY    ((struct color){0x00, 0x00, 0x00, 0x1F})  /* rgba(0,0,0,0.12) */
#define OVERLAY_DIM       ((struct color){0x00, 0x00, 0x00, 0x66})  /* rgba(0,0,0,0.40) */

/* ── Helper ── */
static inline void set_color(cairo_t *cr, struct color c) {
	cairo_set_source_rgba(cr, c.r / 255.0, c.g / 255.0,
		c.b / 255.0, c.a / 255.0);
}

static inline struct color color_with_alpha(struct color c, uint8_t a) {
	return (struct color){c.r, c.g, c.b, a};
}

#endif
