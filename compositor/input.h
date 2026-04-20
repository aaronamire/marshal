#ifndef MARSHAL_INPUT_H
#define MARSHAL_INPUT_H

#include <stdbool.h>
#include <stdint.h>

struct marshal_feed;

struct marshal_input {
	char buf[1024];
	int len;
	int cursor_pos;  /* byte offset */
	bool cursor_visible;
	uint32_t cursor_blink_ms;
	struct marshal_feed *feed;

	/* Selection: sel_anchor == -1 means no active selection.
	 * When sel_anchor != sel_focus, the selected byte range is
	 * [min(sel_anchor,sel_focus), max(sel_anchor,sel_focus)). */
	int sel_anchor;  /* byte offset where selection started */
	int sel_focus;   /* byte offset where selection extends to */
};

void input_init(struct marshal_input *input, struct marshal_feed *feed);
/* Returns true if display needs redraw */
bool input_handle_key(struct marshal_input *input, uint32_t keycode,
	uint32_t mods, const char *utf8, int utf8_len);
bool input_tick_cursor(struct marshal_input *input, uint32_t dt_ms);
/* Insert text at cursor (skips newlines; handles UTF-8 safely) */
void input_paste(struct marshal_input *input, const char *text, int len);
/* Clear the active selection without moving the cursor */
void input_clear_selection(struct marshal_input *input);
/* Delete the selected range and collapse.  Returns true if anything deleted. */
bool input_delete_selection(struct marshal_input *input);
/* Select the word (alphanum/_/-) that contains byte_pos */
void input_select_word(struct marshal_input *input, int byte_pos);

#endif
