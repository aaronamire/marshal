#ifndef LEAVES_INPUT_H
#define LEAVES_INPUT_H

#include <stdbool.h>
#include <stdint.h>

struct leaves_feed;

struct leaves_input {
	char buf[1024];
	int len;
	int cursor_pos;  /* byte offset */
	bool cursor_visible;
	uint32_t cursor_blink_ms;
	struct leaves_feed *feed;
};

void input_init(struct leaves_input *input, struct leaves_feed *feed);
/* Returns true if display needs redraw */
bool input_handle_key(struct leaves_input *input, uint32_t keycode,
	uint32_t mods, const char *utf8, int utf8_len);
bool input_tick_cursor(struct leaves_input *input, uint32_t dt_ms);

#endif
