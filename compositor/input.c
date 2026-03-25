#include "input.h"
#include "feed.h"
#include <string.h>

void input_init(struct leaves_input *input, struct leaves_feed *feed) {
	memset(input, 0, sizeof(*input));
	input->cursor_visible = true;
	input->feed = feed;
}

/* UTF-8: bytes before cursor_pos */
static int utf8_prev(const char *buf, int pos) {
	if (pos <= 0) return 0;
	pos--;
	while (pos > 0 && (buf[pos] & 0xC0) == 0x80) {
		pos--;
	}
	return pos;
}

/* UTF-8: bytes after cursor_pos */
static int utf8_next(const char *buf, int len, int pos) {
	if (pos >= len) return len;
	pos++;
	while (pos < len && (buf[pos] & 0xC0) == 0x80) {
		pos++;
	}
	return pos;
}

/* Modifier bit flags (matching xkbcommon) */
#define MOD_CTRL (1 << 0)

bool input_handle_key(struct leaves_input *input, uint32_t keycode,
		uint32_t mods, const char *utf8, int utf8_len) {
	/* When overlay is active, capture all keys */
	if (input->feed && input->feed->awaiting_confirm) {
		switch (keycode) {
		case 28: /* Enter / Return */
		case 21: /* Y key */
			feed_confirm(input->feed);
			return true;
		case 1:  /* Escape */
		case 49: /* N key */
			feed_cancel(input->feed);
			return true;
		default:
			return true; /* swallow all other keys */
		}
	}

	/* Reset cursor blink on any keypress */
	input->cursor_visible = true;
	input->cursor_blink_ms = 0;

	bool ctrl = (mods & MOD_CTRL) != 0;

	switch (keycode) {
	case 14: /* Backspace */
		if (input->cursor_pos > 0) {
			int prev = utf8_prev(input->buf, input->cursor_pos);
			int rem = input->cursor_pos - prev;
			memmove(&input->buf[prev], &input->buf[input->cursor_pos],
				input->len - input->cursor_pos);
			input->len -= rem;
			input->cursor_pos = prev;
			input->buf[input->len] = '\0';
		}
		return true;

	case 111: /* Delete */
		if (input->cursor_pos < input->len) {
			int next = utf8_next(input->buf, input->len, input->cursor_pos);
			int rem = next - input->cursor_pos;
			memmove(&input->buf[input->cursor_pos],
				&input->buf[next], input->len - next);
			input->len -= rem;
			input->buf[input->len] = '\0';
		}
		return true;

	case 105: /* Left */
		if (input->cursor_pos > 0) {
			input->cursor_pos = utf8_prev(input->buf, input->cursor_pos);
		}
		return true;

	case 106: /* Right */
		if (input->cursor_pos < input->len) {
			input->cursor_pos = utf8_next(input->buf, input->len,
				input->cursor_pos);
		}
		return true;

	case 28: /* Enter */
		if (input->len > 0) {
			input->buf[input->len] = '\0';

			/* Route by prefix */
			if (strncmp(input->buf, "search ", 7) == 0 &&
					input->len > 7)
				feed_search(input->feed, input->buf + 7);
			else if (strncmp(input->buf, "find ", 5) == 0 &&
					input->len > 5)
				feed_search(input->feed, input->buf + 5);
			else if (strncmp(input->buf, "watch ", 6) == 0 &&
					input->len > 6)
				feed_create_watcher(input->feed,
					input->buf + 6);
			else
				feed_submit(input->feed, input->buf);

			input->len = 0;
			input->cursor_pos = 0;
			input->buf[0] = '\0';
		}
		return true;

	case 1: /* Escape */
		if (input->len > 0) {
			input->len = 0;
			input->cursor_pos = 0;
			input->buf[0] = '\0';
			return true;
		}
		return false;

	default:
		break;
	}

	/* Ctrl shortcuts */
	if (ctrl) {
		switch (keycode) {
		case 46: /* Ctrl+C */
			input->len = 0;
			input->cursor_pos = 0;
			input->buf[0] = '\0';
			return true;
		case 38: /* Ctrl+A — scancode for 'a' */
			input->cursor_pos = 0;
			return true;
		case 18: /* Ctrl+E — scancode for 'e' */
			input->cursor_pos = input->len;
			return true;
		case 38 + 12: /* Ctrl+L — scancode for 'l' = 50 */
			if (input->feed) {
				input->feed->scroll_offset = 0;
			}
			return true;
		}
		return false;
	}

	/* Printable text insertion */
	if (utf8_len > 0 && input->len + utf8_len < (int)sizeof(input->buf) - 1) {
		memmove(&input->buf[input->cursor_pos + utf8_len],
			&input->buf[input->cursor_pos],
			input->len - input->cursor_pos);
		memcpy(&input->buf[input->cursor_pos], utf8, utf8_len);
		input->cursor_pos += utf8_len;
		input->len += utf8_len;
		input->buf[input->len] = '\0';
		return true;
	}

	return false;
}

bool input_tick_cursor(struct leaves_input *input, uint32_t dt_ms) {
	input->cursor_blink_ms += dt_ms;
	if (input->cursor_blink_ms >= 530) {
		input->cursor_blink_ms -= 530;
		input->cursor_visible = !input->cursor_visible;
		return true;
	}
	return false;
}
