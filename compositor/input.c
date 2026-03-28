#include "input.h"
#include "feed.h"
#include <ctype.h>
#include <string.h>

void input_init(struct leaves_input *input, struct leaves_feed *feed) {
	memset(input, 0, sizeof(*input));
	input->cursor_visible = true;
	input->feed = feed;
	input->sel_anchor = -1;
}

/* UTF-8: byte offset of codepoint before pos */
static int utf8_prev(const char *buf, int pos) {
	if (pos <= 0) return 0;
	pos--;
	while (pos > 0 && (buf[pos] & 0xC0) == 0x80) {
		pos--;
	}
	return pos;
}

/* UTF-8: byte offset of codepoint after pos */
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
		if (input_delete_selection(input)) return true;
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
		if (input_delete_selection(input)) return true;
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
		if (input->sel_anchor != -1 && input->sel_anchor != input->sel_focus) {
			/* Collapse to left end of selection */
			int left = input->sel_anchor < input->sel_focus
				? input->sel_anchor : input->sel_focus;
			input->cursor_pos = left;
			input->sel_anchor = -1;
			return true;
		}
		if (input->cursor_pos > 0)
			input->cursor_pos = utf8_prev(input->buf, input->cursor_pos);
		return true;

	case 106: /* Right */
		if (input->sel_anchor != -1 && input->sel_anchor != input->sel_focus) {
			/* Collapse to right end of selection */
			int right = input->sel_anchor > input->sel_focus
				? input->sel_anchor : input->sel_focus;
			input->cursor_pos = right;
			input->sel_anchor = -1;
			return true;
		}
		if (input->cursor_pos < input->len)
			input->cursor_pos = utf8_next(input->buf, input->len,
				input->cursor_pos);
		return true;

	case 28: /* Enter */
		input->sel_anchor = -1;
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
		if (input->sel_anchor != -1) {
			/* First Escape clears selection, keeps text */
			input->sel_anchor = -1;
			return true;
		}
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
		case 38: /* Ctrl+L — move to start */
			input->cursor_pos = 0;
			return true;
		case 18: /* Ctrl+E — move to end */
			input->cursor_pos = input->len;
			return true;
		case 38 + 12: /* Ctrl+M — reset scroll = 50 */
			if (input->feed) {
				input->feed->scroll_offset = 0;
			}
			return true;
		}
		return false;
	}

	/* Printable text insertion: replace selection if active */
	if (utf8_len > 0) {
		input_delete_selection(input);
		if (input->len + utf8_len < (int)sizeof(input->buf) - 1) {
			memmove(&input->buf[input->cursor_pos + utf8_len],
				&input->buf[input->cursor_pos],
				input->len - input->cursor_pos);
			memcpy(&input->buf[input->cursor_pos], utf8, utf8_len);
			input->cursor_pos += utf8_len;
			input->len += utf8_len;
			input->buf[input->len] = '\0';
			return true;
		}
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

void input_paste(struct leaves_input *input, const char *text, int len) {
	/* Delete any active selection first */
	input_delete_selection(input);

	/* Build a clean copy: strip CR/LF and other control bytes, but
	 * preserve all UTF-8 continuation bytes so multi-byte chars survive. */
	char clean[1024];
	int clean_len = 0;
	int avail = (int)sizeof(input->buf) - 1 - input->len;
	if (avail <= 0) return;

	for (int i = 0; i < len && clean_len < avail; i++) {
		unsigned char c = (unsigned char)text[i];
		if (c == '\r' || c == '\n') continue;   /* drop newlines */
		if (c < 0x20)               continue;   /* drop other control chars */
		clean[clean_len++] = text[i];
	}
	if (clean_len == 0) return;

	/* Insert clean block at cursor position */
	memmove(&input->buf[input->cursor_pos + clean_len],
		&input->buf[input->cursor_pos],
		input->len - input->cursor_pos);
	memcpy(&input->buf[input->cursor_pos], clean, clean_len);
	input->cursor_pos += clean_len;
	input->len += clean_len;
	input->buf[input->len] = '\0';
}

void input_clear_selection(struct leaves_input *input) {
	input->sel_anchor = -1;
}

bool input_delete_selection(struct leaves_input *input) {
	if (input->sel_anchor == -1 || input->sel_anchor == input->sel_focus)
		return false;
	int start = input->sel_anchor < input->sel_focus
		? input->sel_anchor : input->sel_focus;
	int end = input->sel_anchor < input->sel_focus
		? input->sel_focus : input->sel_anchor;
	memmove(&input->buf[start], &input->buf[end], input->len - end);
	input->len -= (end - start);
	input->buf[input->len] = '\0';
	input->cursor_pos = start;
	input->sel_anchor = -1;
	return true;
}

void input_select_word(struct leaves_input *input, int byte_pos) {
	if (input->len == 0) return;
	if (byte_pos > input->len) byte_pos = input->len;

	/* Scan left to find word start */
	int start = byte_pos;
	while (start > 0) {
		int prev = utf8_prev(input->buf, start);
		unsigned char c = (unsigned char)input->buf[prev];
		if (!isalnum(c) && c != '_' && c != '-') break;
		start = prev;
	}
	/* Scan right to find word end */
	int end = byte_pos;
	while (end < input->len) {
		unsigned char c = (unsigned char)input->buf[end];
		if (!isalnum(c) && c != '_' && c != '-') break;
		end = utf8_next(input->buf, input->len, end);
	}

	if (start < end) {
		input->sel_anchor = start;
		input->sel_focus  = end;
		input->cursor_pos = end;
	}
}
