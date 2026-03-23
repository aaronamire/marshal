#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#include <wayland-server-core.h>
#include <wlr/backend.h>
#include <wlr/render/allocator.h>
#include <wlr/render/wlr_renderer.h>
#include <wlr/render/pass.h>
#include <wlr/types/wlr_output.h>
#include <wlr/types/wlr_output_layout.h>
#include <wlr/types/wlr_keyboard.h>
#include <wlr/types/wlr_seat.h>
#include <wlr/types/wlr_input_device.h>
#include <wlr/util/log.h>
#include <xkbcommon/xkbcommon.h>
#include <drm_fourcc.h>

#include "renderer.h"
#include "feed.h"
#include "input.h"

struct leaves_output {
	struct wl_list link;
	struct leaves_server *server;
	struct wlr_output *wlr_output;
	struct wl_listener frame;
	struct wl_listener request_state;
	struct wl_listener destroy;
};

struct leaves_keyboard {
	struct leaves_server *server;
	struct wlr_keyboard *wlr_keyboard;
	struct wl_listener key;
	struct wl_listener modifiers;
	struct wl_listener destroy;
};

struct leaves_server {
	struct wl_display *display;
	struct wl_event_loop *event_loop;
	struct wlr_backend *backend;
	struct wlr_renderer *renderer;
	struct wlr_allocator *allocator;
	struct wlr_output_layout *output_layout;
	struct wl_list outputs;
	struct wlr_seat *seat;

	struct leaves_feed *feed;
	struct leaves_input input;
	struct leaves_renderer *lrenderer;

	bool needs_redraw;
	struct wl_listener new_output;
	struct wl_listener new_input;
	struct wl_listener backend_destroy;

	/* Cursor blink timer */
	struct wl_event_source *cursor_timer;
	/* Animation timer */
	struct wl_event_source *anim_timer;

	/* xkb state for modifier tracking */
	uint32_t modifiers;

	/* Pending animation timer tracking */
	struct timespec last_frame;
};

/* ── Forward declarations ── */
static void schedule_frame(struct leaves_server *server);

/* ── Cursor blink timer ── */

static int cursor_timer_cb(void *data) {
	struct leaves_server *server = data;
	if (input_tick_cursor(&server->input, 530)) {
		server->needs_redraw = true;
		schedule_frame(server);
	}
	wl_event_source_timer_update(server->cursor_timer, 530);
	return 0;
}

/* ── Animation timer ── */

static int anim_timer_cb(void *data) {
	struct leaves_server *server = data;
	float dt = 1.0f / 60.0f; /* ~16ms */
	if (feed_animate(server->feed, dt)) {
		server->needs_redraw = true;
		schedule_frame(server);
		wl_event_source_timer_update(server->anim_timer, 16);
	}
	return 0;
}

/* ── Wakeup pipe event source ── */

static int wakeup_handler(int fd, uint32_t mask __attribute__((unused)),
		void *data) {
	struct leaves_server *server = data;
	char byte;
	while (read(fd, &byte, 1) == 1) {
		switch (byte) {
		case 'c':
			/* Confirmation overlay needed — already set by feed thread */
			break;
		case 'u':
			/* Card state updated */
			break;
		default:
			break;
		}
	}
	server->needs_redraw = true;
	schedule_frame(server);
	wl_event_source_timer_update(server->anim_timer, 16);
	return 0;
}

/* ── Output frame handler ── */

static void output_frame(struct wl_listener *listener, void *data) {
	struct leaves_output *output = wl_container_of(listener, output, frame);
	struct leaves_server *server = output->server;

	if (!server->needs_redraw) return;
	server->needs_redraw = false;

	int width, height;
	wlr_output_effective_resolution(output->wlr_output, &width, &height);

	/* Ensure renderer surface matches output size */
	if (server->lrenderer->width != width ||
			server->lrenderer->height != height) {
		renderer_resize(server->lrenderer, width, height);
	}

	/* Draw frame with cairo */
	int stride;
	unsigned char *pixels = renderer_draw_frame(server->lrenderer,
		server->feed, &server->input, &stride);
	if (!pixels) return;

	/* Upload to wlroots texture */
	struct wlr_texture *texture = wlr_texture_from_pixels(server->renderer,
		DRM_FORMAT_ARGB8888, stride, width, height, pixels);
	if (!texture) return;

	/* Begin render pass */
	struct wlr_output_state state;
	wlr_output_state_init(&state);

	int buffer_age;
	struct wlr_render_pass *pass = wlr_output_begin_render_pass(
		output->wlr_output, &state, &buffer_age, NULL);
	if (!pass) {
		wlr_texture_destroy(texture);
		wlr_output_state_finish(&state);
		return;
	}

	/* Draw the full-screen texture */
	wlr_render_pass_add_texture(pass, &(struct wlr_render_texture_options){
		.texture = texture,
		.dst_box = { .x = 0, .y = 0, .width = width, .height = height },
	});

	wlr_render_pass_submit(pass);
	wlr_output_commit_state(output->wlr_output, &state);
	wlr_output_state_finish(&state);
	wlr_texture_destroy(texture);
}

/* Accept resize/fullscreen requests from the parent compositor (Hyprland).
 * Without this, the wayland backend ignores configure events and the
 * window stays at its initial size. */
static void output_request_state(struct wl_listener *listener, void *data) {
	struct leaves_output *output =
		wl_container_of(listener, output, request_state);
	const struct wlr_output_event_request_state *event = data;
	wlr_output_commit_state(output->wlr_output, event->state);
	output->server->needs_redraw = true;
	schedule_frame(output->server);
}

static void output_destroy(struct wl_listener *listener, void *data) {
	struct leaves_output *output = wl_container_of(listener, output, destroy);
	wl_list_remove(&output->frame.link);
	wl_list_remove(&output->request_state.link);
	wl_list_remove(&output->destroy.link);
	wl_list_remove(&output->link);
	free(output);
}

static void schedule_frame(struct leaves_server *server) {
	struct leaves_output *output;
	wl_list_for_each(output, &server->outputs, link) {
		wlr_output_schedule_frame(output->wlr_output);
	}
}

/* ── Keyboard handling ── */

static void keyboard_handle_key(struct wl_listener *listener, void *data) {
	struct leaves_keyboard *keyboard =
		wl_container_of(listener, keyboard, key);
	struct leaves_server *server = keyboard->server;
	struct wlr_keyboard_key_event *event = data;

	if (event->state != WL_KEYBOARD_KEY_STATE_PRESSED) return;

	uint32_t keycode = event->keycode + 8;
	const xkb_keysym_t *syms;
	int nsyms = xkb_state_key_get_syms(
		keyboard->wlr_keyboard->xkb_state, keycode, &syms);

	/* Get UTF-8 text for printable keys */
	char utf8[8] = {0};
	int utf8_len = xkb_state_key_get_utf8(
		keyboard->wlr_keyboard->xkb_state, keycode, utf8, sizeof(utf8));

	/* Check for non-printable / control characters */
	bool is_printable = utf8_len > 0 && (unsigned char)utf8[0] >= 0x20;

	if (input_handle_key(&server->input, event->keycode,
			server->modifiers,
			is_printable ? utf8 : NULL,
			is_printable ? utf8_len : 0)) {
		server->needs_redraw = true;
		schedule_frame(server);
		/* Start animation timer for new submissions */
		wl_event_source_timer_update(server->anim_timer, 16);
	}
}

static void keyboard_handle_modifiers(struct wl_listener *listener,
		void *data) {
	struct leaves_keyboard *keyboard =
		wl_container_of(listener, keyboard, modifiers);
	struct leaves_server *server = keyboard->server;

	xkb_mod_mask_t mod_mask = xkb_state_serialize_mods(
		keyboard->wlr_keyboard->xkb_state,
		XKB_STATE_MODS_EFFECTIVE);

	server->modifiers = 0;

	struct xkb_keymap *keymap = xkb_state_get_keymap(
		keyboard->wlr_keyboard->xkb_state);
	xkb_mod_index_t ctrl_idx = xkb_keymap_mod_get_index(keymap,
		XKB_MOD_NAME_CTRL);
	if (ctrl_idx != XKB_MOD_INVALID && (mod_mask & (1 << ctrl_idx))) {
		server->modifiers |= (1 << 0); /* MOD_CTRL */
	}
}

static void keyboard_destroy(struct wl_listener *listener, void *data) {
	struct leaves_keyboard *keyboard =
		wl_container_of(listener, keyboard, destroy);
	wl_list_remove(&keyboard->key.link);
	wl_list_remove(&keyboard->modifiers.link);
	wl_list_remove(&keyboard->destroy.link);
	free(keyboard);
}

/* ── New input device ── */

static void server_new_input(struct wl_listener *listener, void *data) {
	struct leaves_server *server =
		wl_container_of(listener, server, new_input);
	struct wlr_input_device *device = data;

	if (device->type != WLR_INPUT_DEVICE_KEYBOARD) return;

	struct leaves_keyboard *keyboard = calloc(1, sizeof(*keyboard));
	keyboard->server = server;
	keyboard->wlr_keyboard = wlr_keyboard_from_input_device(device);

	struct xkb_context *context = xkb_context_new(XKB_CONTEXT_NO_FLAGS);
	struct xkb_keymap *keymap = xkb_keymap_new_from_names(context, NULL,
		XKB_KEYMAP_COMPILE_NO_FLAGS);
	wlr_keyboard_set_keymap(keyboard->wlr_keyboard, keymap);
	xkb_keymap_unref(keymap);
	xkb_context_unref(context);
	wlr_keyboard_set_repeat_info(keyboard->wlr_keyboard, 25, 600);

	keyboard->key.notify = keyboard_handle_key;
	wl_signal_add(&keyboard->wlr_keyboard->events.key, &keyboard->key);

	keyboard->modifiers.notify = keyboard_handle_modifiers;
	wl_signal_add(&keyboard->wlr_keyboard->events.modifiers,
		&keyboard->modifiers);

	keyboard->destroy.notify = keyboard_destroy;
	wl_signal_add(&device->events.destroy, &keyboard->destroy);

	wlr_seat_set_keyboard(server->seat, keyboard->wlr_keyboard);
}

/* ── New output ── */

static void server_new_output(struct wl_listener *listener, void *data) {
	struct leaves_server *server =
		wl_container_of(listener, server, new_output);
	struct wlr_output *wlr_output = data;

	/* Set preferred mode */
	wlr_output_init_render(wlr_output, server->allocator, server->renderer);

	struct wlr_output_state state;
	wlr_output_state_init(&state);
	wlr_output_state_set_enabled(&state, true);

	struct wlr_output_mode *mode = wlr_output_preferred_mode(wlr_output);
	if (mode) {
		wlr_output_state_set_mode(&state, mode);
	}

	wlr_output_commit_state(wlr_output, &state);
	wlr_output_state_finish(&state);

	struct leaves_output *output = calloc(1, sizeof(*output));
	output->server = server;
	output->wlr_output = wlr_output;

	output->frame.notify = output_frame;
	wl_signal_add(&wlr_output->events.frame, &output->frame);

	output->request_state.notify = output_request_state;
	wl_signal_add(&wlr_output->events.request_state,
		&output->request_state);

	output->destroy.notify = output_destroy;
	wl_signal_add(&wlr_output->events.destroy, &output->destroy);

	wl_list_insert(&server->outputs, &output->link);
	wlr_output_layout_add_auto(server->output_layout, wlr_output);

	/* Initial render */
	server->needs_redraw = true;
}

/* ── Backend destroy ── */

static void backend_destroy_handler(struct wl_listener *listener, void *data) {
	struct leaves_server *server =
		wl_container_of(listener, server, backend_destroy);
	wl_display_terminate(server->display);
}

/* ── Main ── */

int main(int argc, char *argv[]) {
	wlr_log_init(WLR_DEBUG, NULL);

	struct leaves_server server = {0};
	wl_list_init(&server.outputs);

	server.display = wl_display_create();
	server.event_loop = wl_display_get_event_loop(server.display);

	server.backend = wlr_backend_autocreate(server.event_loop, NULL);
	if (!server.backend) {
		fprintf(stderr, "Failed to create wlr_backend\n");
		return 1;
	}

	server.renderer = wlr_renderer_autocreate(server.backend);
	if (!server.renderer) {
		fprintf(stderr, "Failed to create wlr_renderer\n");
		return 1;
	}
	wlr_renderer_init_wl_display(server.renderer, server.display);

	server.allocator = wlr_allocator_autocreate(server.backend,
		server.renderer);
	if (!server.allocator) {
		fprintf(stderr, "Failed to create wlr_allocator\n");
		return 1;
	}

	server.output_layout = wlr_output_layout_create(server.display);

	server.new_output.notify = server_new_output;
	wl_signal_add(&server.backend->events.new_output, &server.new_output);

	server.new_input.notify = server_new_input;
	wl_signal_add(&server.backend->events.new_input, &server.new_input);

	server.backend_destroy.notify = backend_destroy_handler;
	wl_signal_add(&server.backend->events.destroy, &server.backend_destroy);

	server.seat = wlr_seat_create(server.display, "seat0");

	/* Feed + Input + Renderer */
	server.feed = feed_create("http://127.0.0.1:8765");
	input_init(&server.input, server.feed);
	server.lrenderer = renderer_create();

	/* Load history (non-blocking — runs on main thread but with timeout) */
	feed_load_history(server.feed);

	/* Wakeup pipe for HTTP thread notifications */
	int flags = fcntl(server.feed->wakeup_pipe[0], F_GETFL);
	fcntl(server.feed->wakeup_pipe[0], F_SETFL, flags | O_NONBLOCK);
	wl_event_loop_add_fd(server.event_loop, server.feed->wakeup_pipe[0],
		WL_EVENT_READABLE, wakeup_handler, &server);

	/* Cursor blink timer: 530ms */
	server.cursor_timer = wl_event_loop_add_timer(server.event_loop,
		cursor_timer_cb, &server);
	wl_event_source_timer_update(server.cursor_timer, 530);

	/* Animation timer: runs at 60fps when active */
	server.anim_timer = wl_event_loop_add_timer(server.event_loop,
		anim_timer_cb, &server);

	/* Start backend */
	if (!wlr_backend_start(server.backend)) {
		fprintf(stderr, "Failed to start backend\n");
		wlr_backend_destroy(server.backend);
		wl_display_destroy(server.display);
		return 1;
	}

	fprintf(stderr, "Leaves compositor running\n");
	wl_display_run(server.display);

	/* Cleanup */
	wl_display_destroy_clients(server.display);
	feed_destroy(server.feed);
	renderer_destroy(server.lrenderer);
	wl_display_destroy(server.display);

	return 0;
}
