#include "feed.h"
#include <curl/curl.h>
#include <cjson/cJSON.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct curl_buf {
	char *data;
	size_t len;
};

static size_t curl_write_cb(void *ptr, size_t size, size_t nmemb, void *ud) {
	struct curl_buf *buf = ud;
	size_t total = size * nmemb;
	char *tmp = realloc(buf->data, buf->len + total + 1);
	if (!tmp) return 0;
	buf->data = tmp;
	memcpy(buf->data + buf->len, ptr, total);
	buf->len += total;
	buf->data[buf->len] = '\0';
	return total;
}

/* ── JSON escape for request bodies ── */

static void json_escape(const char *src, char *dst, size_t dst_sz) {
	size_t j = 0;
	for (size_t i = 0; src[i] && j < dst_sz - 2; i++) {
		switch (src[i]) {
		case '"':  if (j + 2 < dst_sz) { dst[j++] = '\\'; dst[j++] = '"'; } break;
		case '\\': if (j + 2 < dst_sz) { dst[j++] = '\\'; dst[j++] = '\\'; } break;
		case '\n': if (j + 2 < dst_sz) { dst[j++] = '\\'; dst[j++] = 'n'; } break;
		case '\r': if (j + 2 < dst_sz) { dst[j++] = '\\'; dst[j++] = 'r'; } break;
		case '\t': if (j + 2 < dst_sz) { dst[j++] = '\\'; dst[j++] = 't'; } break;
		default:   dst[j++] = src[i]; break;
		}
	}
	dst[j] = '\0';
}

/* ── Parse history/result response into card ── */

static void parse_intent_response(const cJSON *obj, LeavesIntent *intent) {
	const cJSON *id = cJSON_GetObjectItem(obj, "intent_id");
	if (cJSON_IsString(id))
		snprintf(intent->intent_id, sizeof(intent->intent_id),
			"%s", id->valuestring);

	const cJSON *text = cJSON_GetObjectItem(obj, "natural_text");
	if (!text) text = cJSON_GetObjectItem(obj, "text");
	if (cJSON_IsString(text))
		snprintf(intent->natural_text, sizeof(intent->natural_text),
			"%s", text->valuestring);

	/* Build action chain from actions array */
	const cJSON *actions = cJSON_GetObjectItem(obj, "actions");
	if (cJSON_IsArray(actions)) {
		char chain[256] = {0};
		int off = 0;
		int i = 0;
		const cJSON *action;
		cJSON_ArrayForEach(action, actions) {
			if (i > 0 && off < (int)sizeof(chain) - 4)
				off += snprintf(chain + off, sizeof(chain) - off,
					" → ");
			const cJSON *type = cJSON_GetObjectItem(action, "type");
			const cJSON *params = cJSON_GetObjectItem(action, "params");
			const char *path = NULL;
			if (params) {
				const cJSON *p = cJSON_GetObjectItem(params, "path");
				if (!p) p = cJSON_GetObjectItem(params, "source");
				if (cJSON_IsString(p)) path = p->valuestring;
			}
			if (cJSON_IsString(type)) {
				if (path)
					off += snprintf(chain + off,
						sizeof(chain) - off,
						"%s %s", type->valuestring, path);
				else
					off += snprintf(chain + off,
						sizeof(chain) - off,
						"%s", type->valuestring);
			}
			i++;
		}
		snprintf(intent->action_chain, sizeof(intent->action_chain),
			"%s", chain);
	}

	/* Capability scope from authorization.resources */
	const cJSON *auth = cJSON_GetObjectItem(obj, "authorization");
	if (auth) {
		const cJSON *resources = cJSON_GetObjectItem(auth, "resources");
		if (cJSON_IsArray(resources)) {
			char scope[512] = {0};
			int off = 0;
			const cJSON *res;
			cJSON_ArrayForEach(res, resources) {
				if (cJSON_IsString(res)) {
					if (off > 0 && off < (int)sizeof(scope) - 4)
						off += snprintf(scope + off,
							sizeof(scope) - off, " · ");
					off += snprintf(scope + off,
						sizeof(scope) - off,
						"%s", res->valuestring);
				}
			}
			if (off > 60) {
				scope[57] = '.'; scope[58] = '.';
				scope[59] = '.'; scope[60] = '\0';
			}
			snprintf(intent->capability_scope,
				sizeof(intent->capability_scope), "%s", scope);
		}

		const cJSON *preview = cJSON_GetObjectItem(auth,
			"preview_required");
		if (cJSON_IsBool(preview))
			intent->preview_required = cJSON_IsTrue(preview);

		const cJSON *rev = cJSON_GetObjectItem(auth, "reversible");
		if (cJSON_IsBool(rev))
			intent->reversible = cJSON_IsTrue(rev);
	}

	const cJSON *dur = cJSON_GetObjectItem(obj, "duration_ms");
	if (cJSON_IsNumber(dur))
		intent->duration_ms = dur->valuedouble;

	const cJSON *status = cJSON_GetObjectItem(obj, "status");
	if (cJSON_IsString(status)) {
		if (strcmp(status->valuestring, "failed") == 0)
			intent->state = CARD_STATE_FAILED;
		else if (strcmp(status->valuestring, "done") == 0)
			intent->state = CARD_STATE_DONE;
		else if (strcmp(status->valuestring, "pending") == 0)
			intent->state = CARD_STATE_PENDING;
	}
}

/* ── Parse plan response — populate authorization fields ── */

static void parse_plan_response(const cJSON *obj, LeavesIntent *intent) {
	const cJSON *id = cJSON_GetObjectItem(obj, "intent_id");
	if (cJSON_IsString(id))
		snprintf(intent->intent_id, sizeof(intent->intent_id),
			"%s", id->valuestring);

	const cJSON *text = cJSON_GetObjectItem(obj, "natural_text");
	if (cJSON_IsString(text))
		snprintf(intent->natural_text, sizeof(intent->natural_text),
			"%s", text->valuestring);

	/* Build actions_summary from actions array */
	const cJSON *actions = cJSON_GetObjectItem(obj, "actions");
	if (cJSON_IsArray(actions)) {
		char summary[512] = {0};
		int off = 0;
		const cJSON *action;
		cJSON_ArrayForEach(action, actions) {
			const cJSON *type = cJSON_GetObjectItem(action, "type");
			const cJSON *params = cJSON_GetObjectItem(action, "params");
			const char *path = NULL;
			if (params) {
				const cJSON *p = cJSON_GetObjectItem(params, "path");
				if (!p) p = cJSON_GetObjectItem(params, "source");
				if (cJSON_IsString(p)) path = p->valuestring;
			}
			if (cJSON_IsString(type)) {
				if (off > 0 && off < (int)sizeof(summary) - 4)
					off += snprintf(summary + off,
						sizeof(summary) - off, "\n");
				if (path)
					off += snprintf(summary + off,
						sizeof(summary) - off,
						"%s %s", type->valuestring, path);
				else
					off += snprintf(summary + off,
						sizeof(summary) - off,
						"%s", type->valuestring);
			}
		}
		snprintf(intent->actions_summary,
			sizeof(intent->actions_summary), "%s", summary);

		/* Also build action_chain for card display */
		char chain[256] = {0};
		off = 0;
		int i = 0;
		cJSON_ArrayForEach(action, actions) {
			if (i > 0 && off < (int)sizeof(chain) - 4)
				off += snprintf(chain + off, sizeof(chain) - off,
					" → ");
			const cJSON *type = cJSON_GetObjectItem(action, "type");
			const cJSON *params = cJSON_GetObjectItem(action, "params");
			const char *path = NULL;
			if (params) {
				const cJSON *p = cJSON_GetObjectItem(params, "path");
				if (!p) p = cJSON_GetObjectItem(params, "source");
				if (cJSON_IsString(p)) path = p->valuestring;
			}
			if (cJSON_IsString(type)) {
				if (path)
					off += snprintf(chain + off,
						sizeof(chain) - off,
						"%s %s", type->valuestring, path);
				else
					off += snprintf(chain + off,
						sizeof(chain) - off,
						"%s", type->valuestring);
			}
			i++;
		}
		snprintf(intent->action_chain, sizeof(intent->action_chain),
			"%s", chain);
	}

	/* Authorization */
	const cJSON *auth = cJSON_GetObjectItem(obj, "authorization");
	if (auth) {
		const cJSON *preview = cJSON_GetObjectItem(auth,
			"preview_required");
		if (cJSON_IsBool(preview))
			intent->preview_required = cJSON_IsTrue(preview);

		const cJSON *rev = cJSON_GetObjectItem(auth, "reversible");
		if (cJSON_IsBool(rev))
			intent->reversible = cJSON_IsTrue(rev);

		const cJSON *resources = cJSON_GetObjectItem(auth, "resources");
		if (cJSON_IsArray(resources)) {
			char res_str[512] = {0};
			int off = 0;
			const cJSON *res;
			cJSON_ArrayForEach(res, resources) {
				if (cJSON_IsString(res)) {
					if (off > 0 && off < (int)sizeof(res_str) - 4)
						off += snprintf(res_str + off,
							sizeof(res_str) - off,
							" · ");
					off += snprintf(res_str + off,
						sizeof(res_str) - off,
						"%s", res->valuestring);
				}
			}
			snprintf(intent->resources, sizeof(intent->resources),
				"%s", res_str);
			snprintf(intent->capability_scope,
				sizeof(intent->capability_scope),
				"%s", res_str);
		}
	}
}

/* ── HTTP helpers ── */

static cJSON *http_get(const char *url, long timeout) {
	CURL *curl = curl_easy_init();
	if (!curl) return NULL;

	struct curl_buf buf = {0};
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);

	CURLcode res = curl_easy_perform(curl);
	curl_easy_cleanup(curl);

	cJSON *result = NULL;
	if (res == CURLE_OK && buf.data)
		result = cJSON_Parse(buf.data);
	free(buf.data);
	return result;
}

static cJSON *http_post(const char *url, const char *body, long timeout) {
	CURL *curl = curl_easy_init();
	if (!curl) return NULL;

	struct curl_slist *slist = NULL;
	slist = curl_slist_append(slist, "Content-Type: application/json");

	struct curl_buf buf = {0};
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body);
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, slist);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout);

	CURLcode res = curl_easy_perform(curl);
	curl_slist_free_all(slist);
	curl_easy_cleanup(curl);

	cJSON *result = NULL;
	if (res == CURLE_OK && buf.data)
		result = cJSON_Parse(buf.data);
	free(buf.data);
	return result;
}

/* ── Feed lifecycle ── */

struct leaves_feed *feed_create(const char *api_base) {
	struct leaves_feed *feed = calloc(1, sizeof(*feed));
	if (!feed) return NULL;

	pthread_mutex_init(&feed->mutex, NULL);
	if (pipe(feed->wakeup_pipe) != 0) {
		free(feed);
		return NULL;
	}
	if (pipe(feed->confirm_pipe) != 0) {
		close(feed->wakeup_pipe[0]);
		close(feed->wakeup_pipe[1]);
		free(feed);
		return NULL;
	}
	snprintf(feed->api_base, sizeof(feed->api_base), "%s", api_base);
	return feed;
}

void feed_destroy(struct leaves_feed *feed) {
	if (!feed) return;
	close(feed->wakeup_pipe[0]);
	close(feed->wakeup_pipe[1]);
	close(feed->confirm_pipe[0]);
	close(feed->confirm_pipe[1]);
	pthread_mutex_destroy(&feed->mutex);
	free(feed);
}

void feed_load_history(struct leaves_feed *feed) {
	CURL *curl = curl_easy_init();
	if (!curl) return;

	char url[512];
	snprintf(url, sizeof(url), "%s/v1/history?limit=50", feed->api_base);

	struct curl_buf buf = {0};
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 2L);

	CURLcode res = curl_easy_perform(curl);
	curl_easy_cleanup(curl);

	if (res != CURLE_OK || !buf.data) {
		free(buf.data);
		return;
	}

	cJSON *root = cJSON_Parse(buf.data);
	free(buf.data);
	if (!root) return;

	const cJSON *intents_arr = cJSON_GetObjectItem(root, "intents");
	if (!intents_arr) intents_arr = root;

	if (cJSON_IsArray(intents_arr)) {
		pthread_mutex_lock(&feed->mutex);
		const cJSON *item;
		cJSON_ArrayForEach(item, intents_arr) {
			if (feed->count >= MAX_INTENTS) break;
			LeavesIntent *intent = &feed->intents[feed->count];
			memset(intent, 0, sizeof(*intent));
			parse_intent_response(item, intent);
			intent->state = CARD_STATE_HISTORY;
			/* No animation for history items */
			spring_init(&intent->anim_y, 0, 0);
			spring_init(&intent->anim_opacity, 1.0f, 1.0f);
			intent->anim_y.pos = intent->anim_y.target;
			intent->anim_opacity.pos = intent->anim_opacity.target;
			feed->count++;
		}
		pthread_mutex_unlock(&feed->mutex);
	}

	cJSON_Delete(root);
}

/* ── Briefing fetch ── */

void feed_load_briefing(struct leaves_feed *feed) {
	CURL *curl = curl_easy_init();
	if (!curl) return;

	char url[512];
	snprintf(url, sizeof(url), "%s/v1/cortex/briefing?hours=12",
		feed->api_base);

	struct curl_buf buf = {0};
	curl_easy_setopt(curl, CURLOPT_URL, url);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
	curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);

	CURLcode res = curl_easy_perform(curl);
	curl_easy_cleanup(curl);

	if (res != CURLE_OK || !buf.data) {
		free(buf.data);
		return;
	}

	cJSON *root = cJSON_Parse(buf.data);
	free(buf.data);
	if (!root) return;

	pthread_mutex_lock(&feed->mutex);
	LeavesBriefing *b = &feed->briefing;
	memset(b, 0, sizeof(*b));

	const cJSON *headline = cJSON_GetObjectItem(root, "headline");
	if (cJSON_IsString(headline))
		snprintf(b->headline, sizeof(b->headline),
			"%s", headline->valuestring);

	const cJSON *total = cJSON_GetObjectItem(root, "total_changes");
	if (cJSON_IsNumber(total))
		b->total_changes = (int)total->valuedouble;

	const cJSON *hours = cJSON_GetObjectItem(root, "period_hours");
	if (cJSON_IsNumber(hours))
		b->period_hours = (int)hours->valuedouble;

	const cJSON *empty = cJSON_GetObjectItem(root, "empty");
	if (cJSON_IsBool(empty))
		b->empty = cJSON_IsTrue(empty);

	const cJSON *sections = cJSON_GetObjectItem(root, "sections");
	if (cJSON_IsArray(sections)) {
		const cJSON *sec;
		cJSON_ArrayForEach(sec, sections) {
			if (b->section_count >= MAX_BRIEFING_SECTIONS) break;
			LeavesBriefingSection *s =
				&b->sections[b->section_count];
			memset(s, 0, sizeof(*s));

			const cJSON *st = cJSON_GetObjectItem(sec,
				"source_type");
			if (cJSON_IsString(st))
				snprintf(s->source_type,
					sizeof(s->source_type),
					"%s", st->valuestring);

			const cJSON *cnt = cJSON_GetObjectItem(sec, "count");
			if (cJSON_IsNumber(cnt))
				s->count = (int)cnt->valuedouble;

			const cJSON *groups = cJSON_GetObjectItem(sec,
				"groups");
			if (cJSON_IsArray(groups)) {
				const cJSON *grp;
				cJSON_ArrayForEach(grp, groups) {
					if (s->group_count >=
							MAX_BRIEFING_GROUPS)
						break;
					LeavesBriefingGroup *g =
						&s->groups[s->group_count];
					memset(g, 0, sizeof(*g));

					const cJSON *dir =
						cJSON_GetObjectItem(grp,
							"directory");
					if (cJSON_IsString(dir))
						snprintf(g->directory,
							sizeof(g->directory),
							"%s",
							dir->valuestring);

					const cJSON *gc =
						cJSON_GetObjectItem(grp,
							"count");
					if (cJSON_IsNumber(gc))
						g->count =
							(int)gc->valuedouble;

					const cJSON *items =
						cJSON_GetObjectItem(grp,
							"items");
					if (cJSON_IsArray(items)) {
						const cJSON *it;
						cJSON_ArrayForEach(it, items) {
							if (g->item_count >=
								MAX_BRIEFING_ITEMS)
								break;
							LeavesBriefingItem *bi =
								&g->items[
								g->item_count];
							const cJSON *title =
								cJSON_GetObjectItem(
									it,
									"title");
							if (cJSON_IsString(
								title))
								snprintf(
								bi->title,
								sizeof(
								bi->title),
								"%s",
								title->
								valuestring);
							const cJSON *path =
								cJSON_GetObjectItem(
									it,
									"path");
							if (cJSON_IsString(
								path))
								snprintf(
								bi->path,
								sizeof(
								bi->path),
								"%s",
								path->
								valuestring);
							g->item_count++;
						}
					}
					s->group_count++;
				}
			}
			b->section_count++;
		}
	}

	b->loaded = true;
	spring_init(&b->anim_opacity, 0.0f, 1.0f);
	/* Slower, more deliberate entrance for briefing */
	b->anim_opacity.stiffness = 120.0f;
	b->anim_opacity.damping = 18.0f;
	pthread_mutex_unlock(&feed->mutex);

	cJSON_Delete(root);

	/* Signal main thread to redraw */
	char byte = 'u';
	(void)write(feed->wakeup_pipe[1], &byte, 1);
}

/* ── Two-phase submit thread ── */

struct submit_args {
	struct leaves_feed *feed;
	int index;
	char text[512];
};

static void *submit_thread(void *arg) {
	struct submit_args *sa = arg;
	struct leaves_feed *feed = sa->feed;

	/* Phase 1: POST /v1/intent/plan */
	char plan_url[512];
	snprintf(plan_url, sizeof(plan_url), "%s/v1/intent/plan",
		feed->api_base);

	char escaped[1024];
	json_escape(sa->text, escaped, sizeof(escaped));
	char body[1200];
	snprintf(body, sizeof(body), "{\"text\":\"%s\"}", escaped);

	cJSON *plan = http_post(plan_url, body, 180L);

	pthread_mutex_lock(&feed->mutex);
	LeavesIntent *intent = &feed->intents[sa->index];

	if (!plan) {
		intent->state = CARD_STATE_FAILED;
		pthread_mutex_unlock(&feed->mutex);
		char byte = 'u';
		(void)write(feed->wakeup_pipe[1], &byte, 1);
		free(sa);
		return NULL;
	}

	/* Check plan response status */
	const cJSON *status = cJSON_GetObjectItem(plan, "status");
	if (!cJSON_IsString(status) ||
			strcmp(status->valuestring, "planned") != 0) {
		/* Error or not_implemented — show message in card */
		const cJSON *msg = cJSON_GetObjectItem(plan, "message");
		if (!msg) msg = cJSON_GetObjectItem(plan, "detail");
		if (cJSON_IsString(msg))
			snprintf(intent->action_chain,
				sizeof(intent->action_chain),
				"%s", msg->valuestring);
		intent->state = CARD_STATE_FAILED;
		pthread_mutex_unlock(&feed->mutex);
		cJSON_Delete(plan);
		char byte = 'u';
		(void)write(feed->wakeup_pipe[1], &byte, 1);
		free(sa);
		return NULL;
	}

	/* Populate card from plan response */
	char nat[512];
	snprintf(nat, sizeof(nat), "%s", intent->natural_text);
	parse_plan_response(plan, intent);
	if (intent->natural_text[0] == '\0')
		snprintf(intent->natural_text,
			sizeof(intent->natural_text), "%s", nat);

	if (intent->preview_required) {
		intent->state = CARD_STATE_AWAITING_CONFIRM;
		feed->awaiting_confirm = true;
		feed->confirm_card_idx = sa->index;
		pthread_mutex_unlock(&feed->mutex);

		/* Signal main thread: overlay needed */
		char byte = 'c';
		(void)write(feed->wakeup_pipe[1], &byte, 1);

		/* Block until user decides */
		char decision = 'n';
		(void)read(feed->confirm_pipe[0], &decision, 1);

		if (decision == 'n') {
			pthread_mutex_lock(&feed->mutex);
			intent->state = CARD_STATE_CANCELLED;
			pthread_mutex_unlock(&feed->mutex);
			cJSON_Delete(plan);
			byte = 'u';
			(void)write(feed->wakeup_pipe[1], &byte, 1);
			free(sa);
			return NULL;
		}

		/* User confirmed — proceed to execute */
		pthread_mutex_lock(&feed->mutex);
		intent->state = CARD_STATE_EXECUTING;
		pthread_mutex_unlock(&feed->mutex);
		char ubyte = 'u';
		(void)write(feed->wakeup_pipe[1], &ubyte, 1);
	} else {
		intent->state = CARD_STATE_EXECUTING;
		pthread_mutex_unlock(&feed->mutex);
	}

	/* Phase 2: POST /v1/intent/execute */
	char exec_url[512];
	snprintf(exec_url, sizeof(exec_url), "%s/v1/intent/execute",
		feed->api_base);

	char exec_body[256];
	snprintf(exec_body, sizeof(exec_body),
		"{\"intent_id\":\"%s\",\"confirmed\":true}",
		intent->intent_id);

	cJSON_Delete(plan);

	cJSON *result = http_post(exec_url, exec_body, 180L);

	pthread_mutex_lock(&feed->mutex);

	if (result) {
		const cJSON *rst = cJSON_GetObjectItem(result, "status");
		if (cJSON_IsString(rst) &&
				strcmp(rst->valuestring, "done") == 0) {
			/* Update card with execution results */
			const cJSON *dur = cJSON_GetObjectItem(result,
				"duration_ms");
			if (cJSON_IsNumber(dur))
				intent->duration_ms = dur->valuedouble;

			/* Re-parse action chain and scope from result */
			const cJSON *actions = cJSON_GetObjectItem(result,
				"actions");
			if (cJSON_IsArray(actions)) {
				char chain[256] = {0};
				int off = 0, i = 0;
				const cJSON *action;
				cJSON_ArrayForEach(action, actions) {
					if (i > 0 && off < (int)sizeof(chain) - 4)
						off += snprintf(chain + off,
							sizeof(chain) - off,
							" → ");
					const cJSON *type = cJSON_GetObjectItem(
						action, "type");
					const cJSON *params = cJSON_GetObjectItem(
						action, "params");
					const char *path = NULL;
					if (params) {
						const cJSON *p =
							cJSON_GetObjectItem(
								params, "path");
						if (!p) p = cJSON_GetObjectItem(
							params, "source");
						if (cJSON_IsString(p))
							path = p->valuestring;
					}
					if (cJSON_IsString(type)) {
						if (path)
							off += snprintf(
								chain + off,
								sizeof(chain) - off,
								"%s %s",
								type->valuestring,
								path);
						else
							off += snprintf(
								chain + off,
								sizeof(chain) - off,
								"%s",
								type->valuestring);
					}
					i++;
				}
				snprintf(intent->action_chain,
					sizeof(intent->action_chain),
					"%s", chain);
			}

			/* Injection detection */
			const cJSON *inj = cJSON_GetObjectItem(result,
				"injection_detected");
			if (cJSON_IsTrue(inj)) {
				intent->injection_detected = true;
				const cJSON *ic = cJSON_GetObjectItem(result,
					"injection_content");
				if (cJSON_IsString(ic))
					snprintf(intent->injection_content,
						sizeof(intent->injection_content),
						"%s", ic->valuestring);
			}

			const cJSON *sandbox = cJSON_GetObjectItem(result,
				"sandbox_active");
			if (cJSON_IsTrue(sandbox))
				intent->sandbox_active = true;

			const cJSON *paths = cJSON_GetObjectItem(result,
				"authorized_paths");
			if (cJSON_IsArray(paths)) {
				char ap[512] = {0};
				int apoff = 0;
				const cJSON *p;
				cJSON_ArrayForEach(p, paths) {
					if (cJSON_IsString(p)) {
						if (apoff > 0 && apoff <
							(int)sizeof(ap) - 4)
							apoff += snprintf(
								ap + apoff,
								sizeof(ap) - apoff,
								", ");
						apoff += snprintf(
							ap + apoff,
							sizeof(ap) - apoff,
							"%s",
							p->valuestring);
					}
				}
				snprintf(intent->authorized_paths,
					sizeof(intent->authorized_paths),
					"%s", ap);
			}

			intent->state = CARD_STATE_DONE;
		} else {
			intent->state = CARD_STATE_FAILED;
		}
		cJSON_Delete(result);
	} else {
		intent->state = CARD_STATE_FAILED;
	}

	pthread_mutex_unlock(&feed->mutex);

	char byte = 'u';
	(void)write(feed->wakeup_pipe[1], &byte, 1);

	free(sa);
	return NULL;
}

void feed_submit(struct leaves_feed *feed, const char *text) {
	pthread_mutex_lock(&feed->mutex);

	if (feed->count >= MAX_INTENTS) {
		memmove(&feed->intents[0], &feed->intents[1],
			(MAX_INTENTS - 1) * sizeof(LeavesIntent));
		feed->count = MAX_INTENTS - 1;
	}

	int idx = feed->count;
	LeavesIntent *intent = &feed->intents[idx];
	memset(intent, 0, sizeof(*intent));
	snprintf(intent->natural_text, sizeof(intent->natural_text),
		"%s", text);
	intent->state = CARD_STATE_PENDING;

	/* Animate entry */
	spring_init(&intent->anim_y, 12.0f, 0.0f);
	spring_init(&intent->anim_opacity, 0.0f, 1.0f);

	feed->count++;
	pthread_mutex_unlock(&feed->mutex);

	struct submit_args *sa = calloc(1, sizeof(*sa));
	sa->feed = feed;
	sa->index = idx;
	snprintf(sa->text, sizeof(sa->text), "%s", text);

	pthread_t thread;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_create(&thread, &attr, submit_thread, sa);
	pthread_attr_destroy(&attr);
}

/* ── Watcher loading ── */

void feed_load_watchers(struct leaves_feed *feed) {
	char url[512];
	snprintf(url, sizeof(url), "%s/v1/intent/persistent?active_only=true",
		feed->api_base);

	cJSON *root = http_get(url, 5L);
	if (!root) return;

	if (!cJSON_IsArray(root)) {
		cJSON_Delete(root);
		return;
	}

	pthread_mutex_lock(&feed->mutex);
	feed->watcher_count = 0;

	const cJSON *item;
	cJSON_ArrayForEach(item, root) {
		if (feed->watcher_count >= MAX_WATCHERS) break;

		const cJSON *trigger = cJSON_GetObjectItem(item, "trigger_type");
		if (!cJSON_IsString(trigger) ||
				strcmp(trigger->valuestring, "filesystem") != 0)
			continue;

		LeavesWatcher *w = &feed->watchers[feed->watcher_count];
		memset(w, 0, sizeof(*w));

		const cJSON *id = cJSON_GetObjectItem(item, "id");
		if (cJSON_IsString(id))
			snprintf(w->id, sizeof(w->id), "%s", id->valuestring);

		const cJSON *name = cJSON_GetObjectItem(item, "name");
		if (cJSON_IsString(name))
			snprintf(w->name, sizeof(w->name), "%s",
				name->valuestring);

		const cJSON *fc = cJSON_GetObjectItem(item, "fire_count");
		if (cJSON_IsNumber(fc))
			w->fire_count = (int)fc->valuedouble;

		w->active = true;

		/* Parse trigger_config for path and pattern */
		const cJSON *tc = cJSON_GetObjectItem(item, "trigger_config");
		if (cJSON_IsObject(tc)) {
			const cJSON *path = cJSON_GetObjectItem(tc, "path");
			if (cJSON_IsString(path))
				snprintf(w->watched_path,
					sizeof(w->watched_path),
					"%s", path->valuestring);
			const cJSON *pat = cJSON_GetObjectItem(tc, "pattern");
			if (cJSON_IsString(pat))
				snprintf(w->pattern, sizeof(w->pattern),
					"%s", pat->valuestring);
		} else if (cJSON_IsString(tc)) {
			/* trigger_config may be a JSON string that needs parsing */
			cJSON *parsed = cJSON_Parse(tc->valuestring);
			if (parsed) {
				const cJSON *path = cJSON_GetObjectItem(parsed,
					"path");
				if (cJSON_IsString(path))
					snprintf(w->watched_path,
						sizeof(w->watched_path),
						"%s", path->valuestring);
				const cJSON *pat = cJSON_GetObjectItem(parsed,
					"pattern");
				if (cJSON_IsString(pat))
					snprintf(w->pattern,
						sizeof(w->pattern),
						"%s", pat->valuestring);
				cJSON_Delete(parsed);
			}
		}

		spring_init(&w->anim_opacity, 0.0f, 1.0f);
		feed->watcher_count++;
	}
	pthread_mutex_unlock(&feed->mutex);

	cJSON_Delete(root);

	char byte = 'u';
	(void)write(feed->wakeup_pipe[1], &byte, 1);
}

/* ── Search thread ── */

struct search_args {
	struct leaves_feed *feed;
	int index;
	char query[512];
};

static void *search_thread(void *arg) {
	struct search_args *sa = arg;
	struct leaves_feed *feed = sa->feed;

	/* URL-encode the query (minimal: spaces → +) */
	char encoded[512] = {0};
	int j = 0;
	for (int i = 0; sa->query[i] && j < (int)sizeof(encoded) - 4; i++) {
		if (sa->query[i] == ' ')
			encoded[j++] = '+';
		else if (sa->query[i] == '&' || sa->query[i] == '=' ||
				sa->query[i] == '?') {
			snprintf(encoded + j, sizeof(encoded) - j,
				"%%%02X", (unsigned char)sa->query[i]);
			j += 3;
		} else
			encoded[j++] = sa->query[i];
	}

	char url[1024];
	snprintf(url, sizeof(url), "%s/v1/cortex/search?q=%.480s&top_k=10",
		feed->api_base, encoded);

	cJSON *root = http_get(url, 15L);

	pthread_mutex_lock(&feed->mutex);
	LeavesIntent *intent = &feed->intents[sa->index];

	if (!root) {
		snprintf(intent->action_chain,
			sizeof(intent->action_chain),
			"search failed — API unreachable");
		intent->state = CARD_STATE_FAILED;
		pthread_mutex_unlock(&feed->mutex);
		char byte = 'u';
		(void)write(feed->wakeup_pipe[1], &byte, 1);
		free(sa);
		return NULL;
	}

	const cJSON *count = cJSON_GetObjectItem(root, "count");
	const cJSON *results = cJSON_GetObjectItem(root, "results");

	int n = 0;
	if (cJSON_IsArray(results)) {
		const cJSON *hit;
		cJSON_ArrayForEach(hit, results) {
			if (n >= MAX_SEARCH_HITS) break;
			LeavesSearchHit *h = &intent->search_hits[n];
			memset(h, 0, sizeof(*h));

			const cJSON *title = cJSON_GetObjectItem(hit, "title");
			if (cJSON_IsString(title))
				snprintf(h->title, sizeof(h->title),
					"%s", title->valuestring);

			const cJSON *path = cJSON_GetObjectItem(hit,
				"source_path");
			if (cJSON_IsString(path))
				snprintf(h->path, sizeof(h->path),
					"%s", path->valuestring);
			n++;
		}
	}
	intent->search_hit_count = n;

	int total = cJSON_IsNumber(count) ? (int)count->valuedouble : n;
	snprintf(intent->action_chain, sizeof(intent->action_chain),
		"%d result%s", total, total == 1 ? "" : "s");

	intent->state = CARD_STATE_SEARCH_RESULT;
	pthread_mutex_unlock(&feed->mutex);

	cJSON_Delete(root);

	char byte = 'u';
	(void)write(feed->wakeup_pipe[1], &byte, 1);
	free(sa);
	return NULL;
}

void feed_search(struct leaves_feed *feed, const char *query) {
	pthread_mutex_lock(&feed->mutex);

	if (feed->count >= MAX_INTENTS) {
		memmove(&feed->intents[0], &feed->intents[1],
			(MAX_INTENTS - 1) * sizeof(LeavesIntent));
		feed->count = MAX_INTENTS - 1;
	}

	int idx = feed->count;
	LeavesIntent *intent = &feed->intents[idx];
	memset(intent, 0, sizeof(*intent));
	snprintf(intent->natural_text, sizeof(intent->natural_text),
		"search: %s", query);
	intent->state = CARD_STATE_PENDING;

	spring_init(&intent->anim_y, 12.0f, 0.0f);
	spring_init(&intent->anim_opacity, 0.0f, 1.0f);

	feed->count++;
	pthread_mutex_unlock(&feed->mutex);

	struct search_args *sa = calloc(1, sizeof(*sa));
	sa->feed = feed;
	sa->index = idx;
	snprintf(sa->query, sizeof(sa->query), "%s", query);

	pthread_t thread;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_create(&thread, &attr, search_thread, sa);
	pthread_attr_destroy(&attr);
}

/* ── Watch creation thread ── */

struct watch_args {
	struct leaves_feed *feed;
	int index;
	char text[512];
};

static void *create_watcher_thread(void *arg) {
	struct watch_args *wa = arg;
	struct leaves_feed *feed = wa->feed;

	char url[512];
	snprintf(url, sizeof(url), "%s/v1/intent/persist", feed->api_base);

	char escaped[512];
	json_escape(wa->text, escaped, sizeof(escaped));
	char body[1280];
	snprintf(body, sizeof(body),
		"{\"name\":\"%.200s\",\"text\":\"%.480s\","
		"\"trigger_type\":\"filesystem\","
		"\"trigger_config\":{}}",
		escaped, escaped);

	cJSON *result = http_post(url, body, 30L);

	pthread_mutex_lock(&feed->mutex);
	LeavesIntent *intent = &feed->intents[wa->index];

	if (result) {
		const cJSON *status = cJSON_GetObjectItem(result, "status");
		if (cJSON_IsString(status) &&
				strcmp(status->valuestring, "stored") == 0) {
			snprintf(intent->action_chain,
				sizeof(intent->action_chain),
				"watcher created");
			intent->state = CARD_STATE_DONE;
		} else {
			const cJSON *msg = cJSON_GetObjectItem(result,
				"detail");
			if (!msg) msg = cJSON_GetObjectItem(result, "message");
			snprintf(intent->action_chain,
				sizeof(intent->action_chain),
				"%s", cJSON_IsString(msg) ?
					msg->valuestring : "failed");
			intent->state = CARD_STATE_FAILED;
		}
		cJSON_Delete(result);
	} else {
		snprintf(intent->action_chain,
			sizeof(intent->action_chain),
			"API unreachable");
		intent->state = CARD_STATE_FAILED;
	}
	pthread_mutex_unlock(&feed->mutex);

	/* Reload watchers display */
	feed_load_watchers(feed);

	char byte = 'u';
	(void)write(feed->wakeup_pipe[1], &byte, 1);
	free(wa);
	return NULL;
}

void feed_create_watcher(struct leaves_feed *feed, const char *text) {
	pthread_mutex_lock(&feed->mutex);

	if (feed->count >= MAX_INTENTS) {
		memmove(&feed->intents[0], &feed->intents[1],
			(MAX_INTENTS - 1) * sizeof(LeavesIntent));
		feed->count = MAX_INTENTS - 1;
	}

	int idx = feed->count;
	LeavesIntent *intent = &feed->intents[idx];
	memset(intent, 0, sizeof(*intent));
	snprintf(intent->natural_text, sizeof(intent->natural_text),
		"watch: %s", text);
	intent->state = CARD_STATE_PENDING;

	spring_init(&intent->anim_y, 12.0f, 0.0f);
	spring_init(&intent->anim_opacity, 0.0f, 1.0f);

	feed->count++;
	pthread_mutex_unlock(&feed->mutex);

	struct watch_args *wa = calloc(1, sizeof(*wa));
	wa->feed = feed;
	wa->index = idx;
	snprintf(wa->text, sizeof(wa->text), "%s", text);

	pthread_t thread;
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_create(&thread, &attr, create_watcher_thread, wa);
	pthread_attr_destroy(&attr);
}

void feed_process_updates(struct leaves_feed *feed) {
	/* Drain is handled by compositor.c which reads byte values */
	(void)feed;
}

bool feed_animate(struct leaves_feed *feed, float dt) {
	bool any_active = false;
	pthread_mutex_lock(&feed->mutex);

	/* Briefing fade-in */
	if (feed->briefing.loaded &&
			!spring_settled(&feed->briefing.anim_opacity)) {
		spring_update(&feed->briefing.anim_opacity, dt);
		any_active = true;
	}

	/* Watcher fade-in */
	for (int i = 0; i < feed->watcher_count; i++) {
		if (!spring_settled(&feed->watchers[i].anim_opacity)) {
			spring_update(&feed->watchers[i].anim_opacity, dt);
			any_active = true;
		}
	}

	for (int i = 0; i < feed->count; i++) {
		LeavesIntent *intent = &feed->intents[i];
		if (!spring_settled(&intent->anim_y)) {
			spring_update(&intent->anim_y, dt);
			any_active = true;
		}
		if (!spring_settled(&intent->anim_opacity)) {
			spring_update(&intent->anim_opacity, dt);
			any_active = true;
		}
	}
	/* Pending/executing cards need continuous redraws for animation */
	for (int i = 0; i < feed->count; i++) {
		if (feed->intents[i].state == CARD_STATE_PENDING ||
				feed->intents[i].state == CARD_STATE_EXECUTING ||
				feed->intents[i].state == CARD_STATE_AWAITING_CONFIRM) {
			any_active = true;
			break;
		}
	}
	pthread_mutex_unlock(&feed->mutex);
	return any_active;
}

void feed_confirm(struct leaves_feed *feed) {
	feed->awaiting_confirm = false;
	char byte = 'y';
	(void)write(feed->confirm_pipe[1], &byte, 1);
}

void feed_cancel(struct leaves_feed *feed) {
	feed->awaiting_confirm = false;
	char byte = 'n';
	(void)write(feed->confirm_pipe[1], &byte, 1);
}
