#ifndef MARSHAL_FEED_H
#define MARSHAL_FEED_H

#include <pthread.h>
#include <stdbool.h>
#include "spring.h"

#define MAX_INTENTS 200
#define MAX_BRIEFING_GROUPS 16
#define MAX_BRIEFING_ITEMS 8
#define MAX_WATCHERS 16
#define MAX_SEARCH_HITS 10

typedef enum {
	CARD_STATE_PENDING,           /* inference in progress */
	CARD_STATE_AWAITING_CONFIRM,  /* plan ready, needs Y/N */
	CARD_STATE_EXECUTING,         /* confirmed, running */
	CARD_STATE_DONE,
	CARD_STATE_FAILED,
	CARD_STATE_CANCELLED,         /* user pressed N */
	CARD_STATE_HISTORY,           /* loaded from /v1/history */
	CARD_STATE_SEARCH_RESULT,     /* cortex search results */
} MarshalCardState;

/* ── Search hit (embedded in intent card) ── */

typedef struct {
	char title[128];
	char path[256];
} MarshalSearchHit;

typedef struct {
	char intent_id[64];
	char natural_text[512];
	char action_chain[256];
	char capability_scope[512];
	double duration_ms;
	MarshalCardState state;

	/* Authorization data — populated after /v1/intent/plan */
	bool preview_required;
	bool reversible;
	char resources[512];        /* joined resource paths */
	char actions_summary[512];  /* "DELETE 4 files in ~/Downloads" */

	/* Search results — only populated for CARD_STATE_SEARCH_RESULT */
	MarshalSearchHit search_hits[MAX_SEARCH_HITS];
	int search_hit_count;

	/* Result summary — human-readable text shown in the card.
	 * 64 KB to hold full file listings (500 files × long names). */
	char result_summary[65536];

	/* Injection detection — populated from execution response */
	bool injection_detected;
	char injection_content[512];
	bool sandbox_active;
	char authorized_paths[512];

	/* Proactive intents — pushed from agentd, not user-initiated.
	 * Rendered with an accent border and AI glyph to signal
	 * "the OS noticed something and wrote this card for you." */
	bool proactive;

	/* Time-machine: true once /v1/history/{id}/detail has been
	 * fetched and rendered into result_summary. Prevents re-fetch
	 * on every re-expand. Only meaningful for CARD_STATE_HISTORY. */
	bool detail_fetched;

	/* Spring animation */
	struct spring anim_y;
	struct spring anim_opacity;
} MarshalIntent;

/* ── Briefing data ── */

typedef struct {
	char title[128];
	char path[256];
} MarshalBriefingItem;

typedef struct {
	char directory[256];
	int count;
	MarshalBriefingItem items[MAX_BRIEFING_ITEMS];
	int item_count;
} MarshalBriefingGroup;

typedef struct {
	char source_type[32];
	int count;
	MarshalBriefingGroup groups[MAX_BRIEFING_GROUPS];
	int group_count;
} MarshalBriefingSection;

#define MAX_BRIEFING_SECTIONS 4

typedef struct {
	char headline[256];
	int total_changes;
	int period_hours;
	bool empty;
	bool loaded;            /* true after successful fetch */
	MarshalBriefingSection sections[MAX_BRIEFING_SECTIONS];
	int section_count;
	struct spring anim_opacity;
} MarshalBriefing;

/* ── Watcher data (persistent filesystem intents) ── */

typedef struct {
	char id[64];
	char name[128];
	char watched_path[256];
	char pattern[64];
	int fire_count;
	bool active;
	struct spring anim_opacity;
} MarshalWatcher;

struct marshal_feed {
	MarshalIntent intents[MAX_INTENTS];
	int count;
	float scroll_offset;
	pthread_mutex_t mutex;
	int wakeup_pipe[2];
	int confirm_pipe[2];    /* [0]=read (HTTP thread), [1]=write (main) */
	bool awaiting_confirm;  /* true when overlay is active */
	int confirm_card_idx;   /* which card is awaiting confirmation */
	int selected_card;      /* index of card selected for copy, or -1 */
	int expanded_card;      /* index of card expanded for full detail, or -1 */
	float expanded_scroll;  /* scroll offset within expanded card overlay */
	int expanded_content_h; /* measured content height of expanded card */
	char api_base[256];
	char wayland_display[64];  /* set directly by compositor, not getenv */
	MarshalBriefing briefing;
	MarshalWatcher watchers[MAX_WATCHERS];
	int watcher_count;
};

struct marshal_feed *feed_create(const char *api_base);
void feed_destroy(struct marshal_feed *feed);
void feed_load_history(struct marshal_feed *feed);
/* Insert a proactive intent card pushed from agentd. `intent_json` is a
 * full GoalSpec JSON string (one line, no trailing newline). Dedup by
 * intent_id: if a card with the same id already exists, this is a no-op.
 * Writes a byte to feed->wakeup_pipe to trigger a repaint. Safe to call
 * from any thread (locks feed->mutex). */
void feed_insert_proactive(struct marshal_feed *feed, const char *intent_json);
void feed_load_briefing(struct marshal_feed *feed);
void feed_load_watchers(struct marshal_feed *feed);
void feed_submit(struct marshal_feed *feed, const char *text);
void feed_search(struct marshal_feed *feed, const char *query);
void feed_create_watcher(struct marshal_feed *feed, const char *text);
/* Time-machine: lazy-load the full audit trace for a history card
 * (per-action type/agent/duration/status + state transitions + errors)
 * and format it into the card's result_summary for display. No-op if
 * the card isn't CARD_STATE_HISTORY or detail has already been fetched. */
void feed_load_detail(struct marshal_feed *feed, int card_idx);
/* Time-machine: re-execute a stored GoalSpec under a new intent_id
 * and prepend a replay banner to result_summary. Destructive replays
 * are refused by the API; the banner shows the refusal. */
void feed_replay(struct marshal_feed *feed, int card_idx);
void feed_process_updates(struct marshal_feed *feed);
bool feed_animate(struct marshal_feed *feed, float dt);
void feed_confirm(struct marshal_feed *feed);
void feed_cancel(struct marshal_feed *feed);

#endif
