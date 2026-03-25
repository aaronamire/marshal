#ifndef LEAVES_FEED_H
#define LEAVES_FEED_H

#include <pthread.h>
#include <stdbool.h>
#include "spring.h"

#define MAX_INTENTS 200
#define MAX_BRIEFING_GROUPS 16
#define MAX_BRIEFING_ITEMS 8

typedef enum {
	CARD_STATE_PENDING,           /* inference in progress */
	CARD_STATE_AWAITING_CONFIRM,  /* plan ready, needs Y/N */
	CARD_STATE_EXECUTING,         /* confirmed, running */
	CARD_STATE_DONE,
	CARD_STATE_FAILED,
	CARD_STATE_CANCELLED,         /* user pressed N */
	CARD_STATE_HISTORY,           /* loaded from /v1/history */
} LeavesCardState;

typedef struct {
	char intent_id[64];
	char natural_text[512];
	char action_chain[256];
	char capability_scope[512];
	double duration_ms;
	LeavesCardState state;

	/* Authorization data — populated after /v1/intent/plan */
	bool preview_required;
	bool reversible;
	char resources[512];        /* joined resource paths */
	char actions_summary[512];  /* "DELETE 4 files in ~/Downloads" */

	/* Spring animation */
	struct spring anim_y;
	struct spring anim_opacity;
} LeavesIntent;

/* ── Briefing data ── */

typedef struct {
	char title[128];
	char path[256];
} LeavesBriefingItem;

typedef struct {
	char directory[256];
	int count;
	LeavesBriefingItem items[MAX_BRIEFING_ITEMS];
	int item_count;
} LeavesBriefingGroup;

typedef struct {
	char source_type[32];
	int count;
	LeavesBriefingGroup groups[MAX_BRIEFING_GROUPS];
	int group_count;
} LeavesBriefingSection;

#define MAX_BRIEFING_SECTIONS 4

typedef struct {
	char headline[256];
	int total_changes;
	int period_hours;
	bool empty;
	bool loaded;            /* true after successful fetch */
	LeavesBriefingSection sections[MAX_BRIEFING_SECTIONS];
	int section_count;
	struct spring anim_opacity;
} LeavesBriefing;

struct leaves_feed {
	LeavesIntent intents[MAX_INTENTS];
	int count;
	float scroll_offset;
	pthread_mutex_t mutex;
	int wakeup_pipe[2];
	int confirm_pipe[2];    /* [0]=read (HTTP thread), [1]=write (main) */
	bool awaiting_confirm;  /* true when overlay is active */
	int confirm_card_idx;   /* which card is awaiting confirmation */
	char api_base[256];
	LeavesBriefing briefing;
};

struct leaves_feed *feed_create(const char *api_base);
void feed_destroy(struct leaves_feed *feed);
void feed_load_history(struct leaves_feed *feed);
void feed_load_briefing(struct leaves_feed *feed);
void feed_submit(struct leaves_feed *feed, const char *text);
void feed_process_updates(struct leaves_feed *feed);
bool feed_animate(struct leaves_feed *feed, float dt);
void feed_confirm(struct leaves_feed *feed);
void feed_cancel(struct leaves_feed *feed);

#endif
