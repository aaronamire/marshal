#ifndef LEAVES_STATUS_H
#define LEAVES_STATUS_H

#include <stdbool.h>
#include <pthread.h>

struct leaves_status {
	/* Clock */
	char time_str[16];      /* "2:34 PM"       */
	char date_str[32];      /* "Thu, Mar 27"    */

	/* Battery */
	int  battery_pct;       /* 0–100, or -1 = no battery */
	bool battery_charging;

	/* WiFi */
	bool wifi_connected;
	char wifi_ssid[64];
	int  wifi_signal;       /* 0–100 approx signal quality */

	/* Bluetooth */
	bool bt_available;
	bool bt_enabled;

	/* Keyboard layout */
	char kb_layout[8];      /* "US", "RU", etc. */

	/* UI toggles */
	bool dropdown_open;
	bool history_open;   /* true → show feed cards; false → show wallpaper */
};

struct leaves_status *status_create(void);
void status_destroy(struct leaves_status *s);

/* Update time strings from the wall clock — cheap, call every second. */
void status_update_clock(struct leaves_status *s);

/* Poll sysfs for battery, wifi, bluetooth — call every ~30 s. */
void status_poll(struct leaves_status *s);

#endif
