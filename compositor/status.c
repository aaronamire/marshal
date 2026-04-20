#define _GNU_SOURCE
#include "status.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <alloca.h>
#include <alsa/asoundlib.h>

struct marshal_status *status_create(void) {
	struct marshal_status *s = calloc(1, sizeof(*s));
	if (!s) return NULL;
	s->battery_pct = -1;
	s->volume_pct  = -1;
	snprintf(s->kb_layout, sizeof(s->kb_layout), "US");
	status_update_clock(s);
	status_poll(s);
	return s;
}

void status_destroy(struct marshal_status *s) {
	free(s);
}

void status_update_clock(struct marshal_status *s) {
	time_t now = time(NULL);
	struct tm *tm = localtime(&now);

	int h12 = tm->tm_hour % 12;
	if (h12 == 0) h12 = 12;
	const char *ap = tm->tm_hour < 12 ? "AM" : "PM";
	snprintf(s->time_str, sizeof(s->time_str), "%d:%02d %s", h12, tm->tm_min, ap);

	static const char *day[] = {"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
	static const char *mon[] = {"Jan","Feb","Mar","Apr","May","Jun",
	                            "Jul","Aug","Sep","Oct","Nov","Dec"};
	snprintf(s->date_str, sizeof(s->date_str), "%s, %s %d",
		day[tm->tm_wday], mon[tm->tm_mon], tm->tm_mday);
}

/* ── Battery (pure sysfs, no subprocesses) ── */

static void poll_battery(struct marshal_status *s) {
	static const char *bases[] = {
		"/sys/class/power_supply/BAT0",
		"/sys/class/power_supply/BAT1",
		"/sys/class/power_supply/macsmc-battery",
		NULL
	};

	for (int i = 0; bases[i]; i++) {
		char path[256];
		snprintf(path, sizeof(path), "%s/capacity", bases[i]);
		FILE *f = fopen(path, "r");
		if (!f) continue;

		int pct = -1;
		if (fscanf(f, "%d", &pct) == 1)
			s->battery_pct = pct;
		fclose(f);

		snprintf(path, sizeof(path), "%s/status", bases[i]);
		f = fopen(path, "r");
		if (f) {
			char buf[32] = {0};
			if (fgets(buf, sizeof(buf), f))
				s->battery_charging = (strstr(buf, "Charging") != NULL);
			fclose(f);
		}
		return; /* found a battery — stop */
	}
	s->battery_pct = -1;
}

/* ── WiFi (sysfs + /proc/net/wireless, no subprocesses) ── */

static void poll_wifi(struct marshal_status *s) {
	s->wifi_connected = false;
	s->wifi_ssid[0] = '\0';
	s->wifi_signal = 0;

	DIR *dir = opendir("/sys/class/net");
	if (!dir) return;

	struct dirent *ent;
	while ((ent = readdir(dir))) {
		if (ent->d_name[0] == '.') continue;

		/* Is it wireless? (has /sys/class/net/<iface>/wireless/) */
		char wpath[256];
		snprintf(wpath, sizeof(wpath),
			"/sys/class/net/%s/wireless", ent->d_name);
		if (access(wpath, F_OK) != 0) continue;

		/* Check operstate */
		char opath[256];
		snprintf(opath, sizeof(opath),
			"/sys/class/net/%s/operstate", ent->d_name);
		FILE *f = fopen(opath, "r");
		if (!f) continue;
		char state[16] = {0};
		if (fgets(state, sizeof(state), f))
			; /* just read */
		fclose(f);

		if (strncmp(state, "up", 2) != 0) continue;

		s->wifi_connected = true;

		/* SSID: best-effort via iwgetid (single quick exec) */
		FILE *iw = popen("iwgetid -r 2>/dev/null", "r");
		if (iw) {
			if (fgets(s->wifi_ssid, sizeof(s->wifi_ssid), iw)) {
				char *nl = strchr(s->wifi_ssid, '\n');
				if (nl) *nl = '\0';
			}
			pclose(iw);
		}
		if (!s->wifi_ssid[0])
			snprintf(s->wifi_ssid, sizeof(s->wifi_ssid), "Connected");

		/* Signal quality from /proc/net/wireless */
		FILE *wf = fopen("/proc/net/wireless", "r");
		if (wf) {
			char line[256];
			while (fgets(line, sizeof(line), wf)) {
				if (!strstr(line, ent->d_name)) continue;
				float q = 0;
				if (sscanf(line, "%*s %*d %f", &q) == 1) {
					s->wifi_signal = (int)(q * 100 / 70);
					if (s->wifi_signal > 100) s->wifi_signal = 100;
				}
				break;
			}
			fclose(wf);
		}
		break;
	}
	closedir(dir);
}

/* ── Bluetooth (sysfs + rfkill) ── */

static void poll_bluetooth(struct marshal_status *s) {
	s->bt_available = false;
	s->bt_enabled   = false;

	DIR *dir = opendir("/sys/class/bluetooth");
	if (!dir) return;

	struct dirent *ent;
	while ((ent = readdir(dir))) {
		if (strncmp(ent->d_name, "hci", 3) != 0) continue;
		s->bt_available = true;

		/* Check rfkill soft-block state */
		char rfpath[256];
		snprintf(rfpath, sizeof(rfpath),
			"/sys/class/bluetooth/%s/rfkill", ent->d_name);
		DIR *rdir = opendir(rfpath);
		if (rdir) {
			struct dirent *rent;
			while ((rent = readdir(rdir))) {
				if (rent->d_name[0] == '.') continue;
				char spath[512];
				snprintf(spath, sizeof(spath),
					"%s/%s/soft", rfpath, rent->d_name);
				FILE *f = fopen(spath, "r");
				if (f) {
					int blocked = 1;
					if (fscanf(f, "%d", &blocked) == 1)
						s->bt_enabled = (blocked == 0);
					fclose(f);
				}
				break;
			}
			closedir(rdir);
		} else {
			/* No rfkill directory → assume enabled */
			s->bt_enabled = true;
		}
		break;
	}
	closedir(dir);
}

/* ── Volume (ALSA mixer — works with PipeWire's ALSA compat layer) ── */

static void poll_volume(struct marshal_status *s) {
	snd_mixer_t *mixer = NULL;
	snd_mixer_selem_id_t *sid = NULL;

	s->volume_pct = -1;
	s->volume_muted = false;

	if (snd_mixer_open(&mixer, 0) < 0) return;
	if (snd_mixer_attach(mixer, "default") < 0) goto out;
	if (snd_mixer_selem_register(mixer, NULL, NULL) < 0) goto out;
	if (snd_mixer_load(mixer) < 0) goto out;

	snd_mixer_selem_id_alloca(&sid);
	snd_mixer_selem_id_set_index(sid, 0);

	/* Try common master element names */
	static const char *names[] = { "Master", "PCM", "Speaker", NULL };
	snd_mixer_elem_t *elem = NULL;
	for (int i = 0; names[i]; i++) {
		snd_mixer_selem_id_set_name(sid, names[i]);
		elem = snd_mixer_find_selem(mixer, sid);
		if (elem) break;
	}
	if (!elem) goto out;

	/* Volume percentage */
	long vmin, vmax, vol;
	if (snd_mixer_selem_get_playback_volume_range(elem, &vmin, &vmax) == 0 &&
			vmax > vmin &&
			snd_mixer_selem_get_playback_volume(elem,
				SND_MIXER_SCHN_MONO, &vol) == 0) {
		s->volume_pct = (int)((vol - vmin) * 100 / (vmax - vmin));
	}

	/* Mute state */
	int sw = 1;
	if (snd_mixer_selem_has_playback_switch(elem) &&
			snd_mixer_selem_get_playback_switch(elem,
				SND_MIXER_SCHN_MONO, &sw) == 0) {
		s->volume_muted = (sw == 0);
	}

out:
	snd_mixer_close(mixer);
}

void status_poll(struct marshal_status *s) {
	poll_battery(s);
	poll_wifi(s);
	poll_bluetooth(s);
	poll_volume(s);
}
