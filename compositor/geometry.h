#ifndef LEAVES_GEOMETRY_H
#define LEAVES_GEOMETRY_H

#define GRID              4

#define SPACE_XS          4    /* GRID * 1 */
#define SPACE_S           8    /* GRID * 2 */
#define SPACE_M          12    /* GRID * 3 */
#define SPACE_L          16    /* GRID * 4 */
#define SPACE_XL         20    /* GRID * 5 */
#define SPACE_2XL        24    /* GRID * 6 */
#define SPACE_3XL        32    /* GRID * 8 */

/* Card geometry */
#define CARD_RADIUS        6
#define CARD_PADDING_H    SPACE_L     /* 16px */
#define CARD_PADDING_V    SPACE_M     /* 12px */
#define CARD_GAP          SPACE_XS    /*  4px */
#define CARD_MARGIN_H     SPACE_2XL   /* 24px */
#define CARD_INDICATOR_W   3

/* Taskbar */
#define INPUT_HEIGHT      52
#define INPUT_PADDING_L   SPACE_L     /* 16px */
#define INPUT_INDICATOR_W  3
#define TASKBAR_ICON_W    52          /* left zone: OS icon */
#define TASKBAR_APPS_W    120         /* legacy — still used by hit-test */
#define TASKBAR_DOT_R     4           /* activity dot radius */
#define TASKBAR_DOT_GAP   10          /* gap between dots */
#define HISTORY_ICON_W    32          /* history toggle icon zone */

/* Feed area */
#define FEED_PADDING_T    SPACE_2XL   /* 24px */
#define FEED_PADDING_B    (INPUT_HEIGHT + SPACE_M)

/* Overlay panel */
#define OVERLAY_MAX_W     560
#define OVERLAY_MARGIN_H   48   /* 96/2 — min(screen_w - 96, 560) */
#define OVERLAY_RADIUS     10
#define OVERLAY_PADDING    28
#define OVERLAY_BUTTON_H   36
#define OVERLAY_BUTTON_R    6
#define OVERLAY_BUTTON_GAP SPACE_M  /* 12px between buttons */

/* Status indicators (bottom bar right zone) */
#define STATUS_PAD_H      10          /* right-edge padding */

/* Quick-settings dropdown */
#define DROPDOWN_W        280
#define DROPDOWN_PAD      16
#define DROPDOWN_RADIUS    8
#define DROPDOWN_TOGGLE_R  6          /* toggle dot radius */
#define DROPDOWN_SECTION_GAP SPACE_L  /* between sections */

/* Misc */
#define SEPARATOR_H        1
#define STATUS_BANNER_H   32
#define SCREEN_MARGIN_H   SPACE_2XL   /* 24px */

#endif
