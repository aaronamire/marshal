#ifndef LEAVES_SPRING_H
#define LEAVES_SPRING_H

#include <stdbool.h>

struct spring {
	float pos;
	float vel;
	float target;
	float stiffness;  /* k */
	float damping;    /* c */
};

void spring_init(struct spring *s, float start, float target);
void spring_update(struct spring *s, float dt);
bool spring_settled(const struct spring *s);

#endif
