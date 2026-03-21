#include "spring.h"
#include <math.h>

void spring_init(struct spring *s, float start, float target) {
	s->pos = start;
	s->vel = 0.0f;
	s->target = target;
	s->stiffness = 280.0f;
	s->damping = 26.0f;
}

void spring_update(struct spring *s, float dt) {
	float force = -s->stiffness * (s->pos - s->target) - s->damping * s->vel;
	s->vel += force * dt;
	s->pos += s->vel * dt;
}

bool spring_settled(const struct spring *s) {
	float dist = fabsf(s->pos - s->target);
	float speed = fabsf(s->vel);
	return dist < 0.3f && speed < 0.3f;
}
