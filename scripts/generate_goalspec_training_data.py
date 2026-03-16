#!/usr/bin/env python3
"""Generate GoalSpec training pairs via self-instruct using Claude API.

Usage:
    python scripts/generate_goalspec_training_data.py             # full run → data/goalspec_training_pairs.jsonl
    python scripts/generate_goalspec_training_data.py --dry-run 5 # print 5 pairs and exit
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic
import jsonschema

ROOT = Path(__file__).parent.parent
SEED_FILE = ROOT / "rag" / "seed_examples.jsonl"
SCHEMA_FILE = ROOT / "agents" / "schema" / "goal_spec.json"
OUTPUT_FILE = ROOT / "data" / "goalspec_training_pairs.jsonl"

MODEL = "claude-haiku-4-5-20251001"
TARGET = 1500
# Dummy UUID v4 injected before schema validation (intent_id is never LLM-generated)
DUMMY_UUID = "00000000-0000-4000-8000-000000000001"

ACTION_TYPES = ["QUERY", "READ", "WRITE", "DELETE", "MOVE", "COPY", "SUMMARIZE", "COMPOSE"]

MULTI_ACTION_COMBOS = [
    ("QUERY", "DELETE"),   # find then delete — hardest for 3B
    ("QUERY", "MOVE"),     # find then move
    ("QUERY", "COPY"),     # find then copy
    ("QUERY", "READ"),     # search for file then read it
]

SYSTEM_PROMPT = """\
You generate training data for an AI file assistant called Leaves OS.
For each request, output a JSON array of GoalSpec objects — no markdown, no explanation.

GOALSPEC SCHEMA (do NOT include "intent_id"):
{
  "natural_text": "<verbatim user intent>",
  "category": "<file_task|email_task|web_task|system_task|writing_task>",
  "actions": [
    {
      "action_id": "act-1",
      "type": "<QUERY|READ|WRITE|DELETE|MOVE|COPY|SUMMARIZE|COMPOSE>",
      "agent": "<file|email|web|system|writing>",
      "params": { ... },
      "destructive": <true|false>,
      "depends_on": []
    }
  ],
  "authorization": {
    "resources": ["<path or resource identifier>"],
    "preview_required": <true|false>,
    "reversible": <true|false>
  }
}

ACTION RULES:
  QUERY    → search/list/find files. agent=file. destructive=false.
             params: {path, pattern?, search_type}
             search_type: "glob" | "size" | "modified" | "count" | "content"
  READ     → read file contents. agent=file. destructive=false.
             params: {path}
  WRITE    → create/overwrite file. agent=file. destructive=false (new) or true (overwrite).
             params: {path, content}
  DELETE   → remove files. agent=file. destructive=true. preview_required=true. reversible=false.
             params: {path} or {path, pattern}
  MOVE     → move or rename. agent=file. destructive=true. reversible=true.
             params: {source, destination} or {source, pattern, destination}
  COPY     → copy files. agent=file. destructive=false.
             params: {source, destination} or {source, pattern, destination}
  SUMMARIZE→ summarize content. agent=writing. destructive=false.
             params: {path} or {source}
  COMPOSE  → draft text. agent=writing or email. destructive=false.
             params: {topic, format?}

AUTHORIZATION RULES:
  preview_required=true  → any action is DELETE or MOVE
  preview_required=false → all actions are non-destructive
  reversible=false       → any action is DELETE
  reversible=true        → no DELETE actions

MULTI-ACTION ORDERING:
  QUERY must precede DELETE, MOVE, COPY (Rule 11: discovery before mutation)
  act-2 should have depends_on: ["act-1"]

VARIETY GUIDELINES:
  - Use realistic paths: ~, ~/Documents, ~/Downloads, ~/Desktop, ~/dev, /tmp, /var/log, /home/user
  - Use varied file types: .txt, .pdf, .py, .md, .json, .log, .csv, .jpg, .zip, .sh, .yaml, .db
  - Phrase intents naturally: imperatives, questions, colloquial requests
  - Vary specificity: single files, patterns (*.log), entire directories
  - Include temporal variants: "older than 7 days", "modified last hour", "from yesterday"
"""


def load_seeds(path: Path) -> list[dict]:
    seeds = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            seeds.append(json.loads(line))
    return seeds


def load_schema(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_goalspec(spec: dict, schema: dict) -> bool:
    """Inject dummy intent_id and validate against schema. Returns True if valid."""
    candidate = {**spec, "intent_id": DUMMY_UUID}
    try:
        jsonschema.validate(candidate, schema)
        return True
    except jsonschema.ValidationError:
        return False


def parse_response(text: str) -> list[dict]:
    """Extract a list of GoalSpec dicts from Claude's response."""
    text = text.strip()

    # Primary: expect a JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(text[start : end + 1])
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass

    # Fallback: extract individual top-level JSON objects
    results = []
    depth = 0
    obj_start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start != -1:
                try:
                    obj = json.loads(text[obj_start : i + 1])
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = -1
    return results


def call_api(client: anthropic.Anthropic, user_prompt: str) -> list[dict]:
    """Call Claude and return parsed GoalSpec dicts."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return parse_response(response.content[0].text)


def process_batch(
    specs: list[dict],
    schema: dict,
    pairs: list[dict],
    dry_run_limit: int = 0,
) -> tuple[int, int]:
    """Validate specs and append valid pairs. Returns (added, rejected)."""
    added = 0
    rejected = 0
    for spec in specs:
        if dry_run_limit and len(pairs) >= dry_run_limit:
            break
        if not isinstance(spec, dict):
            rejected += 1
            continue
        intent = spec.get("natural_text", "").strip()
        if not intent:
            rejected += 1
            continue
        spec.pop("intent_id", None)  # remove if LLM accidentally included it
        if validate_goalspec(spec, schema):
            pairs.append({
                "messages": [
                    {"role": "user", "content": intent},
                    {"role": "assistant", "content": json.dumps(spec, separators=(",", ":"))},
                ]
            })
            added += 1
        else:
            rejected += 1
    return added, rejected


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def prompt_seed_variants(seed: dict, n: int) -> str:
    return (
        f"Generate {n} diverse GoalSpec variants inspired by this seed example.\n"
        f"Change the phrasing, file paths, file types, and contexts — but keep the same action pattern.\n"
        f"Seed: {json.dumps(seed)}\n\n"
        f"Return a JSON array of {n} GoalSpec objects."
    )


def prompt_action_type(action_type: str, n: int) -> str:
    return (
        f"Generate {n} diverse GoalSpec examples where the primary action is {action_type}.\n"
        f"Vary: phrasing style, file types, directory paths, and intent specificity.\n"
        f"Make intents sound like real users (imperatives, questions, casual requests).\n\n"
        f"Return a JSON array of {n} GoalSpec objects."
    )


def prompt_multi_action(a1: str, a2: str, n: int) -> str:
    descs = {
        ("QUERY", "DELETE"): "find files matching a pattern, then delete them",
        ("QUERY", "MOVE"):   "find files matching a pattern, then move them to a destination",
        ("QUERY", "COPY"):   "find files matching a pattern, then copy them to a backup location",
        ("QUERY", "READ"):   "search for a specific file by name or pattern, then read its contents",
    }
    desc = descs.get((a1, a2), f"perform {a1} then {a2}")
    return (
        f"Generate {n} GoalSpec examples with exactly two actions:\n"
        f"  act-1: {a1}\n"
        f"  act-2: {a2} with depends_on: [\"act-1\"]\n"
        f"Pattern: {desc}.\n"
        f"Vary file types, glob patterns (*.log, *.tmp, *.bak, *.pdf, *.py, etc.), and directories.\n"
        f"Make intents sound natural, e.g.:\n"
        f"  'clean up all .tmp files in my home folder'\n"
        f"  'find old log files in /var/log and delete them'\n"
        f"  'move all PDFs from Downloads to Documents'\n\n"
        f"Return a JSON array of {n} GoalSpec objects."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GoalSpec training pairs via Claude API self-instruct"
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        metavar="N",
        help="Generate N validated pairs, print them, and exit without writing to disk",
    )
    args = parser.parse_args()

    seeds = load_seeds(SEED_FILE)
    schema = load_schema(SCHEMA_FILE)
    client = anthropic.Anthropic()

    pairs: list[dict] = []
    total_generated = 0
    total_rejected = 0

    # -----------------------------------------------------------------------
    # Dry-run mode
    # -----------------------------------------------------------------------
    if args.dry_run:
        n = args.dry_run
        print(f"--- DRY RUN: targeting {n} validated pairs ---\n")
        # Grab a seed with real actions
        seed = next(s for s in seeds if s.get("actions"))
        specs = call_api(client, prompt_seed_variants(seed, n + 5))
        total_generated += len(specs)
        added, rejected = process_batch(specs, schema, pairs, dry_run_limit=n)
        total_rejected += rejected

        for pair in pairs[:n]:
            print(json.dumps(pair, indent=2))
            print()

        print(f"--- {len(pairs[:n])}/{n} pairs printed  |  {total_generated} generated  |  {total_rejected} rejected ---")
        return

    # -----------------------------------------------------------------------
    # Full run
    # -----------------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Target: {TARGET} validated pairs → {OUTPUT_FILE}\n")

    # Phase 1: variants of every seed that has real actions
    actionable = [s for s in seeds if s.get("actions")]
    print(f"[Phase 1] Seed variants  ({len(actionable)} seeds × 15 each, ~{len(actionable)*15} raw)")
    for i, seed in enumerate(actionable):
        if len(pairs) >= TARGET:
            break
        try:
            specs = call_api(client, prompt_seed_variants(seed, 15))
            total_generated += len(specs)
            added, rejected = process_batch(specs, schema, pairs)
            total_rejected += rejected
            print(f"  seed {i+1:2d}/{len(actionable)}: +{added:2d} valid  {rejected:2d} rejected  → {len(pairs)} total")
        except Exception as e:
            print(f"  seed {i+1}: ERROR {e}", file=sys.stderr)

    # Phase 2: focused examples per action type
    print(f"\n[Phase 2] Per-action-type  (8 types × 25 each, ~200 raw)")
    for action_type in ACTION_TYPES:
        if len(pairs) >= TARGET:
            break
        try:
            specs = call_api(client, prompt_action_type(action_type, 25))
            total_generated += len(specs)
            added, rejected = process_batch(specs, schema, pairs)
            total_rejected += rejected
            print(f"  {action_type:9s}: +{added:2d} valid  {rejected:2d} rejected  → {len(pairs)} total")
        except Exception as e:
            print(f"  {action_type}: ERROR {e}", file=sys.stderr)

    # Phase 3: multi-action combos (hardest cases for 3B model)
    print(f"\n[Phase 3] Multi-action combos  (4 combos × 40 each, ~160 raw)")
    for a1, a2 in MULTI_ACTION_COMBOS:
        if len(pairs) >= TARGET:
            break
        try:
            specs = call_api(client, prompt_multi_action(a1, a2, 20))
            total_generated += len(specs)
            added, rejected = process_batch(specs, schema, pairs)
            total_rejected += rejected
            print(f"  {a1}+{a2}: +{added:2d} valid  {rejected:2d} rejected  → {len(pairs)} total")
        except Exception as e:
            print(f"  {a1}+{a2}: ERROR {e}", file=sys.stderr)

    # Phase 4: top-up until TARGET reached
    if len(pairs) < TARGET:
        print(f"\n[Phase 4] Top-up  (need {TARGET - len(pairs)} more)")
        topup_cycle = ["QUERY", "DELETE", "MOVE", "COPY", "READ", "WRITE"]
        consecutive_empty = 0
        i = 0
        while len(pairs) < TARGET:
            action = topup_cycle[i % len(topup_cycle)]
            i += 1
            batch_n = min(TARGET - len(pairs) + 5, 25)
            try:
                specs = call_api(client, prompt_action_type(action, batch_n))
                total_generated += len(specs)
                added, rejected = process_batch(specs, schema, pairs)
                total_rejected += rejected
                print(f"  {action:9s}: +{added:2d} valid  {rejected:2d} rejected  → {len(pairs)} total")
                if added == 0:
                    consecutive_empty += 1
                    if consecutive_empty >= len(topup_cycle):
                        print("  Stopping top-up: no valid pairs in full cycle", file=sys.stderr)
                        break
                else:
                    consecutive_empty = 0
            except Exception as e:
                print(f"  {action}: ERROR {e}", file=sys.stderr)
                break

    # Write output
    with OUTPUT_FILE.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    reject_pct = 100.0 * total_rejected / max(total_generated, 1)
    print(f"\n{'='*52}")
    print(f"  Generated (raw):   {total_generated}")
    print(f"  Rejected (invalid):{total_rejected:5d}  ({reject_pct:.1f}%)")
    print(f"  Validated pairs:   {len(pairs)}")
    print(f"  Written to:        {OUTPUT_FILE}")
    if len(pairs) < TARGET:
        print(f"  WARNING: only {len(pairs)}/{TARGET} target pairs reached", file=sys.stderr)


if __name__ == "__main__":
    main()
