Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Marshal eval harness — GBNF + Layer1
Timeout: 60s per intent | Cooldown: off (--fast)


Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]
Loading weights:  40%|███▉      | 41/103 [00:00<00:00, 234.50it/s]
Loading weights:  73%|███████▎  | 75/103 [00:00<00:00, 277.16it/s]
Loading weights: 100%|██████████| 103/103 [00:00<00:00, 359.39it/s]
[1mBertModel LOAD REPORT[0m from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status     |  | 
------------------------+------------+--+-
embeddings.position_ids | UNEXPECTED |  | 

[3mNotes:
- UNEXPECTED[3m	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.[0m
Running 29 test cases...

  [FAIL] read file            schema,category,action_type 90547ms
         error: [INFERENCE_TIMEOUT] The inference server timed out. Try again or restart it.
  [PASS] rename file          ok                   4ms
  [PASS] find PDFs            ok                   43959ms
  [PASS] list Python files    ok                   31131ms
  [PASS] find large files     ok                   35044ms
  [PASS] move files           ok                   37250ms
  [PASS] copy file            ok                   33820ms
  [PASS] delete tmp files     ok                   32664ms
  [PASS] delete by pattern    ok                   34426ms
  [PASS] find-then-move       ok                   35879ms
  [PASS] email (not impl)     not-impl             0ms
  [PASS] system (not impl)    not-impl             0ms
  [PASS] open config file     ok                   31385ms
  [PASS] read meeting notes   ok                   29709ms
  [PASS] count downloads      ok                   29327ms
  [PASS] find hidden files    ok                   29074ms
  [PASS] find log files       ok                   30482ms
  [PASS] recent downloads     ok                   28805ms
  [PASS] move screenshot      ok                   2ms
  [PASS] rename draft         ok                   2ms
  [PASS] backup config        ok                   2ms
  [PASS] backup directory     ok                   34136ms
  [PASS] delete single file   ok                   21387ms
  [PASS] create file          ok                   23443ms
  [PASS] find-then-read       ok                   29679ms
  [PASS] find-then-copy       ok                   37218ms
  [PASS] web search (not impl) not-impl             1ms
  [PASS] writing (not impl)   not-impl             0ms
  [PASS] cpu temp (not impl)  not-impl             0ms

============================================================
Results — GBNF + Layer1
============================================================

  Regular intents (24 cases):
    Schema validity :  23/24 (95%)
    Category correct:  23/24 (95%)
    Action type ok  :  23/24 (95%)
    Action ordering :  24/24 (100%)
    All checks pass :  23/24 (95%)

  Not-implemented intents (5 cases):
    Correctly rejected: 5/5 (100%)

  Phase 1 gate:
    schema_validity >= 95%  : PASS (95%)
    action_type_ok  >= 85%  : PASS (95%)
    action_ordering >= 85%  : PASS (100%)

  *** PHASE 1 GATE PASSED ***
