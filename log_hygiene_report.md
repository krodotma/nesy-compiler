# Log Hygiene + Size Report (Pre/Post)

## Scope
- NDJSON/log caps <= 100MB
- Primary paths: /pluribus/.pluribus/meta/events.ndjson, /pluribus/.pluribus/index/task_ledger.ndjson, /pluribus/.pluribus/bus/events.ndjson

## Before (pre-trim)
- /pluribus/.pluribus/meta/events.ndjson: ~101MB
- /pluribus/.pluribus/index/task_ledger.ndjson: ~101MB
- /pluribus/.pluribus/bus/events.ndjson: ~2.4MB

## After (post-trim + watch)
- /pluribus/.pluribus/meta/events.ndjson: 100MB
- /pluribus/.pluribus/index/task_ledger.ndjson: 100MB
- /pluribus/.pluribus/bus/events.ndjson: 2.4MB
- Watcher: /pluribus/nucleus/tools/log_hygiene_watch.sh (PID 2416235)
  - Interval: 300s
  - Max: 100MB
  - Log: /pluribus/agent_logs/log_hygiene_watch.log

## du -sm /pluribus
- Attempted `du -sm /pluribus` (PID 2412872) is stuck in D-state.
- Second attempt (2026-01-26) also timed out (10s) with exit code 124.
- Action: defer and retry when IO pressure clears; avoid additional heavy scans.

## Notes
- Log hygiene watch is running; no services were stopped.
- Bus events published for hygiene start (operator.hygiene).
