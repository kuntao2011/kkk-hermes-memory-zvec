# Changelog

All notable changes to the `memory-zvec` plugin are documented here. The fork
baseline is the author's 2026-09-11 Zvec port of `memory-lancedb`.

## [1.3.2] - 2026-09-22

- Background threads (idle watchdog, prefetch, `sync_turn`, session-end
  batch) now follow the upstream `spawn_context_thread` contract ("never a
  bare `threading.Thread`") — the helper is vendored inline as
  `_spawn_thread` so older cores without the upstream symbol keep working;
  keeps contextvars tenant-correct under multi-profile reuse.
- Manifest dependency key corrected to `python_dependencies` (the key Hermes
  actually consumes for auto-install; the previous `dependencies:` key was
  silently ignored).

## [1.3.1] - 2026-09-19

Adversarial-review fixes:

- `_read_only` is reset on every acquire ladder — a degraded open used to keep
  writes skipped forever.
- `on_session_switch()` rebinds `_session_id`: `/new` previously kept the first
  session's id forever, misattributing the boundary batch (which now also
  snapshots the id before going async).
- Read paths pin a local collection reference against idle-release races.
- Lazy re-opens use a short lock ladder (~5.5s worst case instead of 22.5s).
- Skipped writes log throttled warnings (the session-end batch re-covers them).
- The prefetch cache is bounded (128 entries).
- `/new` on an existing handle skips the Ollama warm-up round-trip.

## [1.3.0] - 2026-09-19

- `shutdown()` no longer closes the collection: the handle survives session
  boundaries, so a new session never drain-waits for the previous session's
  end-of-session batch (in-process writes already serialize on `_insert_lock`).
  Real release happens only via the idle watchdog or process exit.

## [1.2.0] - 2026-09-14

- Idle lock release: a watchdog frees any collection idle past
  `HERMES_MEMORY_ZVEC_IDLE_RELEASE_S` (default 30s; reacts within ~5s) —
  messaging-platform sessions and dashboard-embedded profile chats previously
  held the mutex LOCK forever. The next access re-opens lazily via
  `_ensure_open()`.

## [1.1.0] - 2026-09-14

- `initialize()` drains the previous session's in-flight writes before opening,
  closing the back-to-back-session lock race.
- Externally-held locks are retried with backoff instead of degrading at once.

## [1.0.0] - 2026-09-11 (fork baseline)

- Zvec port of the author's `memory-lancedb` backend: in-process RocksDB
  storage, native hybrid query (one `MultiQuery` call with `RRFReRanker`
  instead of two-pass application-layer RRF), index-level scalar pre-filtering
  (`InvertIndexParam`) instead of post-ANN filtering, RocksDB-native FTS
  instead of Tantivy. Same 5 tool schemas and lifecycle hooks as
  `memory-lancedb`; ~13k docs/s batch insert and sub-ms search under 100k
  docs in the author's testing.
