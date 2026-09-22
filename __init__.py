"""Zvec Vector Memory Provider — a community memory backend for Hermes Agent.

In-process vector storage (Zvec: HNSW + RocksDB FTS) with native hybrid query
(vector similarity + scalar filter + FTS via MultiQuery + RRFReRanker) and
embeddings from any Ollama-compatible endpoint via `base_url` (local Ollama
bge-m3 by default; 1024-dim — same dimension as memory-lancedb, so existing
vectors migrate without re-embedding).

Lock governance for long-lived sessions — the focus of this port (full
history in CHANGELOG.md):
  v1.1.0 — initialize() drains the previous session's in-flight writes before
            opening (closes the back-to-back-session lock race) and retries
            externally-held locks with backoff instead of degrading at once.
  v1.2.0 — idle lock release: a watchdog frees any collection idle past
            HERMES_MEMORY_ZVEC_IDLE_RELEASE_S (default 30s; reacts within
            ~5s) — long-lived messaging/dashboard sessions otherwise hold
            the mutex LOCK forever; the next access re-opens lazily.
  v1.3.0 — shutdown() keeps the collection open across session boundaries:
            a new session never drain-waits for the previous session's
            end-of-session batch. Real release happens only via the idle
            watchdog or process exit.
  v1.3.1 — adversarial-review fixes: _read_only reset on every acquire
            ladder; /new session-id rebind; read paths pin a local
            collection ref against idle-release races; short lock ladder
            (~5.5s worst); bounded prefetch cache (128); throttled
            skipped-write warnings.
  v1.3.2 — background threads (idle watchdog, prefetch, sync_turn,
            session-end batch) spawn via _spawn_thread, following the
            upstream spawn_context_thread contract (contextvars-correct
            under multi-profile reuse) instead of bare threading.Thread;
            vendored inline so older cores without the upstream helper
            keep working.

The config section is plugins.memory-zvec. Same 5 tool schemas and lifecycle
hooks as memory-lancedb — switching backends is a config.yaml edit plus a
one-time data migration.

README.md: install & configuration. CHANGELOG.md: full version history.
Migrating from memory-lancedb:
https://github.com/kuntao2011/kkk-hermes-zvec-memory-migration

Community port by kuntao2011. MIT licensed. Tested with Zvec 0.6.0 / Python 3.11.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import os
import re
import threading
import time
import uuid
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_EMBEDDING_TIMEOUT = 60  # seconds per batch

# Keep the HNSW graph fresh on the HOT write path. `sync_turn` never called
# optimize(), so index_completeness lagged badly on busy profiles (observed
# 0.71) while `enable_hnsw_optimize` looked like it was doing the job — it
# only ran inside on_session_end. Amortize: one optimize() per N writes.
_OPTIMIZE_EVERY_N_WRITES = 64

# ---------------------------------------------------------------------------
# Exit safety net for in-flight writes.
#
# sync_turn() / on_session_end() do their real work on daemon threads, so a
# process that exits right after starting them (cron external workers are the
# canonical case: agent.close() -> on_session_end, then the interpreter exits
# within ~1s) kills the embed+insert mid-flight and the memory is lost with no
# error anywhere. Proven reproducible: a write started and then immediately
# followed by process exit lands 0 rows.
#
# atexit handlers run BEFORE daemon threads are torn down, so a bounded join
# here gives those writes a chance to finish. 0 disables the wait (useful for
# A/B testing and for ops who would rather exit fast).
# ---------------------------------------------------------------------------
_EXIT_DRAIN_TIMEOUT_S = float(os.environ.get("HERMES_MEMORY_ZVEC_EXIT_DRAIN_S", "30"))

# Same drain, but for session START (fork addition, 2026-09-14): initialize()
# waits for the previous session's on_session_end batch writer (tracked in
# _PENDING_WRITE_THREADS above) before opening the collection. zvec 0.6.0
# LOCKs are fully mutex — an open attempted while that writer runs fails
# read-write AND read-only, which used to leave the new session silently
# memoryless for its whole lifetime (observed 2026-09-14 on chip_expert).
# 0 disables the wait.
_INIT_DRAIN_TIMEOUT_S = float(os.environ.get("HERMES_MEMORY_ZVEC_INIT_DRAIN_S", "15"))

_PENDING_WRITE_THREADS: set = set()
_PENDING_WRITE_LOCK = threading.Lock()


def _track_write_thread(thread: threading.Thread) -> None:
    with _PENDING_WRITE_LOCK:
        _PENDING_WRITE_THREADS.add(thread)


def _untrack_write_thread(thread: threading.Thread) -> None:
    with _PENDING_WRITE_LOCK:
        _PENDING_WRITE_THREADS.discard(thread)


def _drain_pending_writes(timeout: Optional[float] = None, context: str = "interpreter exit") -> None:
    """Wait (bounded) for background memory writes still in flight.

    Registered at atexit AND called from initialize() (with its own timeout)
    so a new session outlives the previous session's writer before opening.
    """
    limit = _EXIT_DRAIN_TIMEOUT_S if timeout is None else timeout
    if limit <= 0:
        return
    deadline = time.monotonic() + limit
    while True:
        with _PENDING_WRITE_LOCK:
            alive = [t for t in _PENDING_WRITE_THREADS if t.is_alive()]
            _PENDING_WRITE_THREADS.clear()
            _PENDING_WRITE_THREADS.update(alive)
        if not alive:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "Zvec: %d memory write(s) still in flight at %s; "
                "waited %.0fs, giving up (those turns are lost)",
                len(alive), context, limit,
            )
            return
        for t in alive:
            t.join(timeout=min(remaining, 5.0))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break


atexit.register(_drain_pending_writes)


# ---------------------------------------------------------------------------
# Idle lock release (fork addition, v1.2.0).
#
# The provider holds the Zvec LOCK for as long as the collection object is
# referenced — by design across a whole session. But "session" outlives
# "activity": messaging sessions stay alive indefinitely, and dashboard-
# embedded profile chats never call shutdown() at all, so locks sat held for
# hours or days (observed 2026-09-14: the root-profile dashboard process held
# zunhunfan's LOCK for 35h after one embedded chat, blocking that profile's
# gateway sessions). The watchdog below releases any collection idle past
# _IDLE_RELEASE_S; the next access re-opens lazily (see _ensure_open).
# Re-opening a warm collection is cheap (tens of ms, mmap), so the default is
# deliberately tight: the lock is gone ~30s after the last write lands, and
# mid-conversation pauses longer than that free the lock too. A zero-grace
# release would thrash open/close on rapid back-and-forth turns — 30s is the
# sweet spot. HERMES_MEMORY_ZVEC_IDLE_RELEASE_S=0 disables.
# ---------------------------------------------------------------------------
_IDLE_RELEASE_S = float(os.environ.get("HERMES_MEMORY_ZVEC_IDLE_RELEASE_S", "30"))
_IDLE_WATCHDOG_INTERVAL_S = (
    min(5.0, max(1.0, _IDLE_RELEASE_S / 3.0)) if _IDLE_RELEASE_S > 0 else 15.0
)
_REOPEN_RETRY_INTERVAL_S = 30.0
_PROVIDER_REGISTRY = weakref.WeakSet()
_PROVIDER_REGISTRY_LOCK = threading.Lock()
_IDLE_WATCHDOG_THREAD: Optional[threading.Thread] = None


def _idle_watchdog_loop() -> None:
    while True:
        time.sleep(_IDLE_WATCHDOG_INTERVAL_S)
        if _IDLE_RELEASE_S <= 0:
            continue
        now = time.monotonic()
        with _PENDING_WRITE_LOCK:
            writes_in_flight = any(t.is_alive() for t in _PENDING_WRITE_THREADS)
        with _PROVIDER_REGISTRY_LOCK:
            providers = list(_PROVIDER_REGISTRY)
        for p in providers:
            try:
                if p._coll is None:
                    continue
                idle_for = now - p._last_use
                if idle_for < _IDLE_RELEASE_S:
                    continue
                # In-flight writers pin their own collection reference; letting
                # them finish is both safer and releases the lock for real.
                if writes_in_flight:
                    continue
                logger.info(
                    "Zvec: idle %.0fs (>= %.0fs) — releasing collection lock (%s)",
                    idle_for, _IDLE_RELEASE_S,
                    p._open_params.get("zvec_dir", "?"),
                )
                p._release_collection()
            except Exception as e:
                logger.debug("idle watchdog: %s", e)


def _ensure_idle_watchdog() -> None:
    global _IDLE_WATCHDOG_THREAD
    with _PROVIDER_REGISTRY_LOCK:
        if _IDLE_WATCHDOG_THREAD is not None and _IDLE_WATCHDOG_THREAD.is_alive():
            return
        _IDLE_WATCHDOG_THREAD = _spawn_thread(
            target=_idle_watchdog_loop, daemon=True, name="zvec-idle-watchdog"
        )
        _IDLE_WATCHDOG_THREAD.start()


# ============================================================================
# Ollama /api/embed helper (no Python ollama package needed)
# Identical to memory-lancedb — keeps the swap transparent.
# ============================================================================

# Context-correct background thread spawning. Vendored from upstream
# agent.memory_provider.spawn_context_thread (the memory-provider contract:
# "never a bare threading.Thread"); inlined so the plugin also runs on cores
# that predate that helper. Runs the target under the spawner's contextvars.
def _spawn_thread(target, *, name, daemon=True):
    ctx = contextvars.copy_context()
    return threading.Thread(
        target=lambda *a, **k: ctx.run(target, *a, **k), name=name, daemon=daemon
    )


def _ollama_embed(texts: List[str], base_url: str, model: str) -> List[np.ndarray]:
    """Call Ollama /api/embed for batch embeddings. Returns list of np.float32 arrays."""
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/embed",
        json={"model": model, "input": texts},
        timeout=_EMBEDDING_TIMEOUT * 2,
    )
    resp.raise_for_status()
    return [np.array(e, dtype=np.float32) for e in resp.json()["embeddings"]]


def _ollama_embed_single(text: str, base_url: str, model: str) -> np.ndarray:
    """Embed a single text."""
    return _ollama_embed([text], base_url, model)[0]


# ============================================================================
# Zvec schema + helpers
# ============================================================================

def _get_zvec():
    """Lazy import — zvec is required at runtime, not import-time."""
    import zvec
    return zvec


def _build_memories_schema(vector_dim: int = 1024, collection_name: str = "memories"):
    """Build the memories collection schema.

    Field design:
      - content: STRING with FTSIndexParam (RocksDB-native FTS, tokenizer='jieba')
      - role: STRING (with InvertIndexParam — filterable by role)
      - session_id: STRING with InvertIndexParam (most common filter)
      - created_at: DOUBLE with InvertIndexParam(enable_range_optimization=True)
      - metadata: STRING (raw JSON; not indexed — would explode index size)
      - vector: VECTOR_FP32, dim=vector_dim, HNSW(COSINE)
    """
    zvec = _get_zvec()
    return zvec.CollectionSchema(
        name=collection_name,
        fields=[
            zvec.FieldSchema(
                "content", zvec.DataType.STRING,
                index_param=zvec.FtsIndexParam(tokenizer_name="jieba", filters=["lowercase"]),
            ),
            zvec.FieldSchema(
                "role", zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "session_id", zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),
            zvec.FieldSchema(
                "created_at", zvec.DataType.DOUBLE,
                index_param=zvec.InvertIndexParam(enable_range_optimization=True),
            ),
            zvec.FieldSchema("metadata", zvec.DataType.STRING),
        ],
        vectors=[
            zvec.VectorSchema(
                "vector", zvec.DataType.VECTOR_FP32, vector_dim,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.COSINE,
                    m=16,
                    ef_construction=100,
                ),
            ),
        ],
    )


# ============================================================================
# Tool schemas — IDENTICAL to memory-lancedb so user-facing behavior matches.
# ============================================================================

MEMORY_STORE_SCHEMA = {
    "name": "vec_memory_add",
    "description": (
        "Store a piece of information in vector memory for semantic retrieval. "
        "Use to remember facts, decisions, preferences, commands, or any content "
        "you want to recall later via natural-language queries.\n\n"
        "The content is embedded with bge-m3:latest into a 1024-dim vector and "
        "stored in Zvec with HNSW + FTS index. Retrieval uses native hybrid query "
        "(vector + scalar filter + FTS fused via RRFReRanker).\n\n"
        "Use vec_memory_search to retrieve, vec_memory_list to view stored items."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "role": {"type": "string", "description": "Role context: 'user', 'assistant', 'system' (default: '')."},
            "session_id": {"type": "string", "description": "Session this content belongs to (default: current session)."},
            "metadata": {"type": "string", "description": "Optional JSON metadata string."},
        },
        "required": ["content"],
    },
}

MEMORY_SEARCH_SCHEMA = {
    "name": "vec_memory_search",
    "description": (
        "Semantic search over stored vector memory using natural-language query. "
        "Converts the query to a vector with bge-m3:latest and returns the most "
        "similar stored items via hybrid search (vector + scalar filter + FTS fused "
        "via RRFReRanker). Use for recalling facts, decisions, preferences, and any "
        "stored context from previous sessions.\n\n"
        "Supports time-range filtering via after_timestamp / before_timestamp (Unix timestamps)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query."},
            "top_k": {"type": "integer", "description": "Max results to return (default: 5, max: 20)."},
            "min_score": {"type": "number", "description": "Minimum relevance threshold (0-1, default: 0.0)."},
            "session_id": {"type": "string", "description": "Limit search to a specific session (optional)."},
            "after_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at >= this (optional)."},
            "before_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at <= this (optional)."},
            "mode": {
                "type": "string",
                "enum": ["hybrid", "vector", "keyword"],
                "description": "Search mode: 'hybrid' (default) combines vector + keyword via RRF, 'vector' for semantic-only, 'keyword' for full-text-only."
            },
        },
        "required": ["query"],
    },
}

MEMORY_LIST_SCHEMA = {
    "name": "vec_memory_list",
    "description": (
        "List all stored memory items, optionally filtered by session. "
        "Returns id, content preview, role, session_id, timestamp.\n\n"
        "Supports time-range filtering via after_timestamp / before_timestamp (Unix timestamps)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "Max items to return (default: 20, max: 100)."},
            "session_id": {"type": "string", "description": "Filter by session ID (optional)."},
            "after_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at >= this (optional)."},
            "before_timestamp": {"type": "number", "description": "Unix timestamp — only return results with created_at <= this (optional)."},
        },
    },
}

MEMORY_DELETE_SCHEMA = {
    "name": "vec_memory_delete",
    "description": "Delete one or more memory items by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of memory IDs to delete.",
            },
        },
        "required": ["memory_ids"],
    },
}

MEMORY_STATS_SCHEMA = {
    "name": "vec_memory_stats",
    "description": "Show memory store statistics: total items, sessions, storage size.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ============================================================================
# Config helpers
# ============================================================================

def _load_plugin_config() -> dict:
    """Load memory-zvec plugin config, with backward-compat fallback.

    Primary key: plugins.memory-zvec (current).
    Fallback key: plugins.memory-lancedb (if user already had lancedb config).
    """
    from hermes_cli.config import cfg_get, load_config
    try:
        config = load_config()
    except Exception:
        return {}
    cfg = cfg_get(config, "plugins", "memory-zvec", default={}) or {}
    if not cfg:
        cfg = cfg_get(config, "plugins", "memory-lancedb", default={}) or {}
    return cfg


def _expand_hermes_home(path: str, hermes_home: str) -> str:
    """Expand $HERMES_HOME and ${HERMES_HOME} in path strings."""
    if not path:
        return path
    return path.replace("$HERMES_HOME", hermes_home).replace("${HERMES_HOME}", hermes_home)


# ============================================================================
# Filter building — Zvec SQL-style with single '='
# ============================================================================

def _build_filter(
    session_id: str = "",
    after_timestamp: Optional[float] = None,
    before_timestamp: Optional[float] = None,
    extra: str = "",
) -> str:
    """Build a Zvec filter expression (single '=' SQL-style).

    Returns "" when no filter conditions are present.
    Caller is responsible for quoting string values — this function only assembles
    the boolean expression and validates numeric ranges.
    """
    conditions: List[str] = []

    if session_id:
        # Escape single quotes in session_id (rare but possible)
        sid_escaped = session_id.replace("'", "''")
        conditions.append(f"session_id = '{sid_escaped}'")

    if after_timestamp is not None and after_timestamp > 0:
        conditions.append(f"created_at >= {float(after_timestamp)}")

    if before_timestamp is not None and before_timestamp > 0:
        conditions.append(f"created_at <= {float(before_timestamp)}")

    if extra:
        conditions.append(f"({extra})")

    return " AND ".join(conditions)


def _sanitize_content(content: str) -> str:
    """Strip control characters that could break FTS or filter parsing."""
    # Drop null bytes; collapse other control chars except \n \t
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)


# ============================================================================
# MemoryProvider implementation
# ============================================================================

class ZvecMemoryProvider(MemoryProvider):
    """Vector semantic memory via Ollama bge-m3:latest + Zvec (HNSW + FTS + native hybrid)."""

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._coll = None            # zvec.Collection
        self._read_only = False      # True when opened read-only due to lock conflict
        self._session_id = ""
        self._embedding_lock = threading.Lock()
        self._insert_lock = threading.Lock()    # serialize writes (Zvec is single-writer safe but consistent ordering helps)
        self._writes_since_optimize = 0          # drives amortized HNSW optimize() on the hot path
        # Cache: query -> (results, timestamp) for prefetch
        self._prefetch_cache: Dict[str, tuple] = {}
        self._prefetch_ttl = 30
        # Idle-release bookkeeping (v1.2.0)
        self._last_use = time.monotonic()
        self._last_open_attempt = 0.0
        self._last_open_failed = False
        self._last_skip_warn = 0.0
        self._reopen_lock = threading.Lock()
        self._open_params: Dict[str, Any] = {}
        with _PROVIDER_REGISTRY_LOCK:
            _PROVIDER_REGISTRY.add(self)
        _ensure_idle_watchdog()

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "memory-zvec"

    # -- Health / config -----------------------------------------------------

    def is_available(self) -> bool:
        """Check that both Zvec can be imported and Ollama is responsive.

        Should not do heavy I/O per the MemoryProvider contract — but Ollama
        health-check is the only reliable way to detect the embedding model is
        present. We swallow all errors to keep startup robust.
        """
        # 1. Can import zvec?
        try:
            _get_zvec()
        except Exception as e:
            logger.debug("Zvec memory provider unavailable (zvec import): %s", e)
            return False

        # 2. Ollama responsive?
        base_url = self._config.get("base_url", "http://localhost:11434")
        model = self._config.get("embedding_model", "bge-m3:latest")
        try:
            _ollama_embed_single("health check", base_url, model)
            return True
        except Exception as e:
            logger.debug("Zvec memory provider unavailable (ollama): %s", e)
            return False

    def get_config_schema(self):
        """Configuration keys surfaced to setup wizard / config UI."""
        return [
            {"key": "base_url",            "description": "Ollama server URL",                            "default": "http://localhost:11434"},
            {"key": "embedding_model",     "description": "Embedding model available in Ollama",           "default": "bge-m3:latest"},
            {"key": "vector_dim",          "description": "Embedding vector dimension (bge-m3 = 1024)",   "default": "1024"},
            {"key": "zvec_dir",            "description": "Zvec collection directory",                    "default": "$HERMES_HOME/记忆数据库/zvec_memory"},
            {"key": "collection_name",     "description": "Zvec collection name",                          "default": "memories"},
            {"key": "batch_size",          "description": "Max texts per embedding batch",                "default": "32"},
            {"key": "search_top_k",        "description": "Default top-k results per search",             "default": "5"},
            {"key": "min_content_len",     "description": "Skip content shorter than this (chars)",      "default": "50"},
            {"key": "vector_weight",       "description": "Hybrid search weight for vector branch",       "default": "0.7"},
            {"key": "fts_weight",          "description": "Hybrid search weight for FTS branch",          "default": "0.3"},
            {"key": "enable_hnsw_optimize","description": "Call collection.optimize() after bulk writes", "default": "true"},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Persist config to $HERMES_HOME/config.yaml under plugins.memory-zvec."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["memory-zvec"] = values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning("Failed to save memory-zvec config: %s", e)

    # -- Lifecycle -----------------------------------------------------------

    @staticmethod
    def _open_read_only(path, option):
        """Try to open a Zvec collection in read-only mode. Returns None on failure."""
        try:
            import zvec as _zvec
            return _zvec.open(str(path), option=option)
        except Exception as e:
            logger.error("ZvecMemoryProvider failed to open (read-only) at %s: %s", path, e)
            return None

    def _touch(self) -> None:
        """Mark provider activity (drives the idle watchdog)."""
        self._last_use = time.monotonic()

    def _warn_skip_throttled(self) -> None:
        """1-per-5-min warning that per-turn writes are being skipped.

        Cross-process blackout (another process holds the lock past the lazy
        backoff): turns are NOT lost — on_session_end re-extracts the full
        transcript at the next boundary — but the operator should see it.
        """
        now = time.monotonic()
        if now - self._last_skip_warn < 300.0:
            return
        self._last_skip_warn = now
        logger.warning(
            "Zvec: collection unavailable — per-turn writes skipped (1/5min); "
            "the session-end batch will re-store the full transcript",
        )

    def _acquire_collection(self, *, max_drain_s: Optional[float] = None,
                            backoff_delays: tuple = (0.5, 1.0, 2.0, 4.0)) -> None:
        """Open (or create) the collection using the full lock ladder.

        Drain in-process writers → open(rw) → backoff retry (external holder)
        → open(read-only) → create_and_open. Records the attempt time so
        _ensure_open can rate-limit re-tries after a failed acquisition.

        ``max_drain_s``/``backoff_delays`` let lazy re-opens (access paths:
        turn/prefetch) run a SHORT ladder so a cross-process block stalls the
        caller for seconds, not the full 15s+7.5s.
        """
        self._last_open_attempt = time.monotonic()
        # A fresh ladder starts from a clean rw assumption: _read_only may be
        # stale-True from an earlier degraded open (e.g. a past cross-process
        # lock block); without this reset, writes stay skipped forever even
        # after a later successful rw re-open.
        self._read_only = False
        zvec = _get_zvec()
        zvec_dir = self._open_params["zvec_dir"]
        collection_name = self._open_params["collection_name"]
        vector_dim = self._open_params["vector_dim"]

        _ro_option = zvec.CollectionOption(read_only=True, enable_mmap=True)
        schema = _build_memories_schema(vector_dim, collection_name)
        collection_path = Path(zvec_dir) / collection_name

        # Wait (bounded) for the previous session's background batch writer in
        # THIS process: it holds the Zvec LOCK and zvec 0.6.0 is fully mutex —
        # even read-only opens fail while it runs, which used to leave the new
        # session without memory for its whole lifetime (silent). The writer is
        # tracked at module level, so this catches it regardless of provider
        # instance.
        _drain_pending_writes(
            timeout=_INIT_DRAIN_TIMEOUT_S if max_drain_s is None else max_drain_s,
            context="collection acquire",
        )

        try:
            self._coll = zvec.open(str(collection_path))
            logger.info("ZvecMemoryProvider opened existing collection: %s", collection_path)
        except Exception as open_err:
            is_lock = "lock" in str(open_err).lower() or "LOCK" in str(open_err)

            if is_lock and collection_path.exists():
                # Drain above covers in-process holders; reaching here means an
                # EXTERNAL process holds the lock (health check, migration, ...).
                # Those are usually brief — retry with backoff before degrading.
                last_err: Optional[Exception] = open_err
                for delay in backoff_delays:
                    logger.info(
                        "ZvecMemoryProvider: lock held externally, retrying open in %.1fs...",
                        delay,
                    )
                    import time as _time
                    _time.sleep(delay)
                    try:
                        self._coll = zvec.open(str(collection_path))
                        last_err = None
                        logger.info(
                            "ZvecMemoryProvider opened collection after backoff retry: %s",
                            collection_path,
                        )
                        break
                    except Exception as retry_err:
                        last_err = retry_err
                if last_err is not None:
                    # Still locked → read-only fallback
                    logger.warning(
                        "ZvecMemoryProvider: still locked after backoff retries, "
                        "falling back to read-only: %s",
                        last_err,
                    )
                    self._coll = self._open_read_only(collection_path, _ro_option)
                    if self._coll:
                        self._read_only = True
                    else:
                        self._read_only = False
                        return
            elif collection_path.exists():
                # Non-lock error on existing path → read-only
                logger.warning("ZvecMemoryProvider: open failed (%s), trying read-only", open_err)
                self._coll = self._open_read_only(collection_path, _ro_option)
                if self._coll:
                    self._read_only = True
                else:
                    return
            else:
                # Path doesn't exist → create new collection
                try:
                    self._coll = zvec.create_and_open(str(collection_path), schema)
                    logger.info("ZvecMemoryProvider created new collection: %s", collection_path)
                except Exception as e:
                    logger.error("ZvecMemoryProvider failed to create collection at %s: %s",
                                 collection_path, e)
                    self._coll = None
                    return
        finally:
            self._last_open_failed = self._coll is None
        self._touch()

    def _ensure_open(self) -> None:
        """Lazily re-open after an idle-release (or retry a failed first open).

        No-op when already open. Re-open attempts are rate-limited to one per
        _REOPEN_RETRY_INTERVAL_S so a permanently-blocked collection degrades
        quietly instead of blocking every access on the full drain+backoff
        ladder.
        """
        self._touch()
        if self._coll is not None or not self._open_params:
            return
        # Rate-limit only RETRIES after a failed acquisition — a re-open after
        # an idle release (last attempt succeeded) must proceed immediately,
        # or reads/writes would be dropped for up to the interval.
        if (self._last_open_failed
                and time.monotonic() - self._last_open_attempt < _REOPEN_RETRY_INTERVAL_S):
            return
        with self._reopen_lock:
            if self._coll is not None:
                return
            # Lazy re-opens run on access paths (turn/prefetch) — keep the
            # worst-case stall short (~5.5s); failed attempts are rate-limited
            # to one per _REOPEN_RETRY_INTERVAL_S anyway.
            self._acquire_collection(max_drain_s=2.0, backoff_delays=(0.5, 1.0, 2.0))

    def initialize(self, session_id: str, **kwargs) -> None:
        """Connect to Zvec collection, ensure schema, warm up Ollama.

        v1.3.0: deliberately does NOT touch an already-open collection — the
        handle survives session boundaries (see shutdown()), so starting a new
        session while the previous one's end-of-session batch is still writing
        costs nothing; the new session's writes simply queue behind it on
        _insert_lock.
        """
        from hermes_constants import get_hermes_home

        hermes_home = kwargs.get("hermes_home", str(get_hermes_home()))
        base_url = self._config.get("base_url", "http://localhost:11434")
        model = self._config.get("embedding_model", "bge-m3:latest")
        vector_dim = int(self._config.get("vector_dim", 1024))
        zvec_dir = _expand_hermes_home(
            self._config.get("zvec_dir", f"{hermes_home}/记忆数据库/zvec_memory"),
            hermes_home,
        )
        collection_name = self._config.get("collection_name", "memories")

        Path(zvec_dir).mkdir(parents=True, exist_ok=True)

        new_params = {
            "zvec_dir": zvec_dir,
            "collection_name": collection_name,
            "vector_dim": vector_dim,
        }
        # Reuse the already-open handle across session boundaries (v1.3.0): a
        # second zvec.open() of the same path would collide with our OWN lock.
        # Only reopen when closed, or when the target DB itself changed.
        if (self._coll is not None and self._open_params
                and self._open_params != new_params):
            logger.info("ZvecMemoryProvider: target changed %s -> %s, switching handles",
                        self._open_params.get("zvec_dir"), zvec_dir)
            self._release_collection()
        reused_handle = self._coll is not None
        self._open_params = new_params
        if self._coll is None:
            self._acquire_collection()

        self._session_id = session_id

        if reused_handle:
            # Handle survived the session boundary — model is already warm;
            # /new shouldn't pay the Ollama round-trip again.
            logger.info(
                "ZvecMemoryProvider re-initialized on existing handle — model=%s dim=%d dir=%s coll=%s",
                model, vector_dim, zvec_dir, collection_name,
            )
            return
        # Warm up Ollama (loads model into GPU/RAM, surfaces import errors early)
        try:
            _ollama_embed_single("warmup", base_url, model)
            logger.info(
                "ZvecMemoryProvider initialized — model=%s dim=%d dir=%s coll=%s",
                model, vector_dim, zvec_dir, collection_name,
            )
        except Exception as e:
            logger.warning("ZvecMemoryProvider warmup failed: %s", e)

    def system_prompt_block(self) -> str:
        """Static block describing the active memory system for the system prompt."""
        coll = self._coll
        if not coll:
            return ""
        try:
            total = coll.stats.doc_count
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Ollama Vector Memory\n"
                "Active (Zvec backend). Empty vector store — use vec_memory_add to store facts, "
                "preferences, decisions, commands, or any content you want to recall later.\n"
                "Use vec_memory_search for semantic retrieval, vec_memory_list to view stored items."
            )
        return (
            f"# Ollama Vector Memory\n"
            f"Active (Zvec backend). {total} items stored with bge-m3:latest embeddings "
            f"(1024-dim) via Zvec HNSW + FTS.\n"
            f"Hybrid search fuses vector + scalar filter + FTS via RRFReRanker.\n"
            f"Use vec_memory_add to store, vec_memory_search to retrieve, "
            f"vec_memory_list to browse, vec_memory_stats for overview."
        )

    def shutdown(self) -> None:
        """Session-boundary bookkeeping close (agent/session lifecycle hook).

        v1.3.0: intentionally keeps the collection OPEN. Releasing it here is
        what forced the NEXT session's initialize() to drain-wait for this
        session's end-of-session batch writer (the back-to-back-session lock
        race, re-introduced at every session boundary). In-process writes
        already serialize on _insert_lock, so one shared handle is safe; the
        real release happens in _release_collection(), invoked by the idle
        watchdog (and implicitly at process exit — fds close, LOCK drops).
        """
        return

    def _release_collection(self) -> None:
        """Actually drop the collection and release the Zvec LOCK.

        Only the idle watchdog calls this. Background writers (sync_turn /
        on_session_end) pin their own collection reference, so they complete
        and release the LOCK when they finish; nulling the attribute keeps
        every ``if not self._coll`` guard working for them.
        """
        coll = self._coll
        self._coll = None
        if coll is not None:
            try:
                del coll
                import gc
                gc.collect()  # ensure Zvec's LOCK file is released
            except Exception:
                pass

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "",
        reset: bool = False, rewound: bool = False, **kwargs,
    ) -> None:
        """Rebind per-session state (base-class contract): /new, /resume, /branch...

        Without this, every post-switch write lacking an explicit session_id
        kept the FIRST session's id forever — and the end-of-session batch
        (which snapshots _session_id at entry) misattributed transcripts
        after /new.
        """
        self._session_id = new_session_id

    # -- Recall hooks --------------------------------------------------------

    def _cache_prefetch(self, key: str, results: List[Dict[str, Any]]) -> None:
        """Store prefetch results, keeping the cache bounded (gateway workers
        live for days; an uncapped dict grew without limit)."""
        self._prefetch_cache[key] = (results, datetime.now().timestamp())
        if len(self._prefetch_cache) <= 128:
            return
        now_ts = datetime.now().timestamp()
        kept = {k: v for k, v in self._prefetch_cache.items()
                if now_ts - v[1] < self._prefetch_ttl}
        if len(kept) > 128:
            kept = dict(list(kept.items())[-128:])  # keep most recent inserts
        self._prefetch_cache = kept

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Synchronous recall — return cached or fresh top-3 results as text."""
        self._ensure_open()
        if not query or not self._coll:
            return ""
        cache_key = query[:200]
        if cache_key in self._prefetch_cache:
            results, cached_at = self._prefetch_cache[cache_key]
            if datetime.now().timestamp() - cached_at < self._prefetch_ttl:
                return self._format_results(results[:3])
            del self._prefetch_cache[cache_key]
        try:
            results = self._do_search(query, top_k=3, session_id=session_id)
            self._cache_prefetch(cache_key, results)
            return self._format_results(results)
        except Exception as e:
            logger.debug("prefetch search failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background prefetch — fire-and-forget thread, caches results for the next turn."""
        self._ensure_open()
        if not query or not self._coll:
            return

        def _bg():
            try:
                self._ensure_open()  # entry-time handle may have been idle-released meanwhile
                results = self._do_search(query, top_k=3, session_id=session_id)
                self._cache_prefetch(query[:200], results)
            except Exception:
                pass

        t = _spawn_thread(target=_bg, daemon=True, name="zvec-prefetch")
        t.start()

    # -- Write hooks ---------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "",
                  timestamp: float = None) -> None:
        """Persist one turn (user + assistant) as a single memory row."""
        if not user_content.strip() and not assistant_content.strip():
            return
        self._ensure_open()
        if not self._coll:
            self._warn_skip_throttled()
            return
        if self._read_only:
            logger.warning("Zvec: sync_turn skipped (read-only mode) — this turn is NOT being remembered")
            return
        sid = session_id or self._session_id
        # Pin the collection for the background writer (see _insert docstring):
        # agent close calls shutdown() while this thread may still be running.
        coll = self._coll
        ts = timestamp if isinstance(timestamp, (int, float)) and timestamp > 0 \
            else datetime.now().timestamp()
        ts_iso = datetime.fromtimestamp(ts).isoformat()
        meta = json.dumps({
            "user_preview": user_content[:100],
            "asst_preview": assistant_content[:100],
            "message_timestamps": [ts_iso],
        }, ensure_ascii=False)

        def _store():
            try:
                combined = f"[user]\n{user_content}\n[assistant]\n{assistant_content}"
                if len(combined) < int(self._config.get("min_content_len", 50)):
                    return
                with self._embedding_lock:
                    vec = _ollama_embed_single(combined, self.base_url, self.model)
                self._insert(combined=combined, role="turn", sid=sid,
                             ts=ts, vec=vec, meta=meta, coll=coll)
            except Exception as e:
                # Was logger.debug: an Ollama outage or a bad embed silently lost
                # every turn with zero operator-visible signal. Surface it.
                logger.warning("sync_turn store failed (turn NOT remembered): %s", e)
            finally:
                _untrack_write_thread(threading.current_thread())

        t = _spawn_thread(target=_store, daemon=True, name="zvec-sync")
        _track_write_thread(t)
        t.start()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session extraction — batch-store all user/assistant pairs."""
        self._ensure_open()  # idle watchdog may have released; the final batch must land
        if not self._coll:
            logger.warning("Zvec: on_session_end skipped — collection unavailable "
                           "(lock held elsewhere); session history NOT remembered this time")
            return
        if self._read_only:
            logger.warning("Zvec: on_session_end skipped (read-only mode) — session history is NOT being remembered")
            return
        assert self._coll is not None  # for type checkers
        # Hold our own strong reference for the background thread: shutdown()
        # (called on agent close, possibly while this batch is still running)
        # releases the provider's reference, and a bare self._coll lookup inside
        # the thread would then fail with AttributeError/NoneType and lose the
        # whole batch. With a local ref the batch completes and the Zvec LOCK is
        # released as soon as the object's last reference goes away.
        coll = self._coll
        # Snapshot the session id at boundary time: on_session_switch (fired
        # right after this returns, serialized by the manager) rebinds
        # _session_id to the NEXT session; the async batch below must still
        # land under the OLD one.
        sid = self._session_id
        if not messages:
            return
        pairs = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                content = msg["content"].strip()
                if len(content) < 10:
                    continue
                assistant_content = ""
                assistant_ts = None
                if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                    assistant_content = messages[i + 1].get("content", "") or ""
                    assistant_ts = messages[i + 1].get("timestamp")
                user_ts = msg.get("timestamp")
                pairs.append({
                    "user_content": content,
                    "asst_content": assistant_content,
                    "user_ts": user_ts,
                    "asst_ts": assistant_ts,
                })
        if not pairs:
            return

        def _embed_with_retry(text: str, retries: int = 1, backoff: float = 1.0) -> Optional[np.ndarray]:
            """Embed a single text with one retry on transient failure."""
            try:
                return _ollama_embed_single(text, self.base_url, self.model)
            except Exception as first_err:
                logger.warning(
                    "session_end embed failed (will retry in %.1fs): %s",
                    backoff, first_err,
                )
                import time as _time
                _time.sleep(backoff)
                try:
                    return _ollama_embed_single(text, self.base_url, self.model)
                except Exception as retry_err:
                    logger.warning("session_end embed retry also failed: %s", retry_err)
                    return None

        def _batch_store():
            try:
                min_len = int(self._config.get("min_content_len", 50))
                now_ts = datetime.now().timestamp()
                stored = 0
                skipped = 0
                with self._insert_lock:
                    for pair in pairs:
                        u = pair["user_content"]
                        a = pair["asst_content"]
                        u_ts = pair["user_ts"]
                        a_ts = pair["asst_ts"]
                        combined = f"[user]\n{u}\n[assistant]\n{a}"

                        ts_list = []
                        if u_ts:
                            try:
                                ts_list.append(datetime.fromtimestamp(float(u_ts)).isoformat())
                            except Exception:
                                pass
                        if a_ts:
                            try:
                                ts_list.append(datetime.fromtimestamp(float(a_ts)).isoformat())
                            except Exception:
                                pass

                        meta = {
                            "user_preview": u[:100],
                            "asst_preview": a[:100] if a else "",
                            "message_timestamps": ts_list,
                        }
                        row_ts = u_ts if isinstance(u_ts, (int, float)) and u_ts > 0 else now_ts

                        if len(combined) < min_len:
                            skipped += 1
                            continue

                        vec = _embed_with_retry(combined)
                        if vec is None:
                            skipped += 1
                            continue

                        doc = self._make_doc(
                            id_value=str(uuid.uuid4()),
                            content=combined,
                            role="session_end",
                            sid=sid,
                            ts=row_ts,
                            vec=vec,
                            meta=json.dumps(meta, ensure_ascii=False),
                        )
                        coll.insert(doc)
                        coll.flush()
                        stored += 1

                    # Optimize HNSW once after all inserts
                    if stored > 0 and self._config.get("enable_hnsw_optimize", True):
                        try:
                            coll.optimize()
                        except Exception:
                            pass

                if stored or skipped:
                    logger.info(
                        "Zvec: session-end stored=%d, skipped=%d (short/embed-fail)",
                        stored, skipped,
                    )
            except Exception as e:
                logger.warning("session_end batch store failed: %s", e)
            finally:
                _untrack_write_thread(threading.current_thread())

        t = _spawn_thread(target=_batch_store, daemon=True, name="zvec-session-end")
        _track_write_thread(t)
        t.start()

    # -- Tool registration ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            MEMORY_STORE_SCHEMA,
            MEMORY_SEARCH_SCHEMA,
            MEMORY_LIST_SCHEMA,
            MEMORY_DELETE_SCHEMA,
            MEMORY_STATS_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        self._ensure_open()
        if tool_name == "vec_memory_add":
            return self._tool_add(args)
        elif tool_name == "vec_memory_search":
            return self._tool_search(args)
        elif tool_name == "vec_memory_list":
            return self._tool_list(args)
        elif tool_name == "vec_memory_delete":
            return self._tool_delete(args)
        elif tool_name == "vec_memory_stats":
            return self._tool_stats(args)
        return tool_error(f"Unknown tool: {tool_name}")

    # -- Config property accessors -------------------------------------------

    @property
    def base_url(self) -> str:
        return self._config.get("base_url", "http://localhost:11434")

    @property
    def model(self) -> str:
        return self._config.get("embedding_model", "bge-m3:latest")

    @property
    def vector_dim(self) -> int:
        return int(self._config.get("vector_dim", 1024))

    @property
    def batch_size(self) -> int:
        return int(self._config.get("batch_size", 32))

    @property
    def search_top_k(self) -> int:
        return int(self._config.get("search_top_k", 5))

    @property
    def vector_weight(self) -> float:
        return float(self._config.get("vector_weight", 0.7))

    @property
    def fts_weight(self) -> float:
        return float(self._config.get("fts_weight", 0.3))

    # -- Internal: row builders ---------------------------------------------

    def _make_doc(self, *, id_value: str, content: str, role: str, sid: str,
                  ts: float, vec: np.ndarray, meta: str):
        """Build a zvec.Doc with the standard memories schema."""
        zvec = _get_zvec()
        return zvec.Doc(
            id=id_value,
            vectors={"vector": vec.tolist()},
            fields={
                "content":    _sanitize_content(content or ""),
                "role":       role or "turn",
                "session_id": sid or "",
                "created_at": float(ts) if ts else 0.0,
                "metadata":   meta or "{}",
            },
        )

    def _insert(self, *, combined: str, role: str, sid: str, ts: float,
                vec: np.ndarray, meta: str, coll=None) -> str:
        """Synchronous single-row insert. Returns the new id.

        ``coll`` lets a background writer (sync_turn / on_session_end) pin its own
        reference to the collection, so a concurrent ``shutdown()`` — which nulls
        the provider attribute on agent close — cannot pull the collection out
        from under an in-flight write.
        """
        target = coll if coll is not None else self._coll
        if target is None:
            raise RuntimeError("collection not available (shutdown raced this write)")
        mem_id = str(uuid.uuid4())
        doc = self._make_doc(id_value=mem_id, content=combined, role=role, sid=sid,
                             ts=ts, vec=vec, meta=meta)
        with self._insert_lock:
            target.insert(doc)
            target.flush()
            # Amortized HNSW maintenance: without this, vectors sit in the flat
            # buffer and index_completeness only recovers on on_session_end.
            self._writes_since_optimize += 1
            if (self._writes_since_optimize >= _OPTIMIZE_EVERY_N_WRITES
                    and self._config.get("enable_hnsw_optimize", True)):
                try:
                    target.optimize()
                    self._writes_since_optimize = 0
                except Exception as opt_err:
                    logger.debug("amortized optimize failed: %s", opt_err)
        return mem_id

    # -- Internal: search implementations ------------------------------------

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string (with the embedding lock held)."""
        with self._embedding_lock:
            return _ollama_embed_single(query, self.base_url, self.model)

    def _do_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        session_id: str = "",
        after_timestamp: float = None,
        before_timestamp: float = None,
    ) -> List[Dict[str, Any]]:
        """Vector-only HNSW search via Zvec.

        Returns scored list of {id, content, role, session_id, created_at, score, metadata}.

        Score convention: SIMILARITY (higher = more similar), i.e. cosine similarity.

        ⚠️ Zvec reports the vector branch's ``score`` as a DISTANCE, not a
        similarity — verified empirically on a 3-doc collection with COSINE:
        identical vector → 0.000000, near vector → 0.006116, orthogonal →
        1.000000 (distance = 1 - cosine_similarity). The FTS branch reports a
        SIMILARITY (more term hits → higher). We normalize here so ranking,
        ``min_score`` and the hybrid fallback all share one convention —
        otherwise ``sort(reverse=True)`` surfaces the LEAST relevant memories
        first and ``min_score`` keeps the worst ones.
        """
        query_vec = self._embed_query(query)
        q = query_vec.tolist()
        zvec = _get_zvec()

        coll = self._coll
        if coll is None:
            return []
        where = _build_filter(session_id, after_timestamp, before_timestamp)

        try:
            if where:
                results = coll.query(
                    zvec.Query(field_name="vector", vector=q),
                    filter=where,
                    topk=max(top_k * 2, 10),
                )
            else:
                results = coll.query(
                    zvec.Query(field_name="vector", vector=q),
                    topk=max(top_k * 2, 10),
                )
        except Exception as e:
            logger.warning("Zvec vector search failed: %s", e)
            return []

        scored: List[Dict[str, Any]] = []
        for r in results:
            # Zvec COSINE returns a DISTANCE (0.0 = identical) — convert to
            # similarity so ordering/min_score match the FTS branch.
            similarity = 1.0 - float(getattr(r, "score", 1.0))
            if similarity < min_score:
                continue
            meta = {}
            md = getattr(r, "fields", {}).get("metadata") if hasattr(r, "fields") else None
            if md:
                try:
                    meta = json.loads(md)
                except Exception:
                    pass
            scored.append(self._row_to_dict(r, score=similarity, source="vector", meta=meta))

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _do_fts_search(
        self,
        query: str,
        top_k: int = 5,
        session_id: str = "",
        after_timestamp: float = None,
        before_timestamp: float = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search via Zvec FTS (content field, RocksDB-native)."""
        zvec = _get_zvec()
        coll = self._coll
        if coll is None:
            return []
        where = _build_filter(session_id, after_timestamp, before_timestamp)

        try:
            if where:
                results = coll.query(
                    zvec.Query(
                        field_name="content",
                        fts=zvec.Fts(match_string=query),
                    ),
                    filter=where,
                    topk=max(top_k * 2, 10),
                )
            else:
                results = coll.query(
                    zvec.Query(
                        field_name="content",
                        fts=zvec.Fts(match_string=query),
                    ),
                    topk=max(top_k * 2, 10),
                )
        except Exception as e:
            logger.warning("Zvec FTS search failed: %s", e)
            return []

        scored: List[Dict[str, Any]] = []
        for r in results:
            score = float(getattr(r, "score", 0.0))
            meta = {}
            md = getattr(r, "fields", {}).get("metadata") if hasattr(r, "fields") else None
            if md:
                try:
                    meta = json.loads(md)
                except Exception:
                    pass
            scored.append(self._row_to_dict(r, score=score, source="fts", meta=meta))

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _do_hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        session_id: str = "",
        after_timestamp: float = None,
        before_timestamp: float = None,
        vector_weight: float = None,
        fts_weight: float = None,
    ) -> List[Dict[str, Any]]:
        """Native hybrid query — Zvec MultiQuery + RRFReRanker.

        This replaces LanceDB's two-pass (vector + FTS) + application-layer RRF
        fusion with a single Zvec query that fuses dense vector + FTS + scalar
        filter via RRFReRanker(rank_constant=60).

        Falls back to application-layer RRF if Zvec's native multi-query raises
        (e.g., very old Zvec version).
        """
        if vector_weight is None:
            vector_weight = self.vector_weight
        if fts_weight is None:
            fts_weight = self.fts_weight

        query_vec = self._embed_query(query)
        q = query_vec.tolist()
        zvec = _get_zvec()
        where = _build_filter(session_id, after_timestamp, before_timestamp)

        # Native Zvec MultiQuery path
        try:
            queries = [
                zvec.Query(field_name="vector", vector=q),
                zvec.Query(
                    field_name="content",
                    fts=zvec.Fts(match_string=query),
                ),
            ]
            kwargs = {
                "queries": queries,
                "reranker": zvec.RrfReRanker(rank_constant=60),
                "topk": max(top_k * 2, 10),
            }
            if where:
                kwargs["filter"] = where

            coll = self._coll
            if coll is None:
                return []
            results = coll.query(**kwargs)
        except Exception as native_err:
            logger.warning(
                "Zvec native hybrid failed (%s); falling back to two-pass RRF", native_err
            )
            return self._do_hybrid_search_fallback(
                query, top_k, min_score, session_id,
                after_timestamp, before_timestamp,
                vector_weight, fts_weight,
            )

        scored: List[Dict[str, Any]] = []
        for r in results:
            score = float(getattr(r, "score", 0.0))
            if score < min_score:
                continue
            meta = {}
            md = getattr(r, "fields", {}).get("metadata") if hasattr(r, "fields") else None
            if md:
                try:
                    meta = json.loads(md)
                except Exception:
                    pass
            scored.append(self._row_to_dict(r, score=score, source="hybrid", meta=meta))

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _do_hybrid_search_fallback(
        self,
        query: str,
        top_k: int,
        min_score: float,
        session_id: str,
        after_timestamp: Optional[float],
        before_timestamp: Optional[float],
        vector_weight: float,
        fts_weight: float,
    ) -> List[Dict[str, Any]]:
        """Two-pass RRF — used only when native MultiQuery raises.

        Mirrors memory-lancedb's behavior so the fallback semantics are unchanged.
        """
        vector_results = self._do_search(
            query, top_k=top_k * 2, min_score=0.0,
            session_id=session_id,
            after_timestamp=after_timestamp, before_timestamp=before_timestamp,
        )
        fts_results = self._do_fts_search(
            query, top_k=top_k * 2,
            session_id=session_id,
            after_timestamp=after_timestamp, before_timestamp=before_timestamp,
        )

        k = 60
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for rank, r in enumerate(vector_results):
            item_id = r["id"]
            scores[item_id] = scores.get(item_id, 0.0) + vector_weight / (k + rank + 1)
            items[item_id] = r
            items[item_id]["_source"] = "hybrid"

        for rank, r in enumerate(fts_results):
            item_id = r["id"]
            scores[item_id] = scores.get(item_id, 0.0) + fts_weight / (k + rank + 1)
            if item_id not in items:
                items[item_id] = r
            items[item_id]["_source"] = "hybrid"

        merged = []
        for item_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            item = items[item_id].copy()
            item["score"] = round(rrf_score, 4)
            if item["score"] >= min_score:
                merged.append(item)
            if len(merged) >= top_k:
                break
        return merged

    def _row_to_dict(self, row, *, score: float, source: str, meta: dict) -> Dict[str, Any]:
        """Normalize a zvec result row to the dict shape used by tool handlers."""
        # Zvec returns objects with .id, .score, .vectors, .fields attributes
        fields = getattr(row, "fields", {}) or {}
        return {
            "id":         getattr(row, "id", ""),
            "content":    fields.get("content", ""),
            "role":       fields.get("role", ""),
            "session_id": fields.get("session_id", ""),
            "created_at": float(fields.get("created_at", 0) or 0),
            "score":      round(score, 4),
            "metadata":   meta,
            "_source":    source,
        }

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format top results as a markdown bullet list for system prompt injection."""
        if not results:
            return ""
        lines = []
        for r in results:
            score = r.get("score", 0)
            content = r.get("content", "")[:300]
            lines.append(f"- [{score:.3f}] {content}")
        return "## Ollama Vector Memory\n" + "\n".join(lines)

    # -- Tool handlers -------------------------------------------------------

    def _tool_add(self, args: dict) -> str:
        """vec_memory_add — embed + insert a single memory."""
        if not self._coll:
            return json.dumps({"status": "skipped", "reason": "database not yet initialized"})
        if self._read_only:
            return json.dumps({"status": "skipped", "reason": "collection opened read-only (lock held by another process)"})

        content = args.get("content", "")
        if not content:
            return tool_error("content is required")

        min_len = int(self._config.get("min_content_len", 50))
        if len(content) < min_len:
            return json.dumps({
                "status": "skipped",
                "reason": f"content too short ({len(content)} < {min_len} chars)",
            })

        role = args.get("role", "")
        sid = args.get("session_id", "") or self._session_id
        metadata = args.get("metadata", "{}")

        try:
            with self._embedding_lock:
                vec = _ollama_embed_single(content, self.base_url, self.model)
            mem_id = self._insert(
                combined=content, role=role or "turn", sid=sid,
                ts=datetime.now().timestamp(), vec=vec, meta=metadata,
            )
            return json.dumps({"status": "added", "id": mem_id, "dimension": self.vector_dim})
        except Exception as e:
            return tool_error(f"Failed to add memory: {e}")

    def _tool_search(self, args: dict) -> str:
        """vec_memory_search — three modes: hybrid / vector / keyword."""
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        mode = args.get("mode", "hybrid")
        if mode not in ("hybrid", "vector", "keyword"):
            mode = "hybrid"

        top_k = min(int(args.get("top_k", self.search_top_k)), 20)
        min_score = float(args.get("min_score", 0.0))
        session_id = args.get("session_id", "")

        after_ts = None
        before_ts = None
        if args.get("after_timestamp"):
            try:
                after_ts = float(args["after_timestamp"])
            except (ValueError, TypeError):
                pass
        if args.get("before_timestamp"):
            try:
                before_ts = float(args["before_timestamp"])
            except (ValueError, TypeError):
                pass

        try:
            if mode == "keyword":
                results = self._do_fts_search(
                    query, top_k=top_k,
                    session_id=session_id,
                    after_timestamp=after_ts, before_timestamp=before_ts,
                )
            elif mode == "vector":
                results = self._do_search(
                    query, top_k=top_k, min_score=min_score,
                    session_id=session_id,
                    after_timestamp=after_ts, before_timestamp=before_ts,
                )
            else:  # hybrid
                try:
                    results = self._do_hybrid_search(
                        query, top_k=top_k, min_score=min_score,
                        session_id=session_id,
                        after_timestamp=after_ts, before_timestamp=before_ts,
                    )
                except Exception as hybrid_err:
                    logger.warning("Hybrid search failed, falling back to vector: %s", hybrid_err)
                    results = self._do_search(
                        query, top_k=top_k, min_score=min_score,
                        session_id=session_id,
                        after_timestamp=after_ts, before_timestamp=before_ts,
                    )
                    mode = "vector (fallback)"

            if not results:
                return json.dumps({
                    "query": query, "results": [], "count": 0,
                    "message": "No matching memories found.",
                })

            formatted = []
            for r in results:
                dt = datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M")
                formatted.append({
                    "id":         r["id"],
                    "content":    r["content"],
                    "role":       r["role"],
                    "session_id": r["session_id"],
                    "timestamp":  dt,
                    "score":      r["score"],
                    "metadata":   json.dumps(r.get("metadata") or {}, ensure_ascii=False),
                })

            return json.dumps({
                "query":   query,
                "results": formatted,
                "count":   len(formatted),
                "message": f"Found {len(formatted)} matching memories (mode={mode}).",
                "search_mode": mode,
            })
        except Exception as e:
            return tool_error(f"Search failed: {e}")

    def _tool_list(self, args: dict) -> str:
        """vec_memory_list — recent items, optional session + time-range filter."""
        if not self._coll:
            return json.dumps({"error": "List failed: database not yet initialized"})
        limit = min(int(args.get("limit", 20)), 100)
        session_id = args.get("session_id", "")

        after_ts = None
        before_ts = None
        if args.get("after_timestamp"):
            try:
                after_ts = float(args["after_timestamp"])
            except (ValueError, TypeError):
                pass
        if args.get("before_timestamp"):
            try:
                before_ts = float(args["before_timestamp"])
            except (ValueError, TypeError):
                pass

        try:
            where = _build_filter(session_id, after_ts, before_ts)
            zvec = _get_zvec()
            coll = self._coll
            if coll is None:
                return json.dumps({"error": "List failed: collection not available"})
            # List uses a dummy vector query to get rows (Zvec always needs a query)
            # — limit-only isn't directly exposed; we fetch a large vector-only result
            # and slice. For the "list everything matching filter" case this is fine.
            try:
                results = coll.query(
                    zvec.Query(
                        field_name="vector",
                        vector=[0.0] * self.vector_dim,
                    ),
                    filter=where or "created_at >= 0",   # always-true when no extra filter
                    topk=limit,
                )
            except Exception:
                # Fallback if filter "created_at >= 0" isn't accepted: drop filter
                results = coll.query(
                    zvec.Query(
                        field_name="vector",
                        vector=[0.0] * self.vector_dim,
                    ),
                    topk=limit,
                )

            formatted = []
            for r in results:
                fields = getattr(r, "fields", {}) or {}
                created_at = float(fields.get("created_at", 0) or 0)
                dt = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M") if created_at else "unknown"
                content = fields.get("content", "")
                formatted.append({
                    "id":         getattr(r, "id", ""),
                    "content":    content[:200] + ("..." if len(content) > 200 else ""),
                    "role":       fields.get("role", ""),
                    "session_id": fields.get("session_id", ""),
                    "timestamp":  dt,
                    "metadata":   fields.get("metadata") or "{}",
                })

            return json.dumps({"results": formatted, "count": len(formatted)})
        except Exception as e:
            return tool_error(f"List failed: {e}")

    def _tool_delete(self, args: dict) -> str:
        """vec_memory_delete — delete by IDs."""
        if not self._coll:
            return json.dumps({"deleted": 0, "reason": "database not yet initialized"})
        if self._read_only:
            return json.dumps({"deleted": 0, "reason": "collection opened read-only (lock held by another process)"})
        coll = self._coll
        memory_ids = args.get("memory_ids", [])
        if not memory_ids:
            return tool_error("memory_ids is required")
        try:
            coll.delete(ids=memory_ids)
            return json.dumps({"deleted": len(memory_ids), "requested": len(memory_ids)})
        except Exception as e:
            return tool_error(f"Delete failed: {e}")

    def _tool_stats(self, args: dict) -> str:
        """vec_memory_stats — counts + index info + storage size."""
        if not self._coll:
            return json.dumps({"total_items": 0, "sessions": 0, "storage_size": 0, "message": "database not yet initialized"})
        try:
            total = self._coll.stats.doc_count

            # Distinct sessions by scanning up to 10k rows (Zvec has no native distinct)
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())
            zvec_dir = _expand_hermes_home(
                self._config.get("zvec_dir", f"{hermes_home}/记忆数据库/zvec_memory"),
                hermes_home,
            )
            zvec = _get_zvec()
            coll = self._coll
            try:
                sample = coll.query(
                    zvec.Query(
                        field_name="vector",
                        vector=[0.0] * self.vector_dim,
                    ),
                    topk=min(total, 10000),
                )
                sessions = len({
                    getattr(r, "fields", {}).get("session_id", "")
                    for r in sample
                    if getattr(r, "fields", {}).get("session_id")
                })
            except Exception:
                sessions = 0

            # Storage size
            try:
                size = sum(f.stat().st_size for f in Path(zvec_dir).rglob("*") if f.is_file())
                size_str = f"{size / 1024:.1f} KB"
            except Exception:
                size_str = "unknown"

            return json.dumps({
                "total_memories":     total,
                "total_sessions":     sessions,
                "embedding_model":    self.model,
                "vector_dimension":   self.vector_dim,
                "db_size":            size_str,
                "index":              "HNSW (COSINE) + FTS (RocksDB)",
                "search_modes":       ["hybrid", "vector", "keyword"],
                "fts_index":          True,
                "backend":            "zvec",
                "hybrid_fusion":      "RRFReRanker (native MultiQuery)",
            })
        except Exception as e:
            return tool_error(f"Stats failed: {e}")


# ============================================================================
# Plugin entry point
# ============================================================================

def register(ctx) -> None:
    """Register the Zvec memory provider with Hermes."""
    config = _load_plugin_config()
    provider = ZvecMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
