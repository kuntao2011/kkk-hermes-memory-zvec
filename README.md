# kkk-hermes-memory-zvec

A community **memory backend plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent)**, backed by [Zvec](https://github.com/zvec-ai) vector storage and local [Ollama](https://ollama.com) embeddings.

`memory-zvec` keeps your agent's long-term memory in an in-process vector
database (HNSW + RocksDB FTS) with **native hybrid retrieval** — one Zvec
`MultiQuery` call combining vector similarity, scalar filtering and full-text
search via `RRFReRanker`. Embeddings run locally with Ollama `bge-m3`
(1024-dim), so nothing leaves your machine.

This port's main focus is **lock governance for long-lived sessions**: stock
setups can deadlock or stall when a messaging-platform session and a
dashboard-embedded chat share the same collection. v1.1–v1.3.1 fix that class
of problems end to end (see [CHANGELOG.md](CHANGELOG.md)).

## Features

- **Native hybrid search** — vector + scalar filter + FTS in a single Zvec
  query with RRF re-ranking; scalar fields (`session_id`, `created_at`) use
  index-level pre-filtering, not post-ANN filtering.
- **Session-safe locking** — initialize-time write drain + backoff retry,
  idle lock-release watchdog (default 30s, reacts within ~5s), and a
  collection handle that survives session boundaries.
- **Drop-in compatible with `memory-lancedb`** — same 6 tool schemas
  (`vec_memory_add/search/list/delete/stats`) and the same
  prefetch / `sync_turn` / `on_session_end` hooks. Switching backends is a
  `config.yaml` edit; existing 1024-dim vectors are reused without
  re-embedding.
- **Fast** — ~13,000 docs/s batch insert; sub-ms search under 100k docs.
- **Local-first** — Ollama embeddings, in-process storage, no external
  services beyond the Ollama endpoint you configure.

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) with a working
  plugin directory (`~/.hermes/plugins/`)
- An **embedding model** served over an Ollama-compatible API (`/api/embed`):
  local [Ollama](https://ollama.com) with `bge-m3` (1024-dim, the default), or
  any hosted/online endpoint implementing the same API — just point `base_url`
  at it. The endpoint's vector dimension must match `vector_dim`.
- Python 3.11+ (the Hermes venv); dependencies (`zvec>=0.5.0,<0.7`, `numpy`,
  `requests`) are installed automatically from `plugin.yaml`

## Install

1. Copy the plugin into your Hermes plugin directory **as `memory-zvec`**:

   ```bash
   git clone https://github.com/kuntao2011/kkk-hermes-memory-zvec.git
   cp -r kkk-hermes-memory-zvec ~/.hermes/plugins/memory-zvec
   ```

   The directory name must be `memory-zvec` — memory providers are resolved
   by directory name (bundled first, then `$HERMES_HOME/plugins/`).

2. Point your profile at it in `$HERMES_HOME/config.yaml`:

   ```yaml
   memory:
     provider: memory-zvec

   plugins:
     memory-zvec:
       base_url: http://localhost:11434
       embedding_model: bge-m3:latest
       vector_dim: 1024
       zvec_dir: $HERMES_HOME/memory/zvec_memory   # any path you like
       collection_name: memories
   ```

3. Restart the gateway. Dependencies are installed automatically on the next
   start / `hermes update`.

> **Multi-profile:** Hermes profiles are isolated by `$HERMES_HOME` — each
> profile keeps its own plugin copy, config and data. To use the backend in
> another profile, repeat the install + config there; nothing is shared or
> synced between profiles.

## Configuration

| Key | Default | Description |
|---|---|---|
| `base_url` | `http://localhost:11434` | Ollama server URL |
| `embedding_model` | `bge-m3:latest` | Embedding model available in Ollama |
| `vector_dim` | `1024` | Embedding vector dimension (bge-m3 = 1024) |
| `zvec_dir` | `$HERMES_HOME/记忆数据库/zvec_memory` | Zvec collection directory (profile-scoped; any path works) |
| `collection_name` | `memories` | Zvec collection name |
| `batch_size` | `32` | Max texts per embedding batch |
| `search_top_k` | `5` | Default top-k results per search |
| `min_content_len` | `50` | Skip content shorter than this (chars) |
| `vector_weight` | `0.7` | Hybrid search weight for the vector branch |
| `fts_weight` | `0.3` | Hybrid search weight for the FTS branch |
| `enable_hnsw_optimize` | `true` | Call `collection.optimize()` after large bulk inserts |

Environment knob: `HERMES_MEMORY_ZVEC_IDLE_RELEASE_S` — seconds of idleness
before the watchdog releases a collection lock (default 30).

## Migrating from memory-lancedb

Existing 1024-dim vectors are reused as-is — no re-embedding needed. A full
migration, health-check and repair playbook lives in
[kkk-hermes-zvec-memory-migration](https://github.com/kuntao2011/kkk-hermes-zvec-memory-migration).

## Relation to upstream

This is an independent community plugin, **not** part of the Hermes Agent
tree. Per upstream policy, new memory providers ship as standalone repos that
implement the same `MemoryProvider` ABC and are discovered through the same
path. Thanks to [Nous Research](https://github.com/NousResearch) for the
plugin architecture and to the Zvec team for the storage engine.

## License

[MIT](LICENSE) — the upstream Hermes Agent plugin interface is MIT as well.
