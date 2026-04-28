---
title: Storage Tiers
description: How Grafeo manages the RAM ↔ disk lifecycle of section data.
tags:
  - architecture
  - memory
  - storage
---

# Storage Tiers

Grafeo's section data (LPG store, RDF store, vector indexes, ring index, etc.) lives in one of three tiers at any moment. The buffer manager moves data between tiers in response to memory pressure; configuration overrides let you pin specific sections.

## The three tiers

| Tier | Where the data lives | When |
| ---- | -------------------- | ---- |
| `InMemory` | Heap (Rust `Arc`s, `Vec`s, `HashMap`s) | Default for fresh inserts |
| `OnDisk` | Memory-mapped file in the spill directory | After a spill |
| `Uninitialized` | Section registered but holds no data yet | Right after database open with no inserts |

Reads work the same in either tier. The on-disk path serves through the OS page cache; cold pages fault in lazily.

## Default behavior (`Auto`)

By default every section is `TierOverride::Auto`. The buffer manager tracks a unified budget (default 75% of system RAM) and reacts to four pressure levels:

| Level | Allocated | Action |
| ----- | --------- | ------ |
| Normal | < 70% | No action |
| Moderate | 70-85% | Proactive eviction of cold data |
| High | 85-95% | Aggressive eviction, trigger spilling |
| Critical | > 95% | Block new allocations |

When pressure crosses High, the buffer manager picks consumers in priority order (lowest-priority first, e.g., query caches before graph storage) and asks each to spill or evict until the budget recovers.

## Explicit overrides

Use `Config::with_section_tier` to pin a specific section to a tier:

```rust
use grafeo_engine::{Config, GrafeoDB};
use grafeo_common::storage::{SectionType, TierOverride};

// Force the LPG compact base to mmap mode at database open.
let config = Config::persistent("/path/to/db.grafeo")
    .with_section_tier(SectionType::CompactStore, TierOverride::ForceDisk)
    .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);

let db = GrafeoDB::with_config(config)?;
```

| Override | Behavior |
| -------- | -------- |
| `Auto` | Default. Buffer manager decides based on pressure. |
| `ForceDisk` | At database open, the matching section is spilled immediately. Subsequent reads serve from mmap. |
| `ForceRam` | Declarative: this section must stay in RAM. **Not yet enforced at runtime** (will be in 0.5.43); today behaves identically to `Auto`. |

`ForceDisk` is targeted: only the matching section is spilled, other sections are unaffected. Configure each section type that should start on disk; the rest follow the `Auto` policy.

You can also pair the tier with a hard `max_ram` cap:

```rust
use grafeo_common::storage::SectionMemoryConfig;

let config = Config::in_memory().with_section_config(
    SectionType::VectorStore,
    SectionMemoryConfig {
        max_ram: Some(500 * 1024 * 1024), // 500 MB cap on vector index heap
        tier: TierOverride::Auto,
    },
);
```

## Reading the current tier

`db.storage_tiers()` returns the tier of every registered section consumer:

```rust
use grafeo_common::storage::SectionType;
use grafeo_common::memory::buffer::StorageTier;

let tiers = db.storage_tiers();
match tiers.get(&SectionType::VectorStore) {
    Some(StorageTier::OnDisk) => println!("vector index is on disk"),
    Some(StorageTier::InMemory) => println!("vector index is in RAM"),
    Some(StorageTier::Uninitialized) | None => println!("no vector index registered"),
}
```

This is observability only: it doesn't move data. Useful for tests, dashboards, and confirming a `ForceDisk` config took effect.

## Bringing data back: `reload_eligible`

After memory pressure drops (a long-running workload finishes, a checkpoint freed mutation overlay state) you can ask the buffer manager to bring spilled sections back into RAM:

```rust
// Reload as long as projected usage stays below 70% of the budget.
// Walks consumers highest-priority-first; stops when budget would exceed target.
let count = db.reload_eligible(0.7);
println!("reloaded {count} sections");
```

`reload_eligible` is best-effort: a per-consumer reload that fails (e.g., spill file missing) is logged-and-skipped. The walk visits highest-priority consumers first (graph storage before index buffers before query caches) so the most-important data comes back first.

The reload is synchronous; for large sections, call from a background thread.

## Spill directory

The spill directory holds mmap-backed files for spilled sections. It's set via `Config::with_spill_path` (or auto-derived from the `.grafeo` file path for persistent databases). After the database closes, the spill files persist; reopening the database re-mmaps them so spilled state survives restarts.

```rust
let config = Config::persistent("/var/lib/grafeo/db.grafeo")
    .with_spill_path("/var/lib/grafeo/spill");
```

## Tracing

When the `tracing` feature is enabled, tier transitions emit events under the `grafeo::buffer` and `grafeo::tier` targets:

| Event target / level | When |
| -------------------- | ---- |
| `grafeo::buffer` info | A consumer spills via `spill_consumer_by_name` |
| `grafeo::buffer` info | A consumer reloads in `reload_eligible` |
| `grafeo::buffer` warn | A consumer's reload returned an error |
| `grafeo::tier` info | A `ForceDisk` override fires at database open |

Wire your tracing subscriber (e.g., `tracing-subscriber`) to these targets to log or export tier transitions.
