//! Block descriptors for columnar storage.
//!
//! A "block" is a logical chunk of a column's rows that can be skipped or
//! processed as a unit. Blocks bridge raw codec output (bit-packed
//! integers, dictionary-encoded strings, boolean bitmaps, etc.) and
//! higher-level scan operators that want to prune work using per-block
//! summary statistics.
//!
//! This module lives in `codec` rather than under any particular store
//! because both [`graph::compact::ColumnCodec`](crate::graph::compact::column::ColumnCodec)
//! (the read-only columnar base) and the LPG store's `PropertyColumn`
//! (the mutable in-memory store, modernized later in Phase 2) describe
//! their data using the same block descriptors.
//!
//! # Phasing
//!
//! Phase 2a introduces the descriptor type and the enumeration API on
//! `ColumnCodec`; every column today reports a single block covering
//! all rows. Phase 2b will introduce multi-block serialization, and
//! Phase 2c will add per-block statistics (min/max/null/row counts,
//! optional bloom) so range scans can skip blocks whose stats prove
//! no match. Phase 2d/2e extend the same descriptors to LpgStore.
//!
//! The split is deliberate: Phase 2a keeps the data layout untouched so
//! upstream readers can be extended block-aware before any on-disk
//! format changes.

/// Descriptor for a logical block within a column.
///
/// In Phase 2a this only carries the block's row count. Phase 2b adds a
/// `byte_offset` and `byte_len` for serialized layouts; Phase 2c adds
/// optional `min`, `max`, `null_count`, and `bloom` fields for per-block
/// pruning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockEntry {
    /// Number of logical rows (values) in this block.
    pub row_count: u32,
}

impl BlockEntry {
    /// Constructs a block entry with the given row count.
    #[must_use]
    pub const fn new(row_count: u32) -> Self {
        Self { row_count }
    }
}
