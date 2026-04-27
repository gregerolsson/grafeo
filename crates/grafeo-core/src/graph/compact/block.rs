//! Block descriptors for columnar storage.
//!
//! A "block" is a logical chunk of a column's rows that can be skipped or
//! processed as a unit. Phase 2a introduces the descriptor type and the
//! enumeration API on `ColumnCodec`(super::column::ColumnCodec); every
//! column today reports a single block covering all rows. Phase 2b will
//! introduce multi-block serialization, and Phase 2c will add per-block
//! statistics (min/max/null/row counts, optional bloom) so range scans
//! can skip blocks whose stats prove no match.
//!
//! The split is deliberate: Phase 2a keeps the data layout untouched so
//! upstream readers can be extended block-aware before any on-disk
//! format changes.

/// Descriptor for a logical block within a `ColumnCodec`.
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
