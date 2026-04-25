//! Section-based container format for `.grafeo` files.
//!
//! Extends the single-file format with a section directory, enabling
//! independent read/write of typed sections. The container treats
//! section data as opaque `&[u8]` bytes.
//!
//! ## File Layout (v2)
//!
//! | Offset | Size | Contents |
//! |--------|------|----------|
//! | 0x0000 | 4 KiB | FileHeader (magic, format version) |
//! | 0x1000 | 4 KiB | DbHeader H1 (iteration, checksum) |
//! | 0x2000 | 4 KiB | DbHeader H2 (alternating copy) |
//! | 0x3000 | 4 KiB | Section Directory |
//! | 0x4000+ | variable | Section data (page-aligned) |

pub mod directory;

#[cfg(feature = "wal")]
pub mod mmap;

#[cfg(feature = "wal")]
pub mod page_fetcher;

pub use directory::SectionDirectory;

#[cfg(feature = "wal")]
pub use mmap::MmapSection;

#[cfg(feature = "wal")]
pub use page_fetcher::{AccessHint, MmapPageFetcher, PageFetcher};
