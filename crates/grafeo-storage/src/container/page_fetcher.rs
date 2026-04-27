//! Mmap-backed implementation of the [`PageFetcher`] trait.
//!
//! The trait itself lives in `grafeo-common` so that `Section`
//! implementations across the workspace can accept paged byte access
//! without depending on any specific I/O mechanism. This module provides
//! the v1 implementation that wraps an [`MmapSection`].

use std::io;
use std::sync::Arc;

pub use grafeo_common::storage::{AccessHint, PageFetcher};

use super::mmap::MmapSection;

/// `PageFetcher` implementation backed by a memory-mapped section.
///
/// Wraps an `Arc<MmapSection>` so multiple consumers can share the same
/// mapping. The mapping stays alive as long as any clone of the `Arc`
/// survives.
pub struct MmapPageFetcher {
    section: Arc<MmapSection>,
}

impl MmapPageFetcher {
    /// Create a new fetcher around an mmap'd section.
    #[must_use]
    pub fn new(section: Arc<MmapSection>) -> Self {
        Self { section }
    }

    /// The underlying section.
    #[must_use]
    pub fn section(&self) -> &Arc<MmapSection> {
        &self.section
    }
}

impl PageFetcher for MmapPageFetcher {
    fn fetch(&self, offset: usize, len: usize) -> io::Result<&[u8]> {
        let bytes = self.section.as_bytes();
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "offset + len overflow"))?;
        bytes.get(offset..end).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "requested {len} bytes at offset {offset}, section is {} bytes",
                    bytes.len()
                ),
            )
        })
    }

    fn len(&self) -> usize {
        self.section.len()
    }

    fn advise(&self, offset: usize, len: usize, hint: AccessHint) {
        self.section.advise(offset, len, hint);
    }
}

impl std::fmt::Debug for MmapPageFetcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapPageFetcher")
            .field("section", self.section.as_ref())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::mmap::MmapSection;
    use grafeo_common::storage::SectionType;
    use std::io::Write;
    use std::sync::Arc;

    /// Test helper: create an `MmapSection` over a temp file containing `data`.
    ///
    /// Uses `#[allow(unsafe_code)]` to call the inherently-unsafe
    /// `memmap2::MmapOptions::map`. The file is held alive by the mapping.
    #[allow(unsafe_code)]
    fn make_mmap_section(data: &[u8]) -> MmapSection {
        let mut f = tempfile::NamedTempFile::new().expect("create temp file");
        f.write_all(data).expect("write data");
        f.flush().expect("flush");
        // SAFETY: file is freshly written and not concurrently modified
        // for the lifetime of the mapping (we hold the temp file open via
        // the mmap). Safe per memmap2 contract for read-only mappings of
        // private temp files.
        let mmap = unsafe { memmap2::MmapOptions::new().map(f.as_file()).expect("mmap") };
        MmapSection::new(mmap, SectionType::PropertyIndex, 0)
    }

    #[test]
    fn test_mmap_page_fetcher_roundtrip() {
        // Build deterministic 4 KiB payload: byte i = (i % 251).
        // try_from never fails because (i % 251) < u8::MAX.
        let payload: Vec<u8> = (0u32..4096)
            .map(|i| u8::try_from(i % 251).expect("i % 251 fits in u8"))
            .collect();
        let section = Arc::new(make_mmap_section(&payload));
        let fetcher = MmapPageFetcher::new(section);

        // Verify total length.
        assert_eq!(fetcher.len(), 4096);
        assert!(!fetcher.is_empty());

        // Fetch from start.
        let head = fetcher.fetch(0, 16).expect("fetch head");
        assert_eq!(head, &payload[0..16]);

        // Fetch from middle.
        let mid = fetcher.fetch(100, 32).expect("fetch mid");
        assert_eq!(mid, &payload[100..132]);

        // Fetch entire region.
        let all = fetcher.fetch(0, 4096).expect("fetch all");
        assert_eq!(all, payload.as_slice());

        // Fetch beyond bounds → UnexpectedEof.
        let err = fetcher.fetch(4096, 1).expect_err("expected EOF");
        assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);

        // Fetch with offset+len overflow → InvalidInput.
        let err = fetcher.fetch(usize::MAX, 1).expect_err("expected overflow");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_advise_no_panic() {
        let payload = vec![0u8; 4096];
        let section = Arc::new(make_mmap_section(&payload));
        let fetcher = MmapPageFetcher::new(section);

        // Each hint variant must complete without panic on every platform.
        for hint in [
            AccessHint::Sequential,
            AccessHint::Random,
            AccessHint::WillNeed,
            AccessHint::DontNeed,
        ] {
            fetcher.advise(0, 4096, hint);
        }

        // Advise on zero-length range must also not panic.
        fetcher.advise(0, 0, AccessHint::WillNeed);

        // Advise on an out-of-range region must not panic (best-effort).
        fetcher.advise(8192, 4096, AccessHint::Sequential);
    }

    #[test]
    fn test_is_empty_default_method() {
        struct ZeroFetcher;
        impl PageFetcher for ZeroFetcher {
            fn fetch(&self, _offset: usize, _len: usize) -> std::io::Result<&[u8]> {
                Ok(&[])
            }
            fn len(&self) -> usize {
                0
            }
            fn advise(&self, _offset: usize, _len: usize, _hint: AccessHint) {}
        }
        assert!(ZeroFetcher.is_empty());
    }
}
