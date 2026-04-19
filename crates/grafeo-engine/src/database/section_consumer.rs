//! Adapts storage sections into [`MemoryConsumer`]s for BufferManager integration.
//!
//! Each section (LPG, RDF, Vector, Text, Catalog) is registered with the
//! [`BufferManager`] so that memory tracking and pressure awareness include
//! section memory. This enables accurate `memory_usage()` reporting and
//! lays the groundwork for automatic spilling when tiered storage is added.

use std::sync::Arc;
#[cfg(any(
    all(
        feature = "lpg",
        feature = "vector-index",
        feature = "mmap",
        not(feature = "temporal")
    ),
    all(feature = "lpg", feature = "text-index")
))]
use std::sync::Weak;

use grafeo_common::memory::buffer::{MemoryConsumer, MemoryRegion, SpillError, priorities};
use grafeo_common::storage::Section;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use grafeo_common::types::{PropertyKey, Value};
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use grafeo_core::index::vector::VectorStorage;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use parking_lot::RwLock;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use std::collections::HashMap;
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
use std::path::PathBuf;

/// Wraps a [`Section`] as a [`MemoryConsumer`] for the BufferManager.
///
/// Data sections (Catalog, LPG, RDF) use [`GRAPH_STORAGE`](priorities::GRAPH_STORAGE)
/// priority (evict last). Index sections (Vector, Text, RdfRing, PropertyIndex)
/// use [`INDEX_BUFFERS`](priorities::INDEX_BUFFERS) priority (evict before data).
///
/// Currently, `evict()` returns 0 because sections cannot release memory
/// without a full checkpoint + mmap cycle. The [`can_spill`](MemoryConsumer::can_spill)
/// method returns `true` for mmap-able index sections, signaling that future
/// tiered storage support will enable actual spilling.
pub struct SectionConsumer {
    name: String,
    section: Arc<dyn Section>,
    priority: u8,
    region: MemoryRegion,
    mmap_able: bool,
}

impl SectionConsumer {
    /// Creates a consumer for the given section.
    ///
    /// Priority and region are assigned based on the section type:
    /// - Data sections (types 1-9): `GRAPH_STORAGE` priority, `GraphStorage` region
    /// - Index sections (types 10+): `INDEX_BUFFERS` priority, `IndexBuffers` region
    pub fn new(section: Arc<dyn Section>) -> Self {
        let section_type = section.section_type();
        let is_data = section_type.is_data_section();
        let flags = section_type.default_flags();

        Self {
            name: format!("section:{section_type:?}"),
            section,
            priority: if is_data {
                priorities::GRAPH_STORAGE
            } else {
                priorities::INDEX_BUFFERS
            },
            region: if is_data {
                MemoryRegion::GraphStorage
            } else {
                MemoryRegion::IndexBuffers
            },
            mmap_able: flags.mmap_able,
        }
    }
}

impl MemoryConsumer for SectionConsumer {
    fn name(&self) -> &str {
        &self.name
    }

    fn memory_usage(&self) -> usize {
        self.section.memory_usage()
    }

    fn eviction_priority(&self) -> u8 {
        self.priority
    }

    fn region(&self) -> MemoryRegion {
        self.region
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        // Sections cannot evict in-place. Freeing section memory requires
        // a checkpoint (serialize + write to container) followed by mmap.
        // The engine handles this at a higher level when pressure is detected.
        0
    }

    fn can_spill(&self) -> bool {
        // Index sections with mmap support can be spilled to the container
        // and served via memory-mapped I/O. Data sections require full
        // deserialization and cannot be mmap'd (yet).
        self.mmap_able
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        if !self.mmap_able {
            return Err(SpillError::NotSupported);
        }
        // Actual spill implementation will be added with tiered storage:
        // 1. Serialize section via Section::serialize()
        // 2. Write to container via GrafeoFileManager::write_sections()
        // 3. Mmap the section via GrafeoFileManager::mmap_section()
        // 4. Switch section to mmap-backed read mode
        // 5. Drop in-memory data, return freed bytes
        Err(SpillError::NotSupported)
    }
}

/// Dynamic memory consumer for vector indexes.
///
/// Holds a `Weak<LpgStore>` and re-queries the live index map on each
/// `memory_usage()` call. On `spill()`, vector embedding property columns
/// are drained to `MmapStorage` files, freeing heap memory. Search uses
/// [`SpillableVectorAccessor`](grafeo_core::index::vector::SpillableVectorAccessor)
/// which checks the spill storage first, then falls back to property storage.
#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
pub struct VectorIndexConsumer {
    store: Weak<grafeo_core::graph::lpg::LpgStore>,
    /// Directory for spill files. `None` disables spilling.
    spill_path: Option<PathBuf>,
    /// Map of "label:property" -> MmapStorage for spilled indexes.
    /// Shared with the search path so `SpillableVectorAccessor` can read.
    pub(crate) spilled: Arc<RwLock<HashMap<String, Arc<grafeo_core::index::vector::MmapStorage>>>>,
}

#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
impl VectorIndexConsumer {
    /// Creates a consumer that dynamically queries the store for current vector indexes.
    pub fn new(
        store: &Arc<grafeo_core::graph::lpg::LpgStore>,
        spill_path: Option<PathBuf>,
    ) -> Self {
        Self {
            store: Arc::downgrade(store),
            spill_path,
            spilled: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the shared spill registry for the search path.
    #[must_use]
    pub fn spilled_storages(
        &self,
    ) -> &Arc<RwLock<HashMap<String, Arc<grafeo_core::index::vector::MmapStorage>>>> {
        &self.spilled
    }

    /// Spills a single vector index's embeddings to disk.
    ///
    /// Returns bytes freed, or an error.
    fn spill_index(
        &self,
        store: &grafeo_core::graph::lpg::LpgStore,
        key: &str,
        dimensions: usize,
    ) -> Result<usize, SpillError> {
        let spill_dir = self
            .spill_path
            .as_ref()
            .ok_or(SpillError::NoSpillDirectory)?;

        // Extract property name from key ("label:property" -> "property")
        let property = key
            .split(':')
            .nth(1)
            .ok_or_else(|| SpillError::IoError(format!("invalid index key: {key}")))?;
        let prop_key = PropertyKey::new(property);

        // Drain vector values from the property column
        let drained = store.drain_node_property_column(&prop_key);
        if drained.is_empty() {
            return Ok(0);
        }

        // Create spill directory if needed
        std::fs::create_dir_all(spill_dir).map_err(|e| SpillError::IoError(e.to_string()))?;

        // Sanitize key for filename ("Label:property" -> "Label%3Aproperty")
        // Percent-encodes ':' to preserve label case, underscores, and avoid
        // ambiguity with any separator character.
        let safe_key = key.replace('%', "%25").replace(':', "%3A");
        let spill_file = spill_dir.join(format!("vectors_{safe_key}.bin"));

        // Create MmapStorage and write all vectors
        let mmap_storage = grafeo_core::index::vector::MmapStorage::create(&spill_file, dimensions)
            .map_err(|e| SpillError::IoError(e.to_string()))?;

        let mut freed_bytes = 0;
        for (id, value) in &drained {
            if let Value::Vector(vec_data) = value {
                freed_bytes += vec_data.len() * 4 + std::mem::size_of::<Arc<[f32]>>();
                mmap_storage
                    .insert(*id, vec_data)
                    .map_err(|e| SpillError::IoError(e.to_string()))?;
            }
        }

        mmap_storage
            .flush()
            .map_err(|e| SpillError::IoError(e.to_string()))?;

        // Register the spill storage
        self.spilled
            .write()
            .insert(key.to_string(), Arc::new(mmap_storage));

        Ok(freed_bytes)
    }
}

#[cfg(all(
    feature = "lpg",
    feature = "vector-index",
    feature = "mmap",
    not(feature = "temporal")
))]
impl MemoryConsumer for VectorIndexConsumer {
    fn name(&self) -> &str {
        "section:VectorStore"
    }

    fn memory_usage(&self) -> usize {
        self.store.upgrade().map_or(0, |store| {
            store
                .vector_index_entries()
                .iter()
                .map(|(_, idx)| idx.heap_memory_bytes())
                .sum()
        })
    }

    fn eviction_priority(&self) -> u8 {
        priorities::INDEX_BUFFERS
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::IndexBuffers
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        0
    }

    fn can_spill(&self) -> bool {
        self.spill_path.is_some()
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        let store = self
            .store
            .upgrade()
            .ok_or(SpillError::IoError("store dropped".to_string()))?;

        let indexes = store.vector_index_entries();
        let mut total_freed = 0;

        for (key, index) in &indexes {
            // Skip already-spilled indexes
            if self.spilled.read().contains_key(key) {
                continue;
            }

            let dimensions = index.config().dimensions;
            match self.spill_index(&store, key, dimensions) {
                Ok(freed) => total_freed += freed,
                Err(e) => {
                    // Log but continue: earlier indexes may have already been
                    // drained and persisted. Returning Err would discard the
                    // freed bytes from those, leaving BufferManager with
                    // incorrect pressure tracking.
                    eprintln!("failed to spill vector index {key}: {e}");
                }
            }
        }

        Ok(total_freed)
    }

    fn reload(&self) -> Result<(), SpillError> {
        let store = self
            .store
            .upgrade()
            .ok_or(SpillError::IoError("store dropped".to_string()))?;

        let mut spilled = self.spilled.write();
        for (key, mmap_storage) in spilled.drain() {
            let property = key
                .split(':')
                .nth(1)
                .ok_or_else(|| SpillError::IoError(format!("invalid index key: {key}")))?;
            let prop_key = PropertyKey::new(property);

            // Export vectors from mmap, restore to property store
            let vectors = mmap_storage.export_all();
            store.restore_node_property_column(
                &prop_key,
                vectors
                    .into_iter()
                    .map(|(id, vec_data)| (id, Value::Vector(vec_data))),
            );

            // Delete spill file
            if let Ok(path) = std::fs::canonicalize(mmap_storage.path()) {
                let _ = std::fs::remove_file(path);
            }
        }

        Ok(())
    }
}

/// Dynamic memory consumer for text indexes.
///
/// Same rationale as [`VectorIndexConsumer`]: avoids holding stale `Arc` refs
/// to indexes that may have been dropped, and automatically picks up new ones.
#[cfg(all(feature = "lpg", feature = "text-index"))]
pub struct TextIndexConsumer {
    store: Weak<grafeo_core::graph::lpg::LpgStore>,
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl TextIndexConsumer {
    /// Creates a consumer that dynamically queries the store for current text indexes.
    pub fn new(store: &Arc<grafeo_core::graph::lpg::LpgStore>) -> Self {
        Self {
            store: Arc::downgrade(store),
        }
    }
}

#[cfg(all(feature = "lpg", feature = "text-index"))]
impl MemoryConsumer for TextIndexConsumer {
    fn name(&self) -> &str {
        "section:TextIndex"
    }

    fn memory_usage(&self) -> usize {
        self.store.upgrade().map_or(0, |store| {
            store
                .text_index_entries()
                .iter()
                .map(|(_, idx)| idx.read().heap_memory_bytes())
                .sum()
        })
    }

    fn eviction_priority(&self) -> u8 {
        priorities::INDEX_BUFFERS
    }

    fn region(&self) -> MemoryRegion {
        MemoryRegion::IndexBuffers
    }

    fn evict(&self, _target_bytes: usize) -> usize {
        0
    }

    fn can_spill(&self) -> bool {
        true
    }

    fn spill(&self, _target_bytes: usize) -> Result<usize, SpillError> {
        Err(SpillError::NotSupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::storage::section::SectionType;
    use grafeo_common::utils::error::Result;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Minimal Section implementation for testing.
    struct FakeSection {
        section_type: SectionType,
        usage: usize,
        dirty: AtomicBool,
    }

    impl FakeSection {
        fn new(section_type: SectionType, usage: usize) -> Self {
            Self {
                section_type,
                usage,
                dirty: AtomicBool::new(false),
            }
        }
    }

    impl Section for FakeSection {
        fn section_type(&self) -> SectionType {
            self.section_type
        }
        fn serialize(&self) -> Result<Vec<u8>> {
            Ok(vec![0; self.usage])
        }
        fn deserialize(&mut self, _data: &[u8]) -> Result<()> {
            Ok(())
        }
        fn is_dirty(&self) -> bool {
            self.dirty.load(Ordering::Relaxed)
        }
        fn mark_clean(&self) {
            self.dirty.store(false, Ordering::Relaxed);
        }
        fn memory_usage(&self) -> usize {
            self.usage
        }
    }

    #[test]
    fn data_section_consumer_properties() {
        let section = Arc::new(FakeSection::new(SectionType::LpgStore, 1024));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:LpgStore");
        assert_eq!(consumer.memory_usage(), 1024);
        assert_eq!(consumer.eviction_priority(), priorities::GRAPH_STORAGE);
        assert_eq!(consumer.region(), MemoryRegion::GraphStorage);
        assert!(!consumer.can_spill());
    }

    #[test]
    fn index_section_consumer_properties() {
        let section = Arc::new(FakeSection::new(SectionType::VectorStore, 4096));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:VectorStore");
        assert_eq!(consumer.memory_usage(), 4096);
        assert_eq!(consumer.eviction_priority(), priorities::INDEX_BUFFERS);
        assert_eq!(consumer.region(), MemoryRegion::IndexBuffers);
        assert!(consumer.can_spill());
    }

    #[test]
    fn evict_returns_zero() {
        let section = Arc::new(FakeSection::new(SectionType::TextIndex, 8192));
        let consumer = SectionConsumer::new(section);

        // Sections can't evict in-place
        assert_eq!(consumer.evict(4096), 0);
        // Memory is unchanged
        assert_eq!(consumer.memory_usage(), 8192);
    }

    #[test]
    fn spill_returns_not_supported() {
        let section = Arc::new(FakeSection::new(SectionType::VectorStore, 4096));
        let consumer = SectionConsumer::new(section);

        let result = consumer.spill(2048);
        assert!(result.is_err());
    }

    #[test]
    fn catalog_section_is_data() {
        let section = Arc::new(FakeSection::new(SectionType::Catalog, 256));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.eviction_priority(), priorities::GRAPH_STORAGE);
        assert!(!consumer.can_spill());
    }

    #[test]
    fn rdf_ring_section_is_index() {
        let section = Arc::new(FakeSection::new(SectionType::RdfRing, 2048));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.eviction_priority(), priorities::INDEX_BUFFERS);
        assert!(consumer.can_spill());
    }

    #[test]
    fn property_index_section_is_index() {
        let section = Arc::new(FakeSection::new(SectionType::PropertyIndex, 512));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:PropertyIndex");
        assert_eq!(consumer.eviction_priority(), priorities::INDEX_BUFFERS);
        assert_eq!(consumer.region(), MemoryRegion::IndexBuffers);
        assert!(consumer.can_spill());
    }

    #[test]
    fn rdf_store_section_is_data() {
        let section = Arc::new(FakeSection::new(SectionType::RdfStore, 1024));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.name(), "section:RdfStore");
        assert_eq!(consumer.eviction_priority(), priorities::GRAPH_STORAGE);
        assert_eq!(consumer.region(), MemoryRegion::GraphStorage);
        assert!(!consumer.can_spill(), "data sections cannot spill");
    }

    #[test]
    fn spill_non_mmap_section_returns_not_supported() {
        // LpgStore is a data section (mmap_able=false), spill should fail
        let section = Arc::new(FakeSection::new(SectionType::LpgStore, 4096));
        let consumer = SectionConsumer::new(section);

        assert!(!consumer.can_spill());
        let result = consumer.spill(2048);
        match result {
            Err(SpillError::NotSupported) => {}
            other => panic!("expected NotSupported, got {other:?}"),
        }
    }

    #[test]
    fn zero_memory_section() {
        let section = Arc::new(FakeSection::new(SectionType::Catalog, 0));
        let consumer = SectionConsumer::new(section);

        assert_eq!(consumer.memory_usage(), 0);
        assert_eq!(consumer.evict(1024), 0);
    }

    #[test]
    fn section_consumer_name_format() {
        // Verify all section types produce "section:<Type>" names
        for section_type in [
            SectionType::Catalog,
            SectionType::LpgStore,
            SectionType::RdfStore,
            SectionType::VectorStore,
            SectionType::TextIndex,
            SectionType::RdfRing,
            SectionType::PropertyIndex,
        ] {
            let section = Arc::new(FakeSection::new(section_type, 100));
            let consumer = SectionConsumer::new(section);
            assert!(
                consumer.name().starts_with("section:"),
                "name should start with 'section:' for {section_type:?}"
            );
        }
    }
}
