//! Scan operator for reading data from storage.

use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStoreSearch;
use grafeo_common::types::{EpochId, LogicalType, NodeId, TransactionId};
use std::sync::Arc;

/// A scan operator that reads nodes from storage.
pub struct ScanOperator {
    /// The store to scan from.
    store: Arc<dyn GraphStoreSearch>,
    /// Label filter (None = all nodes).
    label: Option<String>,
    /// When set, only these ids are visited. The label filter (if any) and
    /// MVCC visibility filter still apply.
    pinned_ids: Option<Vec<NodeId>>,
    /// Current position in the scan.
    position: usize,
    /// Batch of node IDs to scan.
    batch: Vec<NodeId>,
    /// Whether the scan is exhausted.
    exhausted: bool,
    /// Chunk capacity.
    chunk_capacity: usize,
    /// Transaction ID for MVCC visibility (None = use current epoch).
    transaction_id: Option<TransactionId>,
    /// Epoch for version visibility.
    viewing_epoch: Option<EpochId>,
}

impl ScanOperator {
    /// Creates a new scan operator for all nodes.
    pub fn new(store: Arc<dyn GraphStoreSearch>) -> Self {
        Self {
            store,
            label: None,
            pinned_ids: None,
            position: 0,
            batch: Vec::new(),
            exhausted: false,
            chunk_capacity: 2048,
            transaction_id: None,
            viewing_epoch: None,
        }
    }

    /// Creates a new scan operator for nodes with a specific label.
    pub fn with_label(store: Arc<dyn GraphStoreSearch>, label: impl Into<String>) -> Self {
        Self {
            store,
            label: Some(label.into()),
            pinned_ids: None,
            position: 0,
            batch: Vec::new(),
            exhausted: false,
            chunk_capacity: 2048,
            transaction_id: None,
            viewing_epoch: None,
        }
    }

    /// Creates a scan operator pinned to a fixed id set. Used when the planner
    /// has absorbed an `id(var) = lit` / `id(var) IN [...]` predicate; the
    /// executor short-circuits the label/all-nodes iteration in favour of
    /// O(1) `get_node` lookups per id. Missing ids and label-mismatch ids are
    /// silently skipped (matches the semantics of the original `Filter` they
    /// were rewritten from). MVCC visibility filtering still applies.
    pub fn with_node_ids(
        store: Arc<dyn GraphStoreSearch>,
        ids: Vec<NodeId>,
        label: Option<String>,
    ) -> Self {
        Self {
            store,
            label,
            pinned_ids: Some(ids),
            position: 0,
            batch: Vec::new(),
            exhausted: false,
            chunk_capacity: 2048,
            transaction_id: None,
            viewing_epoch: None,
        }
    }

    /// Sets the chunk capacity.
    pub fn with_chunk_capacity(mut self, capacity: usize) -> Self {
        self.chunk_capacity = capacity;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    ///
    /// When set, the scan will only return nodes visible to this transaction.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    fn load_batch(&mut self) {
        if !self.batch.is_empty() || self.exhausted {
            return;
        }

        // Get nodes. When we have transaction context, use all_node_ids()
        // to include uncommitted/PENDING versions (nodes_by_label already
        // returns unfiltered IDs from the label index, but node_ids()
        // pre-filters by epoch which excludes PENDING nodes).
        let all_ids = if let Some(pinned) = &self.pinned_ids {
            pinned.clone()
        } else {
            match &self.label {
                Some(label) => self.store.nodes_by_label(label),
                None if self.viewing_epoch.is_some() => self.store.all_node_ids(),
                None => self.store.node_ids(),
            }
        };

        // Filter by visibility if we have tx context.
        // Uses batch methods that hold a single lock for all IDs instead of
        // acquiring/releasing per node (avoids N+1 lock pattern).
        let visible = if let Some(epoch) = self.viewing_epoch {
            if let Some(tx) = self.transaction_id {
                self.store
                    .filter_visible_node_ids_versioned(&all_ids, epoch, tx)
            } else {
                self.store.filter_visible_node_ids(&all_ids, epoch)
            }
        } else {
            all_ids
        };

        // For pinned scans we still need to enforce the label (the candidate
        // set didn't come from the label index) and to drop ids that don't
        // resolve to a real node. The non-pinned path uses the label index
        // directly, so this work is skipped.
        self.batch = if self.pinned_ids.is_some() {
            visible
                .into_iter()
                .filter(|id| match self.store.get_node(*id) {
                    Some(node) => match &self.label {
                        Some(label) => node.has_label(label),
                        None => true,
                    },
                    None => false,
                })
                .collect()
        } else {
            visible
        };

        if self.batch.is_empty() {
            self.exhausted = true;
        }
    }
}

impl Operator for ScanOperator {
    fn next(&mut self) -> OperatorResult {
        self.load_batch();

        if self.exhausted || self.position >= self.batch.len() {
            return Ok(None);
        }

        // Create output chunk with node IDs
        let schema = [LogicalType::Node];
        let mut chunk = DataChunk::with_capacity(&schema, self.chunk_capacity);

        let end = (self.position + self.chunk_capacity).min(self.batch.len());
        let count = end - self.position;

        {
            // Column 0 guaranteed to exist: chunk created with single-column schema above
            let col = chunk
                .column_mut(0)
                .expect("column 0 exists: chunk created with single-column schema");
            for i in self.position..end {
                col.push_node_id(self.batch[i]);
            }
        }

        chunk.set_count(count);
        self.position = end;

        Ok(Some(chunk))
    }

    fn reset(&mut self) {
        self.position = 0;
        self.batch.clear();
        self.exhausted = false;
    }

    fn name(&self) -> &'static str {
        "Scan"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::GraphStoreMut;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_scan_by_label() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);

        let mut scan =
            ScanOperator::with_label(store.clone() as Arc<dyn GraphStoreSearch>, "Person");

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2);

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_scan_reset() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        store.create_node(&["Person"]);

        let mut scan =
            ScanOperator::with_label(store.clone() as Arc<dyn GraphStoreSearch>, "Person");

        // First scan
        let chunk1 = scan.next().unwrap().unwrap();
        assert_eq!(chunk1.row_count(), 1);

        // Reset
        scan.reset();

        // Second scan should work
        let chunk2 = scan.next().unwrap().unwrap();
        assert_eq!(chunk2.row_count(), 1);
    }

    #[test]
    fn test_full_scan() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // Create nodes with different labels
        store.create_node(&["Person"]);
        store.create_node(&["Person"]);
        store.create_node(&["Animal"]);
        store.create_node(&["Place"]);

        // Full scan (no label filter) should return all nodes
        let mut scan = ScanOperator::new(store.clone() as Arc<dyn GraphStoreSearch>);

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 4, "Full scan should return all 4 nodes");

        // Should be exhausted
        let next = scan.next().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_scan_with_mvcc_context() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // Create nodes at epoch 1 (using SYSTEM tx so they get real epochs,
        // not PENDING; this test is about epoch-based time-travel scanning).
        let epoch1 = EpochId::new(1);
        store.create_node_versioned(&["Person"], epoch1, TransactionId::SYSTEM);
        store.create_node_versioned(&["Person"], epoch1, TransactionId::SYSTEM);

        // Create a node at epoch 5
        let epoch5 = EpochId::new(5);
        store.create_node_versioned(&["Person"], epoch5, TransactionId::SYSTEM);

        // Scan at epoch 3 should see only the first 2 nodes (created at epoch 1)
        let mut scan =
            ScanOperator::with_label(store.clone() as Arc<dyn GraphStoreSearch>, "Person")
                .with_transaction_context(EpochId::new(3), None);

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 2, "Should see 2 nodes at epoch 3");

        // Scan at epoch 5 should see all 3 nodes
        let mut scan_all =
            ScanOperator::with_label(store.clone() as Arc<dyn GraphStoreSearch>, "Person")
                .with_transaction_context(EpochId::new(5), None);

        let chunk_all = scan_all.next().unwrap().unwrap();
        assert_eq!(chunk_all.row_count(), 3, "Should see 3 nodes at epoch 5");
    }

    #[test]
    fn test_scan_into_any() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let op = ScanOperator::with_label(store.clone() as Arc<dyn GraphStoreSearch>, "Person");
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ScanOperator>().is_ok());
    }

    // ---- with_node_ids: pinned scan ----

    #[test]
    fn test_pinned_scan_single_id_no_label() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let _ = store.create_node(&["Person"]);
        let target = store.create_node(&["Person"]);
        let _ = store.create_node(&["Animal"]);

        let mut scan = ScanOperator::with_node_ids(
            store.clone() as Arc<dyn GraphStoreSearch>,
            vec![target],
            None,
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        let col = chunk.column(0).unwrap();
        assert_eq!(col.get_node_id(0), Some(target));
        assert!(scan.next().unwrap().is_none());
    }

    #[test]
    fn test_pinned_scan_label_filter_drops_mismatch() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let person = store.create_node(&["Person"]);
        let animal = store.create_node(&["Animal"]);

        let mut scan = ScanOperator::with_node_ids(
            store.clone() as Arc<dyn GraphStoreSearch>,
            vec![person, animal],
            Some("Person".to_string()),
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1, "Animal id should be filtered out");
        assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(person));
    }

    #[test]
    fn test_pinned_scan_missing_id_is_skipped() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let real = store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_node_ids(
            store.clone() as Arc<dyn GraphStoreSearch>,
            vec![real, NodeId(9999), NodeId(10_000)],
            None,
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);
        assert_eq!(chunk.column(0).unwrap().get_node_id(0), Some(real));
    }

    #[test]
    fn test_pinned_scan_multiple_ids_preserves_order() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let a = store.create_node(&["Person"]);
        let b = store.create_node(&["Person"]);
        let c = store.create_node(&["Person"]);

        let mut scan = ScanOperator::with_node_ids(
            store.clone() as Arc<dyn GraphStoreSearch>,
            vec![c, a, b],
            Some("Person".to_string()),
        );

        let chunk = scan.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 3);
        let col = chunk.column(0).unwrap();
        assert_eq!(col.get_node_id(0), Some(c));
        assert_eq!(col.get_node_id(1), Some(a));
        assert_eq!(col.get_node_id(2), Some(b));
    }

    #[test]
    fn test_pinned_scan_empty_after_label_filter() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let animal = store.create_node(&["Animal"]);

        let mut scan = ScanOperator::with_node_ids(
            store.clone() as Arc<dyn GraphStoreSearch>,
            vec![animal],
            Some("Person".to_string()),
        );

        assert!(scan.next().unwrap().is_none());
    }
}
