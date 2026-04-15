//! Two-layer graph store: read-only columnar base + mutable LPG overlay.
//!
//! [`LayeredStore`] coordinates reads between a [`CompactStore`] (cold, columnar)
//! and an [`LpgStore`] (hot, HashMap-based). All writes go to the overlay.
//! Reads check the overlay first and fall through to the compact base for
//! unmodified entities.
//!
//! Requires both `compact-store` and `lpg` features.

use std::sync::Arc;

use arcstr::ArcStr;
use grafeo_common::types::{EdgeId, EpochId, NodeId, PropertyKey, TransactionId, Value};
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use parking_lot::RwLock;

use super::CompactStore;
use crate::graph::Direction;
use crate::graph::lpg::{CompareOp, Edge, LpgStore, Node};
use crate::graph::traits::{GraphStore, GraphStoreMut};
use crate::statistics::Statistics;

/// A two-layer graph store with a columnar base and mutable overlay.
///
/// The compact base serves cold reads (immutable, columnar). The LPG overlay
/// captures all mutations. Reads check the overlay first: if an entity is in
/// `dirty_node_ids` or `dirty_edge_ids`, the overlay is authoritative. If an
/// entity is in `deleted_from_base_nodes` or `deleted_from_base_edges`, it has
/// been deleted and returns `None`. Otherwise, the base is queried.
pub struct LayeredStore {
    /// Read-only columnar base (cold data).
    base: Arc<CompactStore>,
    /// Mutable overlay for new and modified data.
    overlay: Arc<LpgStore>,
    /// Node IDs modified or created in the overlay.
    dirty_node_ids: RwLock<FxHashSet<NodeId>>,
    /// Edge IDs modified or created in the overlay.
    dirty_edge_ids: RwLock<FxHashSet<EdgeId>>,
    /// Base node IDs that have been deleted.
    deleted_from_base_nodes: RwLock<FxHashSet<NodeId>>,
    /// Base edge IDs that have been deleted.
    deleted_from_base_edges: RwLock<FxHashSet<EdgeId>>,
}

impl std::fmt::Debug for LayeredStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayeredStore")
            .field("base_node_count", &self.base.node_count())
            .field("overlay_node_count", &self.overlay.node_count())
            .field("dirty_nodes", &self.dirty_node_ids.read().len())
            .field(
                "deleted_base_nodes",
                &self.deleted_from_base_nodes.read().len(),
            )
            .finish_non_exhaustive()
    }
}

impl LayeredStore {
    /// Creates a layered store from a compact base.
    ///
    /// The `max_node_id` and `max_edge_id` values seed the overlay's ID
    /// allocator so new entities never collide with base IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the overlay `LpgStore` cannot be created.
    pub fn new(
        base: CompactStore,
        max_node_id: u64,
        max_edge_id: u64,
    ) -> Result<Self, grafeo_common::memory::AllocError> {
        let overlay = Arc::new(LpgStore::new()?);
        overlay.set_next_node_id(max_node_id + 1);
        overlay.set_next_edge_id(max_edge_id + 1);

        Ok(Self {
            base: Arc::new(base),
            overlay,
            dirty_node_ids: RwLock::new(FxHashSet::default()),
            dirty_edge_ids: RwLock::new(FxHashSet::default()),
            deleted_from_base_nodes: RwLock::new(FxHashSet::default()),
            deleted_from_base_edges: RwLock::new(FxHashSet::default()),
        })
    }

    /// Returns a reference to the compact base store.
    #[must_use]
    pub fn base_store(&self) -> &CompactStore {
        &self.base
    }

    /// Returns a shared reference to the compact base store.
    #[must_use]
    pub fn base_store_arc(&self) -> Arc<CompactStore> {
        Arc::clone(&self.base)
    }

    /// Returns a reference to the overlay LPG store.
    #[must_use]
    pub fn overlay_store(&self) -> &Arc<LpgStore> {
        &self.overlay
    }

    /// Number of dirty (modified/created) entities in the overlay.
    #[must_use]
    pub fn overlay_mutation_count(&self) -> usize {
        self.dirty_node_ids.read().len()
            + self.dirty_edge_ids.read().len()
            + self.deleted_from_base_nodes.read().len()
            + self.deleted_from_base_edges.read().len()
    }

    /// Approximate heap memory of both layers.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let (store_mem, index_mem, mvcc_mem, pool_mem) = self.overlay.memory_breakdown();
        self.base.memory_bytes()
            + store_mem.total_bytes
            + index_mem.total_bytes
            + mvcc_mem.total_bytes
            + pool_mem.total_bytes
    }

    /// Checks whether a node ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_node_dirty(&self, id: NodeId) -> bool {
        self.dirty_node_ids.read().contains(&id)
    }

    /// Checks whether a node was deleted from the base.
    #[inline]
    fn is_node_deleted_from_base(&self, id: NodeId) -> bool {
        self.deleted_from_base_nodes.read().contains(&id)
    }

    /// Checks whether an edge ID is in the overlay (dirty or deleted).
    #[inline]
    fn is_edge_dirty(&self, id: EdgeId) -> bool {
        self.dirty_edge_ids.read().contains(&id)
    }

    /// Checks whether an edge was deleted from the base.
    #[inline]
    fn is_edge_deleted_from_base(&self, id: EdgeId) -> bool {
        self.deleted_from_base_edges.read().contains(&id)
    }
}

// ── GraphStore implementation ──────────────────────────────────────

impl GraphStore for LayeredStore {
    fn get_node(&self, id: NodeId) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node(id);
        }
        self.base.get_node(id)
    }

    fn get_edge(&self, id: EdgeId) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge(id);
        }
        self.base.get_edge(id)
    }

    fn get_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node_versioned(id, epoch, transaction_id);
        }
        self.base.get_node(id)
    }

    fn get_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_versioned(id, epoch, transaction_id);
        }
        self.base.get_edge(id)
    }

    fn get_node_at_epoch(&self, id: NodeId, epoch: EpochId) -> Option<Node> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node_at_epoch(id, epoch);
        }
        self.base.get_node(id)
    }

    fn get_edge_at_epoch(&self, id: EdgeId, epoch: EpochId) -> Option<Edge> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_at_epoch(id, epoch);
        }
        self.base.get_edge(id)
    }

    fn get_node_property(&self, id: NodeId, key: &PropertyKey) -> Option<Value> {
        if self.is_node_deleted_from_base(id) {
            return None;
        }
        if self.is_node_dirty(id) {
            return self.overlay.get_node_property(id, key);
        }
        self.base.get_node_property(id, key)
    }

    fn get_edge_property(&self, id: EdgeId, key: &PropertyKey) -> Option<Value> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_property(id, key);
        }
        self.base.get_edge_property(id, key)
    }

    fn get_node_property_batch(&self, ids: &[NodeId], key: &PropertyKey) -> Vec<Option<Value>> {
        ids.iter()
            .map(|id| self.get_node_property(*id, key))
            .collect()
    }

    fn get_nodes_properties_batch(&self, ids: &[NodeId]) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                self.get_node(*id)
                    .map(|n| {
                        n.properties
                            .iter()
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect()
    }

    fn get_nodes_properties_selective_batch(
        &self,
        ids: &[NodeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_node_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn get_edges_properties_selective_batch(
        &self,
        ids: &[EdgeId],
        keys: &[PropertyKey],
    ) -> Vec<FxHashMap<PropertyKey, Value>> {
        ids.iter()
            .map(|id| {
                let mut map = FxHashMap::default();
                for key in keys {
                    if let Some(v) = self.get_edge_property(*id, key) {
                        map.insert(key.clone(), v);
                    }
                }
                map
            })
            .collect()
    }

    fn neighbors(&self, node: NodeId, direction: Direction) -> Vec<NodeId> {
        let deleted_nodes = self.deleted_from_base_nodes.read();

        let mut results = Vec::new();

        // Base neighbors (minus deleted).
        if !deleted_nodes.contains(&node) && !self.is_node_dirty(node) {
            for nid in self.base.neighbors(node, direction) {
                if !deleted_nodes.contains(&nid) {
                    results.push(nid);
                }
            }
        }

        // Overlay neighbors (for dirty source node or overlay-only node).
        if self.is_node_dirty(node) || self.overlay.get_node(node).is_some() {
            for nid in self.overlay.neighbors(node, direction) {
                if !deleted_nodes.contains(&nid) {
                    results.push(nid);
                }
            }
        }

        // Overlay edges that connect base nodes.
        // If node is in the base and the overlay has edges for it
        // (e.g., after promote), those are already in dirty.

        results.sort_unstable();
        results.dedup();
        results
    }

    fn edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let deleted_nodes = self.deleted_from_base_nodes.read();
        let deleted_edges = self.deleted_from_base_edges.read();

        let mut results = Vec::new();

        // Base edges (minus deleted).
        if !deleted_nodes.contains(&node) && !self.is_node_dirty(node) {
            for (target, eid) in self.base.edges_from(node, direction) {
                if !deleted_nodes.contains(&target) && !deleted_edges.contains(&eid) {
                    results.push((target, eid));
                }
            }
        }

        // Overlay edges.
        if self.is_node_dirty(node) || self.overlay.get_node(node).is_some() {
            for (target, eid) in self.overlay.edges_from(node, direction) {
                if !deleted_nodes.contains(&target) && !deleted_edges.contains(&eid) {
                    results.push((target, eid));
                }
            }
        }

        results
    }

    fn out_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Outgoing).len()
    }

    fn in_degree(&self, node: NodeId) -> usize {
        self.edges_from(node, Direction::Incoming).len()
    }

    fn has_backward_adjacency(&self) -> bool {
        self.base.has_backward_adjacency() || self.overlay.has_backward_adjacency()
    }

    fn node_ids(&self) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();

        let mut ids: Vec<NodeId> = self
            .base
            .node_ids()
            .into_iter()
            .filter(|id| !deleted.contains(id))
            .collect();
        ids.extend(self.overlay.node_ids());
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn nodes_by_label(&self, label: &str) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();

        let mut ids: Vec<NodeId> = self
            .base
            .nodes_by_label(label)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !self.is_node_dirty(*id))
            .collect();
        ids.extend(
            self.overlay
                .nodes_by_label(label)
                .into_iter()
                .filter(|id| !deleted.contains(id)),
        );
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn node_count(&self) -> usize {
        let base_count = self.base.node_count();
        let deleted = self.deleted_from_base_nodes.read().len();
        let overlay_count = self.overlay.node_count();
        // Dirty nodes that came from the base are counted once in the overlay.
        // We subtract them from the base total to avoid double counting.
        let promoted = self
            .dirty_node_ids
            .read()
            .iter()
            .filter(|id| self.base.get_node(**id).is_some())
            .count();
        base_count - deleted - promoted + overlay_count
    }

    fn edge_count(&self) -> usize {
        let base_count = self.base.edge_count();
        let deleted = self.deleted_from_base_edges.read().len();
        let overlay_count = self.overlay.edge_count();
        let promoted = self
            .dirty_edge_ids
            .read()
            .iter()
            .filter(|id| self.base.get_edge(**id).is_some())
            .count();
        base_count - deleted - promoted + overlay_count
    }

    fn edge_type(&self, id: EdgeId) -> Option<ArcStr> {
        if self.is_edge_deleted_from_base(id) {
            return None;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.edge_type(id);
        }
        self.base.edge_type(id)
    }

    fn find_nodes_by_property(&self, property: &str, value: &Value) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .find_nodes_by_property(property, value)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.find_nodes_by_property(property, value));
        results
    }

    fn find_nodes_by_properties(&self, conditions: &[(&str, Value)]) -> Vec<NodeId> {
        if conditions.is_empty() {
            return self.node_ids();
        }
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .find_nodes_by_properties(conditions)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.find_nodes_by_properties(conditions));
        results
    }

    fn find_nodes_in_range(
        &self,
        property: &str,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> Vec<NodeId> {
        let deleted = self.deleted_from_base_nodes.read();
        let dirty = self.dirty_node_ids.read();

        let mut results: Vec<NodeId> = self
            .base
            .find_nodes_in_range(property, min, max, min_inclusive, max_inclusive)
            .into_iter()
            .filter(|id| !deleted.contains(id) && !dirty.contains(id))
            .collect();

        results.extend(self.overlay.find_nodes_in_range(
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
        ));
        results
    }

    fn node_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.base.node_property_might_match(property, op, value)
            || self.overlay.node_property_might_match(property, op, value)
    }

    fn edge_property_might_match(
        &self,
        property: &PropertyKey,
        op: CompareOp,
        value: &Value,
    ) -> bool {
        self.base.edge_property_might_match(property, op, value)
            || self.overlay.edge_property_might_match(property, op, value)
    }

    fn statistics(&self) -> Arc<Statistics> {
        // Combine base + overlay statistics.
        let base_stats = self.base.statistics();

        let mut combined = (*base_stats).clone();
        combined.total_nodes = self.node_count() as u64;
        combined.total_edges = self.edge_count() as u64;

        // Merge label stats from overlay.
        for label in self.overlay.all_labels() {
            let count = self.overlay.nodes_by_label(&label).len() as u64;
            if let Some(existing) = combined.get_label(&label) {
                combined.update_label(
                    &label,
                    crate::statistics::LabelStatistics::new(existing.node_count + count),
                );
            } else {
                combined.update_label(&label, crate::statistics::LabelStatistics::new(count));
            }
        }

        Arc::new(combined)
    }

    fn estimate_label_cardinality(&self, label: &str) -> f64 {
        self.base.estimate_label_cardinality(label) + self.overlay.estimate_label_cardinality(label)
    }

    fn estimate_avg_degree(&self, edge_type: &str, outgoing: bool) -> f64 {
        // Rough approximation: weighted average.
        let base_est = self.base.estimate_avg_degree(edge_type, outgoing);
        let overlay_est = self.overlay.estimate_avg_degree(edge_type, outgoing);
        let base_edges = self.base.edge_count() as f64;
        let overlay_edges = self.overlay.edge_count() as f64;
        let total = base_edges + overlay_edges;
        if total == 0.0 {
            return 0.0;
        }
        (base_est * base_edges + overlay_est * overlay_edges) / total
    }

    fn current_epoch(&self) -> EpochId {
        self.overlay.current_epoch()
    }

    fn all_labels(&self) -> Vec<String> {
        let mut labels: FxHashSet<String> = self.base.all_labels().into_iter().collect();
        labels.extend(self.overlay.all_labels());
        labels.into_iter().collect()
    }

    fn all_edge_types(&self) -> Vec<String> {
        let mut types: FxHashSet<String> = self.base.all_edge_types().into_iter().collect();
        types.extend(self.overlay.all_edge_types());
        types.into_iter().collect()
    }

    fn all_property_keys(&self) -> Vec<String> {
        let mut keys: FxHashSet<String> = self.base.all_property_keys().into_iter().collect();
        keys.extend(self.overlay.all_property_keys());
        keys.into_iter().collect()
    }

    fn is_node_visible_at_epoch(&self, id: NodeId, epoch: EpochId) -> bool {
        if self.is_node_deleted_from_base(id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self.overlay.is_node_visible_at_epoch(id, epoch);
        }
        self.base.is_node_visible_at_epoch(id, epoch)
    }

    fn is_node_visible_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_node_deleted_from_base(id) {
            return false;
        }
        if self.is_node_dirty(id) {
            return self
                .overlay
                .is_node_visible_versioned(id, epoch, transaction_id);
        }
        self.base
            .is_node_visible_versioned(id, epoch, transaction_id)
    }

    fn is_edge_visible_at_epoch(&self, id: EdgeId, epoch: EpochId) -> bool {
        if self.is_edge_deleted_from_base(id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return self.overlay.is_edge_visible_at_epoch(id, epoch);
        }
        self.base.is_edge_visible_at_epoch(id, epoch)
    }

    fn is_edge_visible_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_edge_deleted_from_base(id) {
            return false;
        }
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .is_edge_visible_versioned(id, epoch, transaction_id);
        }
        self.base
            .is_edge_visible_versioned(id, epoch, transaction_id)
    }

    fn filter_visible_node_ids(&self, ids: &[NodeId], epoch: EpochId) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_at_epoch(*id, epoch))
            .collect()
    }

    fn filter_visible_node_ids_versioned(
        &self,
        ids: &[NodeId],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> Vec<NodeId> {
        ids.iter()
            .copied()
            .filter(|id| self.is_node_visible_versioned(*id, epoch, transaction_id))
            .collect()
    }

    fn get_node_history(&self, id: NodeId) -> Vec<(EpochId, Option<EpochId>, Node)> {
        if self.is_node_dirty(id) {
            return self.overlay.get_node_history(id);
        }
        Vec::new()
    }

    fn get_edge_history(&self, id: EdgeId) -> Vec<(EpochId, Option<EpochId>, Edge)> {
        if self.is_edge_dirty(id) {
            return self.overlay.get_edge_history(id);
        }
        Vec::new()
    }
}

// ── GraphStoreMut implementation ───────────────────────────────────

impl GraphStoreMut for LayeredStore {
    fn create_node(&self, labels: &[&str]) -> NodeId {
        let id = self.overlay.create_node(labels);
        self.dirty_node_ids.write().insert(id);
        id
    }

    fn create_node_versioned(
        &self,
        labels: &[&str],
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> NodeId {
        let id = self
            .overlay
            .create_node_versioned(labels, epoch, transaction_id);
        self.dirty_node_ids.write().insert(id);
        id
    }

    fn create_edge(&self, src: NodeId, dst: NodeId, edge_type: &str) -> EdgeId {
        // Promote base-only endpoints into the overlay.
        self.ensure_in_overlay(src);
        self.ensure_in_overlay(dst);
        let id = self.overlay.create_edge(src, dst, edge_type);
        self.dirty_edge_ids.write().insert(id);
        id
    }

    fn create_edge_versioned(
        &self,
        src: NodeId,
        dst: NodeId,
        edge_type: &str,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> EdgeId {
        self.ensure_in_overlay(src);
        self.ensure_in_overlay(dst);
        let id = self
            .overlay
            .create_edge_versioned(src, dst, edge_type, epoch, transaction_id);
        self.dirty_edge_ids.write().insert(id);
        id
    }

    fn batch_create_edges(&self, edges: &[(NodeId, NodeId, &str)]) -> Vec<EdgeId> {
        for &(src, dst, _) in edges {
            self.ensure_in_overlay(src);
            self.ensure_in_overlay(dst);
        }
        let ids = self.overlay.batch_create_edges(edges);
        let mut dirty = self.dirty_edge_ids.write();
        for &id in &ids {
            dirty.insert(id);
        }
        ids
    }

    fn delete_node(&self, id: NodeId) -> bool {
        if self.is_node_dirty(id) {
            // Node is in the overlay: delete from overlay.
            return self.overlay.delete_node(id);
        }
        if self.base.get_node(id).is_some() {
            self.deleted_from_base_nodes.write().insert(id);
            return true;
        }
        false
    }

    fn delete_node_versioned(
        &self,
        id: NodeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_node_dirty(id) {
            return self
                .overlay
                .delete_node_versioned(id, epoch, transaction_id);
        }
        if self.base.get_node(id).is_some() {
            self.deleted_from_base_nodes.write().insert(id);
            return true;
        }
        false
    }

    fn delete_node_edges(&self, node_id: NodeId) {
        // Delete overlay edges.
        if self.is_node_dirty(node_id) {
            self.overlay.delete_node_edges(node_id);
        }
        // Mark base edges as deleted.
        for (_, eid) in self.base.edges_from(node_id, Direction::Both) {
            self.deleted_from_base_edges.write().insert(eid);
        }
    }

    fn delete_edge(&self, id: EdgeId) -> bool {
        if self.is_edge_dirty(id) {
            return self.overlay.delete_edge(id);
        }
        if self.base.get_edge(id).is_some() {
            self.deleted_from_base_edges.write().insert(id);
            return true;
        }
        false
    }

    fn delete_edge_versioned(
        &self,
        id: EdgeId,
        epoch: EpochId,
        transaction_id: TransactionId,
    ) -> bool {
        if self.is_edge_dirty(id) {
            return self
                .overlay
                .delete_edge_versioned(id, epoch, transaction_id);
        }
        if self.base.get_edge(id).is_some() {
            self.deleted_from_base_edges.write().insert(id);
            return true;
        }
        false
    }

    fn set_node_property(&self, id: NodeId, key: &str, value: Value) {
        self.ensure_in_overlay(id);
        self.overlay.set_node_property(id, key, value);
    }

    fn set_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.ensure_in_overlay(id);
        self.overlay
            .set_node_property_versioned(id, key, value, transaction_id);
    }

    fn set_edge_property(&self, id: EdgeId, key: &str, value: Value) {
        self.ensure_edge_in_overlay(id);
        self.overlay.set_edge_property(id, key, value);
    }

    fn set_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        value: Value,
        transaction_id: TransactionId,
    ) {
        self.ensure_edge_in_overlay(id);
        self.overlay
            .set_edge_property_versioned(id, key, value, transaction_id);
    }

    fn remove_node_property(&self, id: NodeId, key: &str) -> Option<Value> {
        self.ensure_in_overlay(id);
        self.overlay.remove_node_property(id, key)
    }

    fn remove_node_property_versioned(
        &self,
        id: NodeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.ensure_in_overlay(id);
        self.overlay
            .remove_node_property_versioned(id, key, transaction_id)
    }

    fn remove_edge_property(&self, id: EdgeId, key: &str) -> Option<Value> {
        self.ensure_edge_in_overlay(id);
        self.overlay.remove_edge_property(id, key)
    }

    fn remove_edge_property_versioned(
        &self,
        id: EdgeId,
        key: &str,
        transaction_id: TransactionId,
    ) -> Option<Value> {
        self.ensure_edge_in_overlay(id);
        self.overlay
            .remove_edge_property_versioned(id, key, transaction_id)
    }

    fn add_label(&self, node_id: NodeId, label: &str) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay.add_label(node_id, label)
    }

    fn add_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay
            .add_label_versioned(node_id, label, transaction_id)
    }

    fn remove_label(&self, node_id: NodeId, label: &str) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay.remove_label(node_id, label)
    }

    fn remove_label_versioned(
        &self,
        node_id: NodeId,
        label: &str,
        transaction_id: TransactionId,
    ) -> bool {
        self.ensure_in_overlay(node_id);
        self.overlay
            .remove_label_versioned(node_id, label, transaction_id)
    }
}

// ── Private helpers ────────────────────────────────────────────────

impl LayeredStore {
    /// Ensures a node exists in the overlay. If the node is base-only,
    /// copies its labels and properties into the overlay and marks it dirty.
    fn ensure_in_overlay(&self, id: NodeId) {
        if self.is_node_dirty(id) {
            return; // already in overlay
        }
        let Some(base_node) = self.base.get_node(id) else {
            return; // not in base either (new node case handled by caller)
        };

        // Copy the node into the overlay at the same ID.
        // We temporarily lower the ID counter, create the node, then restore it.
        let saved_next = self.overlay.next_node_id();
        self.overlay.set_next_node_id(id.as_u64());
        let labels: Vec<&str> = base_node.labels.iter().map(|l| l.as_str()).collect();
        let promoted_id = self.overlay.create_node(&labels);
        debug_assert_eq!(
            promoted_id, id,
            "promoted node should reuse the original ID"
        );
        self.overlay.set_next_node_id(saved_next);

        // Copy properties.
        for (key, value) in base_node.properties.iter() {
            self.overlay
                .set_node_property(id, key.as_str(), value.clone());
        }

        self.dirty_node_ids.write().insert(id);
    }

    /// Ensures an edge exists in the overlay.
    fn ensure_edge_in_overlay(&self, id: EdgeId) {
        if self.is_edge_dirty(id) {
            return;
        }
        let Some(base_edge) = self.base.get_edge(id) else {
            return;
        };

        // Ensure endpoints are in the overlay first.
        self.ensure_in_overlay(base_edge.src);
        self.ensure_in_overlay(base_edge.dst);

        // Create the edge at the same ID.
        let saved_next = self.overlay.next_edge_id();
        self.overlay.set_next_edge_id(id.as_u64());
        let promoted_id =
            self.overlay
                .create_edge(base_edge.src, base_edge.dst, base_edge.edge_type.as_str());
        debug_assert_eq!(
            promoted_id, id,
            "promoted edge should reuse the original ID"
        );
        self.overlay.set_next_edge_id(saved_next);

        // Copy properties.
        for (key, value) in base_edge.properties.iter() {
            self.overlay
                .set_edge_property(id, key.as_str(), value.clone());
        }

        self.dirty_edge_ids.write().insert(id);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::compact::from_graph_store_preserving_ids;

    fn build_test_layered() -> LayeredStore {
        let store = LpgStore::new().unwrap();

        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));

        let gus = store.create_node(&["Person"]);
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(25));

        let amsterdam = store.create_node(&["City"]);
        store.set_node_property(amsterdam, "name", Value::from("Amsterdam"));

        store.create_edge(alix, amsterdam, "LIVES_IN");
        store.create_edge(gus, amsterdam, "LIVES_IN");

        let compact = from_graph_store_preserving_ids(&store).unwrap();
        let max_nid = store
            .node_ids()
            .into_iter()
            .map(|id| id.as_u64())
            .max()
            .unwrap_or(0);
        let max_eid = 10u64; // edges start at 0 in LpgStore
        LayeredStore::new(compact, max_nid, max_eid).unwrap()
    }

    #[test]
    fn test_read_through_base() {
        let layered = build_test_layered();
        assert_eq!(layered.node_count(), 3);
        assert_eq!(layered.edge_count(), 2);

        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);
    }

    #[test]
    fn test_create_node_in_overlay() {
        let layered = build_test_layered();
        let vincent = layered.create_node(&["Person"]);
        layered.set_node_property(vincent, "name", Value::from("Vincent"));

        assert_eq!(layered.node_count(), 4);
        let node = layered.get_node(vincent).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String(ArcStr::from("Vincent")))
        );
    }

    #[test]
    fn test_delete_base_node() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        assert_eq!(persons.len(), 2);

        let deleted = layered.delete_node(persons[0]);
        assert!(deleted);
        assert!(layered.get_node(persons[0]).is_none());

        let remaining_persons = layered.nodes_by_label("Person");
        assert_eq!(remaining_persons.len(), 1);
        assert_eq!(layered.node_count(), 2);
    }

    #[test]
    fn test_modify_base_node_property() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let first = persons[0];

        // Original value.
        let original_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert!(matches!(original_age, Value::Int64(_)));

        // Modify: this should promote the node to the overlay.
        layered.set_node_property(first, "age", Value::Int64(99));

        let new_age = layered
            .get_node_property(first, &PropertyKey::new("age"))
            .unwrap();
        assert_eq!(new_age, Value::Int64(99));
    }

    #[test]
    fn test_create_edge_between_base_and_overlay() {
        let layered = build_test_layered();
        let paris = layered.create_node(&["City"]);
        layered.set_node_property(paris, "name", Value::from("Paris"));

        let persons = layered.nodes_by_label("Person");
        let first_person = persons[0];

        // Create cross-layer edge.
        let eid = layered.create_edge(first_person, paris, "VISITS");
        assert!(layered.get_edge(eid).is_some());

        let edge = layered.get_edge(eid).unwrap();
        assert_eq!(edge.src, first_person);
        assert_eq!(edge.dst, paris);
    }

    #[test]
    fn test_traversal_merges_layers() {
        let layered = build_test_layered();
        let cities = layered.nodes_by_label("City");
        let amsterdam = cities[0];

        // Base has 2 incoming LIVES_IN edges.
        let incoming = layered.edges_from(amsterdam, Direction::Incoming);
        assert_eq!(incoming.len(), 2);
    }

    #[test]
    fn test_node_ids_combines_layers() {
        let layered = build_test_layered();
        let initial = layered.node_ids();
        assert_eq!(initial.len(), 3);

        layered.create_node(&["New"]);
        let after = layered.node_ids();
        assert_eq!(after.len(), 4);
    }

    #[test]
    fn test_delete_edge() {
        let layered = build_test_layered();
        let persons = layered.nodes_by_label("Person");
        let edges = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(edges.len(), 1);

        let (_, eid) = edges[0];
        let deleted = layered.delete_edge(eid);
        assert!(deleted);

        let after = layered.edges_from(persons[0], Direction::Outgoing);
        assert_eq!(after.len(), 0);
    }

    #[test]
    fn test_all_labels_combines() {
        let layered = build_test_layered();
        layered.create_node(&["NewLabel"]);

        let labels = layered.all_labels();
        assert!(labels.contains(&"Person".to_string()));
        assert!(labels.contains(&"City".to_string()));
        assert!(labels.contains(&"NewLabel".to_string()));
    }
}
