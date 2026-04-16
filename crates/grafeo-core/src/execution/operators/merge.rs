//! Merge operator for MERGE clause execution.
//!
//! The MERGE operator implements the Cypher MERGE semantics:
//! 1. Try to match the pattern in the graph
//! 2. If found, return existing element (optionally apply ON MATCH SET)
//! 3. If not found, create the element (optionally apply ON CREATE SET)

use super::{ConstraintValidator, Operator, OperatorResult, PropertySource};
use crate::execution::chunk::{DataChunk, DataChunkBuilder};
use crate::graph::{GraphStore, GraphStoreMut};
use grafeo_common::types::{
    EdgeId, EpochId, LogicalType, NodeId, PropertyKey, TransactionId, Value,
};
use std::sync::Arc;

/// Configuration for a node merge operation.
pub struct MergeConfig {
    /// Variable name for the merged node.
    pub variable: String,
    /// Labels to match/create.
    pub labels: Vec<String>,
    /// Properties that must match (also used for creation).
    pub match_properties: Vec<(String, PropertySource)>,
    /// Properties to set on CREATE.
    pub on_create_properties: Vec<(String, PropertySource)>,
    /// Properties to set on MATCH.
    pub on_match_properties: Vec<(String, PropertySource)>,
    /// Output schema (input columns + node column).
    pub output_schema: Vec<LogicalType>,
    /// Column index where the merged node ID is placed.
    pub output_column: usize,
    /// If the merge variable was already bound in the input, this column index
    /// is used to detect NULL references (e.g., from unmatched OPTIONAL MATCH).
    /// `None` for standalone MERGE that introduces a new variable.
    pub bound_variable_column: Option<usize>,
}

/// Merge operator for MERGE clause.
///
/// Tries to match a node with the given labels and properties.
/// If found, returns the existing node. If not found, creates a new node.
///
/// When an input operator is provided (chained MERGE), input rows are
/// passed through with the merged node ID appended as an additional column.
pub struct MergeOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
    /// Optional input operator (for chained MERGE patterns).
    input: Option<Box<dyn Operator>>,
    /// Merge configuration.
    config: MergeConfig,
    /// Whether we've already executed (standalone mode only).
    executed: bool,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
}

impl MergeOperator {
    /// Creates a new merge operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Option<Box<dyn Operator>>,
        config: MergeConfig,
    ) -> Self {
        Self {
            store,
            input,
            config,
            executed: false,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
        }
    }

    /// Returns the variable name for the merged node.
    #[must_use]
    pub fn variable(&self) -> &str {
        &self.config.variable
    }

    /// Sets the transaction context for versioned mutations.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the constraint validator for schema enforcement.
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Resolves property sources to concrete values for a given row.
    fn resolve_properties(
        props: &[(String, PropertySource)],
        chunk: Option<&DataChunk>,
        row: usize,
        store: &dyn GraphStore,
    ) -> Vec<(String, Value)> {
        props
            .iter()
            .map(|(name, source)| {
                let value = if let Some(chunk) = chunk {
                    source.resolve(chunk, row, store)
                } else {
                    // Standalone mode: only constants are valid
                    match source {
                        PropertySource::Constant(v) => v.clone(),
                        _ => Value::Null,
                    }
                };
                (name.clone(), value)
            })
            .collect()
    }

    /// Tries to find a matching node with the given resolved properties.
    fn find_matching_node(&self, resolved_match_props: &[(String, Value)]) -> Option<NodeId> {
        let candidates: Vec<NodeId> = if let Some(first_label) = self.config.labels.first() {
            self.store.nodes_by_label(first_label)
        } else {
            self.store.node_ids()
        };

        for node_id in candidates {
            if let Some(node) = self.store.get_node(node_id) {
                let has_all_labels = self.config.labels.iter().all(|label| node.has_label(label));
                if !has_all_labels {
                    continue;
                }

                let has_all_props = resolved_match_props.iter().all(|(key, expected_value)| {
                    let prop = node.properties.get(&PropertyKey::new(key.as_str()));
                    if expected_value.is_null() {
                        // Null in a MERGE pattern matches both absent and explicitly null properties
                        prop.map_or(true, |v| v.is_null())
                    } else {
                        prop.is_some_and(|v| v == expected_value)
                    }
                });

                if has_all_props {
                    return Some(node_id);
                }
            }
        }

        None
    }

    /// Creates a new node with the specified labels and resolved properties.
    fn create_node(
        &self,
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Result<NodeId, super::OperatorError> {
        // Validate constraints before creating the node
        if let Some(ref validator) = self.validator {
            validator.validate_node_labels_allowed(&self.config.labels)?;

            let all_props: Vec<(String, Value)> = resolved_match_props
                .iter()
                .chain(resolved_create_props.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (name, value) in &all_props {
                validator.validate_node_property(&self.config.labels, name, value)?;
                validator.check_unique_node_property(&self.config.labels, name, value)?;
            }
            validator.validate_node_complete(&self.config.labels, &all_props)?;
        }

        let mut all_props: Vec<(PropertyKey, Value)> = resolved_match_props
            .iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone()))
            .collect();

        for (k, v) in resolved_create_props {
            if let Some(existing) = all_props.iter_mut().find(|(key, _)| key.as_str() == k) {
                existing.1 = v.clone();
            } else {
                all_props.push((PropertyKey::new(k.as_str()), v.clone()));
            }
        }

        let labels: Vec<&str> = self.config.labels.iter().map(String::as_str).collect();
        Ok(self.store.create_node_with_props(&labels, &all_props))
    }

    /// Finds or creates a matching node for a single row, applying ON MATCH/ON CREATE.
    fn merge_node_for_row(
        &self,
        chunk: Option<&DataChunk>,
        row: usize,
    ) -> Result<NodeId, super::OperatorError> {
        let store_ref: &dyn GraphStore = self.store.as_ref();
        let resolved_match =
            Self::resolve_properties(&self.config.match_properties, chunk, row, store_ref);

        if let Some(existing_id) = self.find_matching_node(&resolved_match) {
            let resolved_on_match =
                Self::resolve_properties(&self.config.on_match_properties, chunk, row, store_ref);
            self.apply_on_match(existing_id, &resolved_on_match)?;
            Ok(existing_id)
        } else {
            let resolved_on_create =
                Self::resolve_properties(&self.config.on_create_properties, chunk, row, store_ref);
            self.create_node(&resolved_match, &resolved_on_create)
        }
    }

    /// Applies ON MATCH properties to an existing node.
    fn apply_on_match(
        &self,
        node_id: NodeId,
        resolved_on_match: &[(String, Value)],
    ) -> Result<(), super::OperatorError> {
        for (key, value) in resolved_on_match {
            if let Some(ref validator) = self.validator {
                validator.validate_node_property(&self.config.labels, key, value)?;
            }
            if let Some(tid) = self.transaction_id {
                self.store
                    .set_node_property_versioned(node_id, key.as_str(), value.clone(), tid);
            } else {
                self.store
                    .set_node_property(node_id, key.as_str(), value.clone());
            }
        }
        Ok(())
    }
}

impl Operator for MergeOperator {
    fn next(&mut self) -> OperatorResult {
        // When we have an input operator, pass through input rows with the
        // merged node ID appended (used for chained inline MERGE patterns).
        if let Some(ref mut input) = self.input {
            if let Some(chunk) = input.next()? {
                let mut builder =
                    DataChunkBuilder::with_capacity(&self.config.output_schema, chunk.row_count());

                for row in chunk.selected_indices() {
                    // Reject NULL bound variables (e.g., from unmatched OPTIONAL MATCH)
                    if let Some(bound_col) = self.config.bound_variable_column {
                        let is_null = chunk.column(bound_col).map_or(true, |col| col.is_null(row));
                        if is_null {
                            return Err(super::OperatorError::TypeMismatch {
                                expected: format!(
                                    "non-null node for MERGE variable '{}'",
                                    self.config.variable
                                ),
                                found: "NULL".to_string(),
                            });
                        }
                    }

                    // Merge the node per-row: resolve properties from this row
                    let node_id = self.merge_node_for_row(Some(&chunk), row)?;

                    // Copy input columns to output
                    for col_idx in 0..chunk.column_count() {
                        if let (Some(src), Some(dst)) =
                            (chunk.column(col_idx), builder.column_mut(col_idx))
                        {
                            if let Some(val) = src.get_value(row) {
                                dst.push_value(val);
                            } else {
                                dst.push_value(Value::Null);
                            }
                        }
                    }

                    // Append the merged node ID
                    if let Some(dst) = builder.column_mut(self.config.output_column) {
                        dst.push_node_id(node_id);
                    }

                    builder.advance_row();
                }

                return Ok(Some(builder.finish()));
            }
            return Ok(None);
        }

        // Standalone mode (no input operator)
        if self.executed {
            return Ok(None);
        }
        self.executed = true;

        let node_id = self.merge_node_for_row(None, 0)?;

        let mut builder = DataChunkBuilder::new(&self.config.output_schema);
        if let Some(dst) = builder.column_mut(self.config.output_column) {
            dst.push_node_id(node_id);
        }
        builder.advance_row();

        Ok(Some(builder.finish()))
    }

    fn reset(&mut self) {
        self.executed = false;
        if let Some(ref mut input) = self.input {
            input.reset();
        }
    }

    fn name(&self) -> &'static str {
        "Merge"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Configuration for a relationship merge operation.
pub struct MergeRelationshipConfig {
    /// Column index for the source node ID in the input.
    pub source_column: usize,
    /// Column index for the target node ID in the input.
    pub target_column: usize,
    /// Variable name for the source node (for error messages).
    pub source_variable: String,
    /// Variable name for the target node (for error messages).
    pub target_variable: String,
    /// Relationship type to match/create.
    pub edge_type: String,
    /// Properties that must match (also used for creation).
    pub match_properties: Vec<(String, PropertySource)>,
    /// Properties to set on CREATE.
    pub on_create_properties: Vec<(String, PropertySource)>,
    /// Properties to set on MATCH.
    pub on_match_properties: Vec<(String, PropertySource)>,
    /// Output schema (input columns + edge column).
    pub output_schema: Vec<LogicalType>,
    /// Column index for the edge variable in the output.
    pub edge_output_column: usize,
}

/// Merge operator for relationship patterns.
///
/// Takes input rows containing source and target node IDs, then for each row:
/// 1. Searches for an existing relationship matching the type and properties
/// 2. If found, applies ON MATCH properties and returns the existing edge
/// 3. If not found, creates a new relationship and applies ON CREATE properties
pub struct MergeRelationshipOperator {
    /// The graph store.
    store: Arc<dyn GraphStoreMut>,
    /// Input operator providing rows with source/target node columns.
    input: Box<dyn Operator>,
    /// Merge configuration.
    config: MergeRelationshipConfig,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for undo log tracking.
    transaction_id: Option<TransactionId>,
    /// Optional constraint validator for schema enforcement.
    validator: Option<Arc<dyn ConstraintValidator>>,
}

impl MergeRelationshipOperator {
    /// Creates a new merge relationship operator.
    pub fn new(
        store: Arc<dyn GraphStoreMut>,
        input: Box<dyn Operator>,
        config: MergeRelationshipConfig,
    ) -> Self {
        Self {
            store,
            input,
            config,
            viewing_epoch: None,
            transaction_id: None,
            validator: None,
        }
    }

    /// Sets the transaction context for versioned mutations.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the constraint validator for schema enforcement.
    pub fn with_validator(mut self, validator: Arc<dyn ConstraintValidator>) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Tries to find a matching relationship between source and target.
    fn find_matching_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        resolved_match_props: &[(String, Value)],
    ) -> Option<EdgeId> {
        use crate::graph::Direction;

        for (target, edge_id) in self.store.edges_from(src, Direction::Outgoing) {
            if target != dst {
                continue;
            }

            if let Some(edge) = self.store.get_edge(edge_id) {
                if edge.edge_type.as_str() != self.config.edge_type {
                    continue;
                }

                let has_all_props = resolved_match_props
                    .iter()
                    .all(|(key, expected)| edge.get_property(key).is_some_and(|v| v == expected));

                if has_all_props {
                    return Some(edge_id);
                }
            }
        }

        None
    }

    /// Creates a new edge with resolved match and on_create properties.
    fn create_edge(
        &self,
        src: NodeId,
        dst: NodeId,
        resolved_match_props: &[(String, Value)],
        resolved_create_props: &[(String, Value)],
    ) -> Result<EdgeId, super::OperatorError> {
        // Validate constraints before creating the edge
        if let Some(ref validator) = self.validator {
            validator.validate_edge_type_allowed(&self.config.edge_type)?;

            let all_props: Vec<(String, Value)> = resolved_match_props
                .iter()
                .chain(resolved_create_props.iter())
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            for (name, value) in &all_props {
                validator.validate_edge_property(&self.config.edge_type, name, value)?;
            }
            validator.validate_edge_complete(&self.config.edge_type, &all_props)?;
        }

        let mut all_props: Vec<(PropertyKey, Value)> = resolved_match_props
            .iter()
            .map(|(k, v)| (PropertyKey::new(k.as_str()), v.clone()))
            .collect();

        for (k, v) in resolved_create_props {
            if let Some(existing) = all_props.iter_mut().find(|(key, _)| key.as_str() == k) {
                existing.1 = v.clone();
            } else {
                all_props.push((PropertyKey::new(k.as_str()), v.clone()));
            }
        }

        Ok(self
            .store
            .create_edge_with_props(src, dst, &self.config.edge_type, &all_props))
    }

    /// Applies ON MATCH properties to an existing edge.
    fn apply_on_match_edge(
        &self,
        edge_id: EdgeId,
        resolved_on_match: &[(String, Value)],
    ) -> Result<(), super::OperatorError> {
        for (key, value) in resolved_on_match {
            if let Some(ref validator) = self.validator {
                validator.validate_edge_property(&self.config.edge_type, key, value)?;
            }
            if let Some(tid) = self.transaction_id {
                self.store
                    .set_edge_property_versioned(edge_id, key.as_str(), value.clone(), tid);
            } else {
                self.store
                    .set_edge_property(edge_id, key.as_str(), value.clone());
            }
        }
        Ok(())
    }
}

impl Operator for MergeRelationshipOperator {
    fn next(&mut self) -> OperatorResult {
        use super::OperatorError;

        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.config.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let src_val = chunk
                    .column(self.config.source_column)
                    .and_then(|c| c.get_node_id(row))
                    .ok_or_else(|| OperatorError::TypeMismatch {
                        expected: format!(
                            "non-null node for MERGE variable '{}'",
                            self.config.source_variable
                        ),
                        found: "NULL".to_string(),
                    })?;

                let dst_val = chunk
                    .column(self.config.target_column)
                    .and_then(|c| c.get_node_id(row))
                    .ok_or_else(|| OperatorError::TypeMismatch {
                        expected: format!(
                            "non-null node for MERGE variable '{}'",
                            self.config.target_variable
                        ),
                        found: "None".to_string(),
                    })?;

                let store_ref: &dyn GraphStore = self.store.as_ref();
                let resolved_match = MergeOperator::resolve_properties(
                    &self.config.match_properties,
                    Some(&chunk),
                    row,
                    store_ref,
                );

                let edge_id = if let Some(existing) =
                    self.find_matching_edge(src_val, dst_val, &resolved_match)
                {
                    let resolved_on_match = MergeOperator::resolve_properties(
                        &self.config.on_match_properties,
                        Some(&chunk),
                        row,
                        store_ref,
                    );
                    self.apply_on_match_edge(existing, &resolved_on_match)?;
                    existing
                } else {
                    let resolved_on_create = MergeOperator::resolve_properties(
                        &self.config.on_create_properties,
                        Some(&chunk),
                        row,
                        store_ref,
                    );
                    self.create_edge(src_val, dst_val, &resolved_match, &resolved_on_create)?
                };

                // Copy input columns to output, then add the edge column
                for col_idx in 0..self.config.output_schema.len() {
                    if col_idx == self.config.edge_output_column {
                        if let Some(dst_col) = builder.column_mut(col_idx) {
                            dst_col.push_edge_id(edge_id);
                        }
                    } else if let (Some(src_col), Some(dst_col)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                        && let Some(val) = src_col.get_value(row)
                    {
                        dst_col.push_value(val);
                    }
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }

        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "MergeRelationship"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    fn const_props(props: Vec<(&str, Value)>) -> Vec<(String, PropertySource)> {
        props
            .into_iter()
            .map(|(k, v)| (k.to_string(), PropertySource::Constant(v)))
            .collect()
    }

    #[test]
    fn test_merge_creates_new_node() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // MERGE should create a new node since none exists
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Alix".into()))]),
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // Verify node was created
        let nodes = store.nodes_by_label("Person");
        assert_eq!(nodes.len(), 1);

        let node = store.get_node(nodes[0]).unwrap();
        assert!(node.has_label("Person"));
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String("Alix".into()))
        );
    }

    #[test]
    fn test_merge_matches_existing_node() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // Create an existing node
        store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::new("name"), Value::String("Gus".into()))],
        );

        // MERGE should find the existing node
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Gus".into()))]),
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let result = merge.next().unwrap();
        assert!(result.is_some());

        // Verify only one node exists (no new node created)
        let nodes = store.nodes_by_label("Person");
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_merge_with_on_create() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // MERGE with ON CREATE SET
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Vincent".into()))]),
                on_create_properties: const_props(vec![("created", Value::Bool(true))]),
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let _ = merge.next().unwrap();

        // Verify node has both match properties and on_create properties
        let nodes = store.nodes_by_label("Person");
        let node = store.get_node(nodes[0]).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("name")),
            Some(&Value::String("Vincent".into()))
        );
        assert_eq!(
            node.properties.get(&PropertyKey::new("created")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_merge_with_on_match() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());

        // Create an existing node
        let node_id = store.create_node_with_props(
            &["Person"],
            &[(PropertyKey::new("name"), Value::String("Jules".into()))],
        );

        // MERGE with ON MATCH SET
        let mut merge = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: const_props(vec![("name", Value::String("Jules".into()))]),
                on_create_properties: vec![],
                on_match_properties: const_props(vec![("updated", Value::Bool(true))]),
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );

        let _ = merge.next().unwrap();

        // Verify node has the on_match property added
        let node = store.get_node(node_id).unwrap();
        assert_eq!(
            node.properties.get(&PropertyKey::new("updated")),
            Some(&Value::Bool(true))
        );
    }

    #[test]
    fn test_merge_into_any() {
        let store: Arc<dyn GraphStoreMut> = Arc::new(LpgStore::new().unwrap());
        let op = MergeOperator::new(
            Arc::clone(&store),
            None,
            MergeConfig {
                variable: "n".to_string(),
                labels: vec!["Person".to_string()],
                match_properties: vec![],
                on_create_properties: vec![],
                on_match_properties: vec![],
                output_schema: vec![LogicalType::Node],
                output_column: 0,
                bound_variable_column: None,
            },
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<MergeOperator>().is_ok());
    }
}
