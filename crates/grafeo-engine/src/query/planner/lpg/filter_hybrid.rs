//! Hybrid (text + vector) predicate pushdown into index scan operators.
//!
//! Extracted from filter.rs to keep that file focused on general filter planning.
//! All functions here are `impl super::Planner` methods; Rust merges impl blocks
//! within the same module, so they share visibility with the rest of the planner.

#[cfg(feature = "text-index")]
use super::{
    Arc, BinaryOp, ExpressionPredicate, FilterOp, FilterOperator, GraphStore, HashMap,
    LogicalExpression, LogicalOperator, Operator, Result, Value,
};

#[cfg(all(feature = "vector-index", feature = "text-index"))]
use super::{HashJoinOperator, PhysicalJoinType};

// ============================================================================
// Text predicate extraction and pushdown
// ============================================================================

/// Extracted text predicate from a filter expression.
#[cfg(feature = "text-index")]
pub(super) struct ExtractedTextPredicate {
    pub(super) property: String,
    /// Variable bound in the property access (e.g. `n` in `n.body`).
    /// Validated against the enclosing NodeScan variable before pushdown.
    pub(super) variable: String,
    pub(super) query_expr: LogicalExpression,
    pub(super) threshold: f64,
    pub(super) remaining: Option<LogicalExpression>,
}

#[cfg(feature = "text-index")]
impl super::Planner {
    /// Tries to push a text search predicate down into a `TextScan` operator.
    ///
    /// Recognizes patterns like:
    /// - `text_score(n.body, "search terms") > 0.5`
    /// - `text_match(n.body, "search terms")`  (standalone boolean)
    ///
    /// Falls through to per-row evaluation when the text index is absent (D1).
    ///
    /// Returns `Ok(Some(...))` if rewritten, `Ok(None)` to fall through.
    pub(super) fn try_plan_filter_with_text_index(
        &self,
        filter: &super::FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        // Only push down when input is a full label scan (no nested input)
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        let Some(ref label) = scan.label else {
            return Ok(None);
        };

        // Extract a text predicate from the filter expression
        let Some(extracted) = self.extract_text_predicate(&filter.predicate) else {
            return Ok(None);
        };

        // Ensure the predicate references the same variable as the scan being planned
        if extracted.variable != scan.variable {
            return Ok(None);
        }

        // No text index: fall through to per-row evaluation, same as vector behavior.
        // text_score returns 0.0 and text_match returns false for every row.
        if !self.store.has_text_index(label, &extracted.property) {
            return Ok(None);
        }

        // Build TextScanOp. Always project the score column so downstream
        // RETURN/ORDER BY expressions can reference it instead of recomputing.
        let text_scan = super::TextScanOp {
            variable: scan.variable.clone(),
            property: extracted.property.clone(),
            label: label.clone(),
            query: extracted.query_expr.clone(),
            k: None,
            threshold: Some(extracted.threshold),
            score_column: Some(super::project::text_score_column_name(
                &scan.variable,
                &extracted.property,
                &extracted.query_expr,
            )),
        };

        // Plan through the TextScan path
        let (scan_op, scan_columns) = self.plan_operator(&LogicalOperator::TextScan(text_scan))?;

        // If there are remaining predicates (AND with non-text conditions), wrap in filter
        if let Some(remaining) = &extracted.remaining {
            let variable_columns: HashMap<String, usize> = scan_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            let filter_op = Box::new(FilterOperator::new(scan_op, Box::new(predicate)));
            Ok(Some((filter_op, scan_columns)))
        } else {
            Ok(Some((scan_op, scan_columns)))
        }
    }

    /// Recursively extracts a text predicate from a (potentially compound) expression.
    pub(super) fn extract_text_predicate(
        &self,
        expr: &LogicalExpression,
    ) -> Option<ExtractedTextPredicate> {
        match expr {
            LogicalExpression::Binary { left, op, right } => {
                // text_score(n.prop, "query") > threshold
                if let LogicalExpression::FunctionCall { name, args, .. } = left.as_ref()
                    && name == "text_score"
                    && matches!(op, BinaryOp::Gt | BinaryOp::Ge)
                    && let Some(extracted) = self.try_extract_text_fn(args, right)
                {
                    return Some(extracted);
                }
                // AND: recurse into both sides, accumulating remaining predicates
                if *op == BinaryOp::And {
                    // Try left as text, right as remaining
                    if let Some(mut extracted) = self.extract_text_predicate(left) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: Box::new(prev),
                                op: BinaryOp::And,
                                right: right.clone(),
                            },
                            None => *right.clone(),
                        });
                        return Some(extracted);
                    }
                    // Try right as text, left as remaining
                    if let Some(mut extracted) = self.extract_text_predicate(right) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: left.clone(),
                                op: BinaryOp::And,
                                right: Box::new(prev),
                            },
                            None => *left.clone(),
                        });
                        return Some(extracted);
                    }
                }
                None
            }
            // text_match(n.prop, "query") — standalone boolean (score > 0.0)
            LogicalExpression::FunctionCall { name, args, .. } if name == "text_match" => {
                self.try_extract_text_fn(args, &LogicalExpression::Literal(Value::Float64(0.0)))
            }
            _ => None,
        }
    }

    /// Tries to extract the property, variable, query expression, and threshold
    /// from a `text_score` or `text_match` argument list.
    fn try_extract_text_fn(
        &self,
        args: &[LogicalExpression],
        threshold_expr: &LogicalExpression,
    ) -> Option<ExtractedTextPredicate> {
        if args.len() != 2 {
            return None;
        }

        let LogicalExpression::Property { variable, property } = &args[0] else {
            return None;
        };

        let threshold = match threshold_expr {
            LogicalExpression::Literal(Value::Float64(v)) => *v,
            LogicalExpression::Literal(Value::Int64(v)) => *v as f64,
            _ => return None,
        };

        Some(ExtractedTextPredicate {
            property: property.clone(),
            variable: variable.clone(),
            query_expr: args[1].clone(),
            threshold,
            remaining: None,
        })
    }
}

// ============================================================================
// Compound hybrid (vector AND/OR text) pushdown
// ============================================================================

#[cfg(all(feature = "vector-index", feature = "text-index"))]
impl super::Planner {
    /// Tries to plan a filter that contains BOTH a vector predicate and a text predicate.
    ///
    /// When both predicates are present, runs both index scans independently and
    /// hash-joins the results: intersect (Inner join) for AND, union (Full join) for OR.
    /// Both scores are projected as columns for downstream use.
    ///
    /// For AND: extractors recurse into the AND tree (finding vector/text anywhere).
    /// For OR: extracts from each OR operand independently, requiring each side
    /// to be a pure vector or text predicate (no mixed scalar conditions).
    ///
    /// Falls through when only one predicate is present or an index is missing (D1).
    pub(super) fn try_plan_filter_compound_hybrid(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return Ok(None);
        };
        let Some(ref label) = scan.label else {
            return Ok(None);
        };

        // Extract vector and text predicates. Strategy depends on AND vs OR.
        let (vector_pred, text_pred, is_or) = if let LogicalExpression::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } = &filter.predicate
        {
            // OR: extract from each operand independently.
            // Try vector-left + text-right, then vector-right + text-left.
            let result = self
                .extract_vector_predicate(left)
                .and_then(|v| self.extract_text_predicate(right).map(|t| (v, t)))
                .or_else(|| {
                    self.extract_vector_predicate(right)
                        .and_then(|v| self.extract_text_predicate(left).map(|t| (v, t)))
                });
            let Some((v, t)) = result else {
                return Ok(None);
            };
            (v, t, true)
        } else {
            // AND: extractors recurse into the AND tree (existing behavior).
            let vector = self.extract_vector_predicate(&filter.predicate);
            let text = self.extract_text_predicate(&filter.predicate);
            match (vector, text) {
                (Some(v), Some(t)) => (v, t, false),
                _ => return Ok(None),
            }
        };

        // Validate both reference the scan variable
        if vector_pred.variable != scan.variable || text_pred.variable != scan.variable {
            return Ok(None);
        }

        // If either index is missing, fall through to per-row evaluation (D1).
        if !self.store.has_vector_index(label, &vector_pred.property) {
            return Ok(None);
        }
        if !self.store.has_text_index(label, &text_pred.property) {
            return Ok(None);
        }

        // Pushdown needs a resolvable query vector. Fall through otherwise.
        if self
            .resolve_vector_literal(&vector_pred.query_vector)
            .is_err()
        {
            return Ok(None);
        }

        // Build VectorScanOp (threshold mode, return all candidates above threshold).
        let vector_scan_op = LogicalOperator::VectorScan(super::VectorScanOp {
            variable: scan.variable.clone(),
            index_name: Some(format!("{}:{}", label, vector_pred.property)),
            property: vector_pred.property.clone(),
            label: Some(label.clone()),
            query_vector: vector_pred.query_vector.clone(),
            k: None, // threshold mode
            metric: Some(vector_pred.metric),
            min_similarity: vector_pred.min_similarity,
            max_distance: vector_pred.max_distance,
            input: None,
        });

        // Build TextScanOp (threshold mode, always project score column).
        let text_scan_op = LogicalOperator::TextScan(super::TextScanOp {
            variable: scan.variable.clone(),
            property: text_pred.property.clone(),
            label: label.clone(),
            query: text_pred.query_expr.clone(),
            k: None,
            threshold: Some(text_pred.threshold),
            score_column: Some(super::project::text_score_column_name(
                &scan.variable,
                &text_pred.property,
                &text_pred.query_expr,
            )),
        });

        let (left_op, left_cols) = self.plan_operator(&vector_scan_op)?;
        let (right_op, right_cols) = self.plan_operator(&text_scan_op)?;

        // For OR with scalar remainders on either branch, wrap that branch
        // in a filter before the join. E.g., for
        //   cosine_similarity(...) > 0.8 OR (text_match(...) AND published = true)
        // the text branch gets a Filter(published = true) around its TextScan.
        let left_op = if let Some(remaining) = &vector_pred.remaining {
            let variable_columns: HashMap<String, usize> = left_cols
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            Box::new(FilterOperator::new(left_op, Box::new(predicate))) as Box<dyn Operator>
        } else {
            left_op
        };

        let right_op = if let Some(remaining) = &text_pred.remaining {
            let variable_columns: HashMap<String, usize> = right_cols
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            Box::new(FilterOperator::new(right_op, Box::new(predicate))) as Box<dyn Operator>
        } else {
            right_op
        };

        // Determine join type: AND → Inner (intersect), OR → Full (union)
        let join_type = if is_or {
            PhysicalJoinType::Full
        } else {
            PhysicalJoinType::Inner
        };

        // HashJoin outputs all left columns + all right columns.
        // Right side: [variable, _tscore_variable]. The variable column is a duplicate
        // of left column 0 and must be projected out.
        let mut all_cols = left_cols.clone();
        all_cols.extend(right_cols.iter().cloned());
        let all_schema = self.derive_schema_from_columns(&all_cols);

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            vec![0], // probe key: column 0 (NodeId / variable)
            vec![0], // build key: column 0 (NodeId / variable)
            join_type,
            all_schema,
        ));

        // Project out the duplicate node-variable column from the right side.
        // left_cols = [variable, _vscore_variable]  (indices 0, 1)
        // right_cols = [variable, _tscore_variable]  (indices 2, 3 in all_cols)
        // Output: [variable, _vscore_variable, _tscore_variable]
        //
        // For OR (Full join): right-only rows have NULL in left column 0.
        // Use COALESCE(left_var, right_var) so the variable is always non-NULL.
        let left_count = left_cols.len();
        let mut proj_exprs: Vec<super::ProjectExpr> = Vec::new();
        let mut output_cols: Vec<String> = Vec::new();

        for (i, col) in left_cols.iter().enumerate() {
            if i == 0 && is_or {
                // For OR joins, coalesce the variable from both sides
                proj_exprs.push(super::ProjectExpr::Coalesce {
                    first: 0,
                    second: left_count,
                });
            } else {
                proj_exprs.push(super::ProjectExpr::Column(i));
            }
            output_cols.push(col.clone());
        }
        for (i, col) in right_cols.iter().enumerate() {
            if i == 0 {
                continue; // Duplicate variable column (merged via Coalesce above)
            }
            proj_exprs.push(super::ProjectExpr::Column(left_count + i));
            output_cols.push(col.clone());
        }

        let proj_schema = self.derive_schema_from_columns(&output_cols);
        let proj_op: Box<dyn Operator> = Box::new(super::ProjectOperator::new(
            join_op,
            proj_exprs,
            proj_schema,
        ));

        // Apply any remaining scalar predicates (parts of the expression that are
        // neither vector nor text, e.g. an extra AND condition)
        let scalar_remaining = self.extract_scalar_remaining(&filter.predicate);
        if let Some(remaining) = scalar_remaining {
            let variable_columns: HashMap<String, usize> = output_cols
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(&remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone());
            Ok(Some((
                Box::new(FilterOperator::new(proj_op, Box::new(predicate))),
                output_cols,
            )))
        } else {
            Ok(Some((proj_op, output_cols)))
        }
    }

    /// Extracts the parts of a predicate that are neither a vector nor a text sub-predicate.
    ///
    /// Recursively walks AND trees, keeping only scalar (non-index) conditions.
    /// Used to find conditions that must be applied after the hash join.
    pub(super) fn extract_scalar_remaining(
        &self,
        expr: &LogicalExpression,
    ) -> Option<LogicalExpression> {
        match expr {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                let left_scalar = self.extract_scalar_remaining(left);
                let right_scalar = self.extract_scalar_remaining(right);

                match (left_scalar, right_scalar) {
                    (Some(l), Some(r)) => Some(LogicalExpression::Binary {
                        left: Box::new(l),
                        op: BinaryOp::And,
                        right: Box::new(r),
                    }),
                    (Some(l), None) => Some(l),
                    (None, Some(r)) => Some(r),
                    (None, None) => None,
                }
            }
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Or,
                right,
            } => {
                // If both sides of OR are index predicates, the full-join
                // already computes the union — no scalar filter needed.
                let left_remaining = self.extract_scalar_remaining(left);
                let right_remaining = self.extract_scalar_remaining(right);
                match (left_remaining, right_remaining) {
                    (None, None) => None,
                    _ => Some(expr.clone()),
                }
            }
            // Leaf node: check if it's a vector or text predicate
            other => {
                let is_vector = self.extract_vector_predicate(other).is_some();
                let is_text = self.extract_text_predicate(other).is_some();
                if is_vector || is_text {
                    None // Handled by index scan, drop it
                } else {
                    Some(other.clone()) // Scalar predicate, keep it
                }
            }
        }
    }
}
