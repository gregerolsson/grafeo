//! End-to-end checks for the `id(var) = lit` / `id(var) IN [...]` pushdown.
//!
//! The unit tests in the optimizer module already cover the plan rewrite. These
//! tests run the full query pipeline (translator -> optimizer -> executor) and
//! confirm that:
//!   * EXPLAIN shows the Filter has been folded into the NodeScan (`[id=...]`).
//!   * The query returns the same rows it would have via the slow Filter path.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Returns (db, alix_id, gus_id, animal_id) — looked up by name so the test
/// doesn't depend on how the store assigns NodeIds at insertion time.
fn seed() -> (GrafeoDB, i64, i64, i64) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    for name in ["Alix", "Gus", "Vincent"] {
        session
            .create_node_with_props(&["Person"], [("name", Value::String(name.into()))])
            .unwrap();
    }
    session
        .create_node_with_props(&["Animal"], [("name", Value::String("Rufus".into()))])
        .unwrap();

    let id_of = |label: &str, name: &str| -> i64 {
        let q = format!("MATCH (n:{label}) WHERE n.name = '{name}' RETURN id(n)");
        let r = session.execute(&q).unwrap();
        match &r.rows()[0][0] {
            Value::Int64(i) => *i,
            other => panic!("expected Int64 id, got {other:?}"),
        }
    };

    (db, id_of("Person", "Alix"), id_of("Person", "Gus"), id_of("Animal", "Rufus"))
}

#[test]
fn explain_shows_filter_folded_into_scan() {
    let (db, alix, _gus, _animal) = seed();
    let session = db.session();

    let q = format!("EXPLAIN MATCH (p:Person) WHERE id(p) = {alix} RETURN p");
    let result = session.execute(&q).unwrap();
    let plan = match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("expected EXPLAIN String row, got {other:?}"),
    };

    assert!(
        plan.contains(&format!("[id={alix}]")),
        "expected pinned-id marker in EXPLAIN, got:\n{plan}"
    );
    // The Filter should be gone from the tree.
    assert!(
        !plan.contains("Filter"),
        "Filter should be folded into NodeScan, but EXPLAIN still has one:\n{plan}"
    );
}

#[test]
fn id_eq_literal_returns_single_matching_row() {
    let (db, alix, _gus, _animal) = seed();
    let session = db.session();

    let q = format!("MATCH (p:Person) WHERE id(p) = {alix} RETURN p.name");
    let result = session.execute(&q).unwrap();
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
}

#[test]
fn id_in_list_returns_matching_rows() {
    let (db, alix, gus, _animal) = seed();
    let q = format!("MATCH (p:Person) WHERE id(p) IN [{alix}, {gus}] RETURN p.name");
    let session = db.session();
    let result = session.execute(&q).unwrap();
    assert_eq!(result.row_count(), 2);
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => panic!("expected String, got {other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(names, vec!["Alix".to_string(), "Gus".to_string()]);
}

#[test]
fn id_eq_with_wrong_label_returns_zero_rows() {
    // The Animal node's id is queried for :Person — pinned scan must still
    // enforce the label filter.
    let (db, _alix, _gus, animal_id) = seed();
    let session = db.session();

    let q = format!("MATCH (p:Person) WHERE id(p) = {animal_id} RETURN p.name");
    let result = session.execute(&q).unwrap();
    assert_eq!(result.row_count(), 0);
}

#[test]
fn id_eq_nonexistent_id_returns_zero_rows() {
    let (db, _, _, _) = seed();
    let session = db.session();

    let result = session
        .execute("MATCH (p:Person) WHERE id(p) = 99999 RETURN p.name")
        .unwrap();
    assert_eq!(result.row_count(), 0);
}

#[test]
fn id_eq_with_residual_predicate_still_filters() {
    // `id(p) = <alix> AND p.name = 'Gus'` — id matches Alix but name doesn't.
    // The pinned scan should produce 1 row, the residual Filter should drop it.
    let (db, alix, _gus, _animal) = seed();
    let session = db.session();

    // Control: same shape but a satisfiable predicate — should return Alix.
    let consistent = session
        .execute(&format!(
            "MATCH (p:Person) WHERE id(p) = {alix} AND p.name = 'Alix' RETURN p.name"
        ))
        .unwrap();
    assert_eq!(consistent.row_count(), 1);
    assert_eq!(consistent.rows()[0][0], Value::String("Alix".into()));

    let q = format!("MATCH (p:Person) WHERE id(p) = {alix} AND p.name = 'Gus' RETURN p.name");
    let result = session.execute(&q).unwrap();
    assert_eq!(result.row_count(), 0);
}

#[test]
fn id_eq_via_parameter_is_also_pushed_down() {
    // Parameters are substituted to literals before the optimizer runs, so this
    // should take the same fast path. Verify via EXPLAIN.
    let (db, alix, _gus, _animal) = seed();
    let session = db.session();

    let q = "EXPLAIN MATCH (p:Person) WHERE id(p) = $pid RETURN p";
    let mut params: std::collections::HashMap<String, Value> = std::collections::HashMap::new();
    params.insert("pid".to_string(), Value::Int64(alix));
    let result = session.execute_with_params(q, params).unwrap();
    let plan = match &result.rows()[0][0] {
        Value::String(s) => s.to_string(),
        other => panic!("expected EXPLAIN row, got {other:?}"),
    };
    assert!(
        plan.contains(&format!("[id={alix}]")),
        "parameter-bound id() should be pinned after substitution; plan:\n{plan}"
    );
}
